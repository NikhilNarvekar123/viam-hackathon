"""Live camera -> CPU scene-change diagnostics. NEVER moves or publishes events.

Polls every 2 seconds and automatically rearms after confirmed changes.
Keep the robot stationary. This does NOT automatically identify arm/hand occlusion.
Optional --manual mode: o=obscured, c=clear, r=new reference, q=quit (+ ENTER).
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import select
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from monitors.overview_detection import DetectionSettings, OverviewChangeDetector


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--fake', action='store_true', help='Synthetic frames; no hardware/credentials')
    result.add_argument('--manual', action='store_true', help='Opt-in interactive reference/occlusion controls')
    result.add_argument('--camera', default='overview-cam')
    result.add_argument('--config', type=Path, default=ROOT / 'config/config.yaml')
    result.add_argument('--width', type=int, default=640, help='Maximum processing width; never upscale')
    result.add_argument('--hz', type=float, default=0.5, help='Polling rate (0, 10]; default 0.5 = every 2 seconds')
    result.add_argument('--seconds', type=float, default=120, help='Maximum monitoring duration')
    result.add_argument('--timeout', type=float, default=10, help='Connection/capture timeout; no retry')
    result.add_argument('--threshold', type=float, default=25)
    result.add_argument('--min-area', type=int, default=12, help='Connected changed area in processing pixels')
    result.add_argument('--persistence', type=int, default=3, help='Consecutive clear samples')
    result.add_argument('--roi', nargs=4, type=int, metavar=('X0','Y0','X1','Y1'),
                        help='Workspace rectangle in RESIZED processing-image pixels')
    result.add_argument('--exclude', nargs=4, type=int, action='append', default=[],
                        metavar=('X0','Y0','X1','Y1'), help='Exclude base/etc.; repeatable, processing pixels')
    result.add_argument('--record', action='store_true', help='Save every frame and an offline replay manifest')
    result.add_argument('--out', type=Path, default=ROOT / 'logs/overview_changes')
    return result


def validate(args):
    for name in ('hz', 'seconds', 'timeout'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if args.hz > 10 or args.width < 1:
        raise ValueError('hz must be <= 10 and width must be positive')
    return DetectionSettings(difference_threshold=args.threshold, min_area_px=args.min_area,
                             persistence_frames=args.persistence)


def prepare(image, width):
    image = image.convert('RGB')
    if image.width > width:
        image = image.resize((width, max(1, round(image.height * width / image.width))), Image.Resampling.BOX)
    return image


def masks(size, roi, excluded):
    width, height = size
    def rectangle(bounds):
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'Rectangle must fit processing image {width}x{height}: {bounds}')
        result = np.zeros((height, width), dtype=bool)
        result[y0:y1, x0:x1] = True
        return result
    workspace = rectangle(roi or (0, 0, width, height))
    exclusion = np.zeros_like(workspace)
    for box in excluded:
        exclusion |= rectangle(box)
    return workspace, exclusion


class SyntheticSource:
    def __init__(self):
        self.index = 0

    async def image(self):
        frame = Image.new('RGB', (320, 240), (160, 160, 160))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 180, 50, 239), fill=(40, 40, 40))  # stationary base
        if self.index >= 3:
            draw.rectangle((140, 90, 160, 110), fill=(20, 20, 20))
        self.index += 1
        return frame


@asynccontextmanager
async def source(args):
    if args.fake:
        yield SyntheticSource()
        return
    # Import ONLY the camera-only helper in live mode. Never construct arm,
    # gripper, motion, teleop, models, or an EventRuntime.
    import yaml
    from robot.camera import capture_camera
    config = yaml.safe_load(args.config.read_text())
    config['hardware']['viam']['camera'] = args.camera
    manager = capture_camera(config, ROOT, False)
    camera = await asyncio.wait_for(manager.__aenter__(), args.timeout)
    class LiveSource:
        async def image(self):
            return (await camera.get_frame()).image
    try:
        yield LiveSource()
    finally:
        await asyncio.wait_for(manager.__aexit__(None, None, None), args.timeout)


def terminal_command():
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    line = sys.stdin.readline()
    return 'q' if line == '' else line.strip().lower()


async def watch(args, camera, command=None):
    settings = validate(args)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = args.out / ('watch_' + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    print(f'Diagnostics only; NO robot events/actions. Output: {folder}', flush=True)
    print(f'Polling every {1 / args.hz:g} seconds; {args.persistence} consecutive changed samples to confirm.', flush=True)
    if args.manual:
        print('Commands + ENTER: o=obscured, c=clear, r=accept fresh reference, q=quit.', flush=True)
    else:
        print('Automatic reference/rearm; no keyboard input required. Ctrl+C to quit.', flush=True)
    print('No automatic arm/hand detection. Unmarked occlusion can cause false changes.', flush=True)
    detector, clear, reference_pending = None, True, True
    samples = []
    manifest = {'settings': asdict(settings), 'workspace_mask': 'workspace.png',
                'excluded_mask': 'excluded.png', 'samples': samples}
    started = time.monotonic()
    previous_status, index = None, 0
    with (folder / 'detections.jsonl').open('x') as log:
        try:
            while time.monotonic() - started < args.seconds:
                tick = time.monotonic()
                cmd = command() if command else None
                if cmd == 'q':
                    break
                if cmd == 'o':
                    clear = False
                    reference_pending = False  # Never silently rebaseline when clearing occlusion.
                    print('OBSCURED: preserving reference; comparison suspended.', flush=True)
                elif cmd == 'c':
                    clear = True
                    print('CLEAR: comparing against the preserved reference.', flush=True)
                elif cmd == 'r':
                    if clear:
                        reference_pending = True
                    else:
                        print('Reference refused while obscured. Use c after the view is clear.', flush=True)
                elif cmd not in (None, '', 'q'):
                    print('Unknown command. Use o, c, r, or q + ENTER.', flush=True)
                # One capture in flight, fail on timeout; no blind retry.
                image = prepare(await asyncio.wait_for(camera.image(), args.timeout), args.width)
                rgb = np.asarray(image)
                if detector is None:
                    workspace, exclusion = masks(image.size, args.roi, args.exclude)
                    detector = OverviewChangeDetector(workspace, exclusion, settings)
                    if not clear:
                        raise ValueError('Initial reference must be clear')
                    reference_pending = True
                    Image.fromarray(workspace.astype(np.uint8) * 255).save(folder / 'workspace.png')
                    Image.fromarray(exclusion.astype(np.uint8) * 255).save(folder / 'excluded.png')
                    print(f'Processing {image.width}x{image.height}; ROI/exclusions use these pixels.', flush=True)
                operation = 'reference' if reference_pending else 'observe'
                auto_rearmed = False
                if reference_pending:
                    detector.set_reference(rgb, clear_view=clear)
                    reference_pending = False
                    record = {'status': 'reference', 'boxes_xyxy': [], 'would_publish': None}
                    image.save(folder / f'reference_{index:06d}.png')
                    print('REFERENCE accepted. Add, remove, or move an object, then withdraw hands.', flush=True)
                else:
                    result = detector.observe(rgb, clear_view=clear)
                    record = asdict(result)
                    record['would_publish'] = 'scene_changed' if result.publish_scene_changed else None
                    if result.publish_scene_changed:
                        suffix = ('Clear hands then r + ENTER to rearm.' if args.manual
                                  else 'Automatically rearming and continuing.')
                        print(f'SCENE_CHANGED (diagnostic only): {result.reason}; '
                              f'boxes={result.boxes_xyxy}. {suffix}', flush=True)
                        image.save(folder / f'change_{index:06d}.png')
                        if not args.manual:
                            # Diagnostic mode ONLY: acknowledge the exact observed
                            # frame, not a later capture that could swallow changes.
                            # This is NOT the production controller reset handshake.
                            detector.set_reference(rgb, clear_view=clear)
                            image.save(folder / f'reference_{index:06d}.png')
                            auto_rearmed = True
                if record['status'] != previous_status:
                    print(f"[{index:06d}] {record['status']}", flush=True)
                    previous_status = record['status']
                record.update(sample=index, clear_view=clear, operation=operation, fake=args.fake,
                              auto_rearmed=auto_rearmed, elapsed_s=time.monotonic() - started,
                              capture_and_process_ms=(time.monotonic() - tick) * 1000)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                for x0, y0, x1, y1 in record['boxes_xyxy']:
                    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline='red', width=2)
                draw.text((5, 5), record['status'], fill='red')
                annotated.save(folder / 'latest.png')
                if args.record:
                    filename = f'frame_{index:06d}.png'
                    image.save(folder / filename)
                    samples.append({'image': filename, 'clear_view': clear, 'operation': operation})
                    if auto_rearmed:
                        samples.append({'image': filename, 'clear_view': clear, 'operation': 'reference'})
                index += 1
                await asyncio.sleep(max(0, 1 / args.hz - (time.monotonic() - tick)))
        finally:
            if args.record:
                (folder / 'replay.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return folder


async def run(args):
    async with source(args) as camera:
        return await watch(args, camera, terminal_command if args.manual and not args.fake else None)


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        validate(args)
        if not args.fake:
            print('Camera-only diagnostic. Keep the arm stationary; stop other picking/teleop controllers.')
            print('Leave the workspace visible, withdraw hands, and keep the camera fixed.')
            if args.manual:
                if not sys.stdin.isatty():
                    raise ValueError('--manual requires an interactive terminal (SSH: ssh -t viam-nuc)')
                if input('ENTER to capture the initial reference, or q + ENTER to cancel: ').strip():
                    return
        asyncio.run(run(args))
    except KeyboardInterrupt:
        cli.exit(130, 'Stopped camera diagnostics. No robot commands issued.\n')
    except Exception as exc:
        # SDK errors can contain machine details; keep terminal error output bounded.
        cli.exit(1, f'Camera diagnostic failed ({type(exc).__name__}). No automatic retry.\n')


if __name__ == '__main__':
    main()
