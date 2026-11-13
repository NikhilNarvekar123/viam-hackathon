"""Save ROI-cropped training images; --preview saves one full/crop pair."""
import argparse
import asyncio
from contextlib import ExitStack
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from robot.camera import capture_camera
from robot.viam_hw import crop_image


def save_images(out: Path, images: dict, prefix="image") -> list[Path]:
    """Reserve a numbered image/pair exclusively; never overwrite old photos."""
    out.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        paths = [out / f"{prefix}_{index:06d}{suffix}.png" for suffix in images]
        created = []
        try:
            with ExitStack() as stack:
                streams = []
                for path in paths:
                    streams.append(stack.enter_context(path.open("xb")))
                    created.append(path)
                for image, stream in zip(images.values(), streams):
                    image.save(stream, format="PNG")
            return paths
        except FileExistsError:
            for path in created:
                path.unlink()
            index += 1
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise


async def wait_for_enter(prompt):
    """Cancellable stdin wait on the target Linux/macOS event loop."""
    print(prompt, end="", flush=True)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    def ready():
        if future.done():
            return
        line = sys.stdin.readline()
        if line == "":
            future.set_exception(EOFError("stdin closed; use --interval for unattended capture"))
        else:
            future.set_result(None)
    try:
        loop.add_reader(sys.stdin.fileno(), ready)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise ValueError("Interactive stdin unavailable; use --interval") from exc
    try:
        await future
    finally:
        loop.remove_reader(sys.stdin.fileno())


async def capture(camera, out: Path, roi, n=50, interval=None, preview=False,
                  wait=wait_for_enter, sleep=asyncio.sleep):
    if n < 1:
        raise ValueError("--n must be at least 1")
    if interval is not None and (not math.isfinite(interval) or interval < 0):
        raise ValueError("--interval must be finite and nonnegative")
    saved = []
    for index in range(1 if preview else n):
        if not preview:
            if interval is None:
                await wait(f"Photo {index + 1}/{n}: press Enter after positioning the item… ")
            elif index:
                await sleep(interval)  # Interval between captures; first is immediate.
        frame = await camera.get_frame()
        cropped = crop_image(frame.image, roi)
        if preview:
            paths = save_images(out, {"_uncropped": frame.image, "_cropped": cropped}, "preview")
        else:
            paths = save_images(out, {"": cropped})
        saved.extend(paths)
        for path in paths:
            print(f"Saved {path}")
    return saved


async def run(config, args):
    async with capture_camera(config, ROOT, args.fake) as camera:
        print("FAKE camera: synthetic/sample images, not training data from the station."
              if args.fake else "REAL camera: image capture only; no arm or gripper connection.")
        if args.preview:
            print("Preview only: saving one full/crop pair, not a training batch.")
        await capture(camera, args.out, config["camera"]["roi"], args.n,
                      args.interval, args.preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--interval", type=float,
                        help="Seconds between captures; 0 captures without waiting")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.interval is not None and (not math.isfinite(args.interval) or args.interval < 0):
        parser.error("--interval must be finite and nonnegative")
    try:
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        asyncio.run(run(config, args))
    except KeyboardInterrupt:
        parser.exit(130, "Capture interrupted; connection cleanup attempted.\n")
    except (ValueError, OSError, ImportError, KeyError, EOFError) as exc:
        parser.exit(1, f"Capture error: {exc}\n")


if __name__ == "__main__":
    main()
