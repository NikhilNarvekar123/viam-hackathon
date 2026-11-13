"""Overview camera -> object crops and pixel coordinates. NEVER moves hardware."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import yaml

from robot.camera import capture_camera
from robot.locate import annotate, locate_objects


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


async def capture_frame(config, settings, fake):
    # Independent camera-only helper. No arm/gripper resource is ever constructed.
    config["hardware"]["viam"]["camera"] = settings["camera_name"]
    async with capture_camera(config, ROOT, fake) as camera:
        return (await camera.get_frame()).image


def save_results(image, background, settings, output_root, camera_name):
    started = time.perf_counter()
    result, mask = locate_objects(image, background, settings)
    result["localization_ms"] = (time.perf_counter() - started) * 1000
    stamp = datetime.now(timezone.utc)
    result.update(timestamp=stamp.isoformat(), camera_name=camera_name, settings=settings)
    folder = output_root / stamp.strftime("locate_%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    image.save(folder / "frame.png")
    annotate(image, result, mask).save(folder / "annotated.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "foreground_mask.png")
    crops = folder / "crops"
    crops.mkdir()
    for obj in result["objects"]:
        relative = f"crops/object_{obj['id']:03d}.png"
        image.crop(obj["crop_bbox_xyxy"]).save(folder / relative)
        obj["crop_path"] = relative
    (folder / "objects.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Detected {len(result['objects'])} candidate objects in {result['localization_ms']:.1f} ms")
    for obj in result["objects"]:
        angle = obj["image_angle_deg"]
        text = "unknown/symmetric" if angle is None else f"{angle:.1f} deg"
        print(f"  {obj['id']}: center={obj['center_uv']}, angle={text}, bbox={obj['bbox_xyxy']}")
    print(f"Annotated image: {folder / 'annotated.png'}")
    print(f"Coordinates/crops: {folder / 'objects.json'}")
    print("PIXELS ONLY. Not calibrated robot targets. Review segmentation before using crops.")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Analyze an existing full overview frame")
    source.add_argument("--capture", action="store_true", help="Capture one full camera frame")
    parser.add_argument("--fake", action="store_true", help="Use fake camera with --capture")
    parser.add_argument("--background", type=Path, help="Empty-table image from exactly this viewpoint")
    parser.add_argument("--save-background", type=Path, help="Save this frame as reference, then exit")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/localization")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--locator-config", type=Path, default=ROOT / "config/localization.yaml")
    args = parser.parse_args()
    if bool(args.background) == bool(args.save_background):
        parser.error("Choose exactly one of --background or --save-background")
    if args.fake and not args.capture:
        parser.error("--fake applies only to --capture; --image is already offline")
    try:
        settings = yaml.safe_load(args.locator_config.read_text())
        if args.save_background and args.save_background.exists():
            raise ValueError("Reference already exists; choose a new filename to avoid overwriting")
        if args.capture:
            config = yaml.safe_load(args.config.read_text())
            print(f"Capturing {'FAKE camera' if args.fake else settings['camera_name']} (no arm commands)")
            image = asyncio.run(capture_frame(config, settings, args.fake))
        else:
            image = load_image(args.image)
        if args.save_background:
            args.save_background.parent.mkdir(parents=True, exist_ok=True)
            with args.save_background.open("xb") as stream:
                image.save(stream, format="PNG")
            print(f"Saved reference: {args.save_background} ({image.width}x{image.height})")
            print("This must show an EMPTY table. Keep camera position/exposure/lighting fixed.")
            return
        save_results(image, load_image(args.background), settings, args.out,
                     "fake" if args.fake else settings["camera_name"])
    except KeyboardInterrupt:
        parser.exit(130, "Capture cancelled.\n")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Localization error: {exc}\n")


if __name__ == "__main__":
    main()
