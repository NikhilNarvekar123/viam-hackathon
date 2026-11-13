"""Capture wrist-camera object crops and run local anomaly inference. NEVER moves hardware."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import yaml

from anomaly.detector import OpenVINODetector
from anomaly.object_crops import automatic_crops
from robot.locate import annotate
from core.types import Frame
from robot.camera import capture_camera


def validate_rois(image, rois):
    if not rois:
        raise ValueError("Specify --roi L T R B, configure rois, or explicitly use --whole-frame")
    for box in rois:
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Each ROI must contain four integer pixel coordinates")
        left, top, right, bottom = box
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError(f"ROI {box} is outside image size {image.size}")
    return rois


async def capture(settings, fake):
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["hardware"]["viam"]["camera"] = settings["camera_name"]
    async with capture_camera(config, ROOT, fake) as camera:
        return (await camera.get_frame()).image


def inspect(image, rois, detector, threshold, folder):
    """One model call per ROI. Color scale is per crop, NOT a decision threshold."""
    validate_rois(image, rois)
    records = []
    for index, box in enumerate(rois, 1):
        crop = image.crop(box)
        started = time.perf_counter()
        detection = detector.detect(Frame(str(index), crop))
        elapsed = (time.perf_counter() - started) * 1000
        prefix = folder / f"object_{index:03d}"
        crop.save(str(prefix) + ".png")
        record = dict(id=index, bbox_xyxy=box, score=detection.score,
                      threshold=threshold, inference_ms=elapsed,
                      classification="unclassified" if threshold is None else
                      ("anomalous" if detection.score >= threshold else "normal"))
        if detection.heatmap is not None:
            heatmap = detection.heatmap
            np.save(str(prefix) + "_map.npy", heatmap)
            lo, hi = float(heatmap.min()), float(heatmap.max())
            normalized = (heatmap - lo) / (hi - lo) if hi > lo else np.zeros_like(heatmap)
            color = np.stack([normalized, np.zeros_like(normalized), 1 - normalized], axis=-1)
            rendered = Image.fromarray(np.uint8(np.clip(color, 0, 1) * 255)).resize(crop.size)
            rendered.save(str(prefix) + "_heatmap.png")
            Image.blend(crop, rendered, 0.45).save(str(prefix) + "_overlay.png")
            record.update(heatmap_min=lo, heatmap_max=hi, heatmap_display="per-crop min/max; blue low, red high")
        records.append(record)
        print(f"Object {index}: score={detection.score:.6f}, {record['classification']}, {elapsed:.1f} ms")
    return records


def annotate_scores(image, records):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    for record in records:
        left, top, right, bottom = record["bbox_xyxy"]
        label = f"#{record['id']} {record['score']:.3f} {record['classification']}"
        color = {"anomalous": "red", "normal": "lime"}.get(record["classification"], "yellow")
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=2)
        bounds = draw.textbbox((0, 0), label)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(left, image.width - width - 4))
        y = max(0, top - height - 6)
        draw.rectangle((x, y, x + width + 4, y + height + 4), fill="black")
        draw.text((x + 2 - bounds[0], y + 2 - bounds[1]), label, fill=color)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--capture", action="store_true")
    source.add_argument("--image", type=Path)
    source.add_argument("--fake", action="store_true", help="Fake camera; still uses real OpenVINO model unless --preview")
    parser.add_argument("--preview", action="store_true", help="Save camera frame without loading the model")
    parser.add_argument("--camera", help="Override camera resource name (e.g. overview-cam for preview)")
    regions = parser.add_mutually_exclusive_group()
    regions.add_argument("--roi", nargs=4, type=int, action="append", metavar=("L", "T", "R", "B"))
    regions.add_argument("--whole-frame", action="store_true", help="Explicitly treat full image as ONE object crop")
    parser.add_argument("--config", type=Path, default=ROOT / "config/wrist_inspection.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/wrist_inspection")
    args = parser.parse_args()
    try:
        settings = yaml.safe_load(args.config.read_text())
        if args.camera:
            settings["camera_name"] = args.camera
        threshold = settings["model"].get("threshold")
        if threshold is not None:
            threshold = float(threshold)
            if not math.isfinite(threshold):
                raise ValueError("Threshold must be finite or null")
        if args.image:
            with Image.open(args.image) as source_image:
                image = source_image.convert("RGB")
        else:
            print(f"Capturing {'FAKE' if args.fake else settings['camera_name']} (NO arm/gripper commands)")
            image = asyncio.run(capture(settings, args.fake))
        stamp = datetime.now(timezone.utc)
        folder = args.out / stamp.strftime("inspect_%Y%m%dT%H%M%S_%fZ")
        folder.mkdir(parents=True, exist_ok=False)
        image.save(folder / "frame.png")
        print(f"Frame: {folder / 'frame.png'} ({image.width}x{image.height})")
        if args.preview:
            return
        rois = [[0, 0, image.width, image.height]] if args.whole_frame else (args.roi or settings.get("rois"))
        localization = None
        if not rois:
            rois, localization, mask = automatic_crops(image, settings["auto_crop"])
            annotate(image, localization, mask).save(folder / "detected_objects.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "foreground_mask.png")
            (folder / "candidates.json").write_text(json.dumps(localization, indent=2, allow_nan=False) + "\n")
            print(f"Auto-crop: {len(rois)} candidates. Review {folder / 'detected_objects.png'}")
            print("Heuristic crops: shadows/clutter may be detected; touching objects may merge.")
            if not rois:
                print("No usable candidates; no inference performed. Adjust view or crop settings.")
                return
        validate_rois(image, rois)
        model_settings = dict(settings["model"])
        # Adapter requires a threshold; its boolean is ignored when threshold is unset.
        model_settings["threshold"] = threshold if threshold is not None else 0.0
        print("Preprocessing is provisional. Confirm with trainer; scores are not probabilities.")
        detector = OpenVINODetector(model_settings, ROOT)
        records = inspect(image, rois, detector, threshold, folder)
        scored = annotate_scores(image, records)
        scored.save(folder / "scored_objects.png")
        if localization is not None:
            scored.save(folder / "detected_objects.png")
        print(f"Scores on image: {folder / 'scored_objects.png'}")
        report = dict(timestamp=stamp.isoformat(), source=str(args.image) if args.image else
                      ("fake" if args.fake else settings["camera_name"]), settings=settings,
                      device=detector.device, objects=records, localization=localization)
        (folder / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Results and overlays: {folder}")
    except KeyboardInterrupt:
        parser.exit(130, "Inspection cancelled.\n")
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Inspection error: {exc}\n")


if __name__ == "__main__":
    main()
