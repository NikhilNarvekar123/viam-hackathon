"""Inspect a single image with the configured decider; never connects to hardware."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import yaml

from core.factory import build_decider
from core.types import Detection, Frame
from robot.viam_hw import crop_image
from vlm.overlay import build_overlay
from vlm.prompt import build_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="Already-cropped item image, e.g. from tools/capture.py")
    parser.add_argument("--heatmap", type=Path, help="Optional 2D NumPy .npy heatmap")
    parser.add_argument("--uncropped", action="store_true", help="Apply config ROI to a raw image")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--save-overlay", type=Path, help="Optional new PNG output (no overwriting)")
    args = parser.parse_args()
    try:
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        if args.uncropped:
            if args.heatmap:
                raise ValueError("Use a cropped image with --heatmap; alignment must be explicit")
            if config["camera"]["roi"] is None:
                raise ValueError("--uncropped requires a configured ROI")
            image = crop_image(image, config["camera"]["roi"])
        heatmap = np.load(args.heatmap, allow_pickle=False) if args.heatmap else None
        settings = config["models"]["decider"]
        overlay = build_overlay(image, heatmap, **settings["overlay"])
        if args.save_overlay:
            args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
            with args.save_overlay.open("xb") as stream:
                overlay.save(stream, format="PNG")
        prompt = build_prompt(settings["prompt"], config["defect_types"], config["allowed_actions"])
        print("Prompt:\n" + prompt)
        decider = build_decider(config, ROOT)
        actual = getattr(decider, "implementation", "unknown")
        print(f"Decider: requested={settings.get('implementation', 'stub')}, actual={actual}")
        if actual != "openvino":
            print("No VLM inference: stub/rules mode; prompt and overlay shown for inspection only.")
        start = perf_counter()
        # This tool exercises the flagged-item decision path, not anomaly detection.
        decision = decider.decide(Frame(args.image.stem, image), Detection(0.0, True, heatmap))
        elapsed = (perf_counter() - start) * 1000
        print("Raw output: " + decision.raw_output)
        print("Parsed decision: " + json.dumps(asdict(decision)))
        print(f"Latency ms: {elapsed:.3f}")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"VLM test error: {exc}\n")


if __name__ == "__main__":
    main()
