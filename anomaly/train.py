"""Anomalib 2.x CPU train/export scaffolding; live API calls need verification.

Install CPU torch/torchvision from https://download.pytorch.org/whl/cpu before
installing the ML requirements on the NUC. This tool never installs packages.
Backbones may download weights: check download sizes before running training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image

from anomaly.detector import OpenVINODetector
from anomaly.preprocess import prepare_image
from anomaly.threshold import score_stats, suggest_threshold
from core.types import Frame

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def discover_images(data: Path) -> dict:
    groups = {label: sorted(path for path in (data / label).glob("*")
                            if path.is_file() and path.suffix.lower() in EXTENSIONS)
              for label in ("good", "defect")}
    if len(groups["good"]) < 2:
        raise ValueError("Need at least two good images for training/validation splitting")
    return groups


def stage_dataset(groups, destination, image_size, roi=None):
    """Preserve raw data. Both training and inference use identical PIL resizing."""
    for label, paths in groups.items():
        if not paths:
            continue
        folder = destination / label
        folder.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                prepare_image(image, image_size, roi).save(folder / f"{index:06d}.png")


def export_xml_path(exported) -> Path:
    if exported is None:
        raise ValueError("Export returned no path; VERIFY Engine.export API")
    path = Path(exported)
    if path.is_file() and path.suffix == ".xml":
        return path
    matches = list(path.rglob("*.xml")) if path.is_dir() else []
    if len(matches) != 1:
        raise ValueError("Expected exactly one exported OpenVINO XML; VERIFY export path")
    return matches[0]


def train(args, config):
    groups = discover_images(args.data)
    roi = config["camera"]["roi"] if args.images_uncropped else None
    if args.images_uncropped and roi is None:
        raise ValueError("--images-uncropped requires a configured camera ROI")
    # No heavy imports at module scope, and no CUDA training/builds.
    import torch
    if torch.version.cuda is not None:
        raise ValueError("CPU torch required; install CPU wheels from download.pytorch.org/whl/cpu")
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd, Padim, Patchcore

    model_type = {"patchcore": Patchcore, "padim": Padim, "efficientad": EfficientAd}[args.model]
    # VERIFY: configure_pre_processor(image_size=...) and pre_processor= on pinned 2.x.
    # VERIFY: exported preprocessor includes all model-specific normalization.
    # Staged RGB images already have the requested size, avoiding resize differences.
    preprocessor = model_type.configure_pre_processor(image_size=(args.image_size,) * 2)
    model = model_type(pre_processor=preprocessor)
    out = ROOT / "models/anomaly" / f"{args.data.resolve().name}_{args.model}"
    # Do not replace a previous calibration/model accidentally.
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="viam-training-") as directory:
        staged = Path(directory)
        stage_dataset(groups, staged, args.image_size, roi)
        # VERIFY: Folder split options and absence of defect/ support in pinned 2.x.
        # Only normal_dir is used to fit; defect examples are validation/test data.
        dataset = Folder(name=args.data.resolve().name, root=staged, normal_dir="good",
                         abnormal_dir="defect" if groups["defect"] else None,
                         normal_split_ratio=0.2, train_batch_size=1,
                         eval_batch_size=1, num_workers=0)
        # VERIFY: EfficientAd's external Imagenette dependency/download requirement.
        engine = Engine(accelerator="cpu", devices=1, max_epochs=args.epochs,
                        default_root_dir=str(out))
        engine.fit(model=model, datamodule=dataset)
        # VERIFY: export return path, ExportType import, export_root/input_size args.
        exported = engine.export(model=model, export_type=ExportType.OPENVINO,
                                 export_root=out, input_size=(args.image_size,) * 2)
    model_path = export_xml_path(exported).resolve()
    settings = dict(config["models"]["detector"])
    settings.update(implementation="openvino", path=str(model_path), device="CPU",
                    image_size=args.image_size, normalization_in_model=True)
    detector = OpenVINODetector(settings, ROOT)
    scores = {"good": [], "defect": []}
    records = []
    for label, paths in groups.items():
        for path in paths:
            with Image.open(path) as image:
                prepared = prepare_image(image, args.image_size, roi)
            result = detector.detect(Frame(str(path), prepared, ground_truth=label))
            scores[label].append(result.score)
            records.append({"image": str(path), "ground_truth": label, "score": result.score})
            print(f"{label}: {path} score={result.score:.6f}")
    calibration = suggest_threshold(scores["good"], scores["defect"], args.margin)
    stats = {label: score_stats(values) for label, values in scores.items()}
    settings["threshold"] = calibration["threshold"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path), "threshold": calibration["threshold"],
        "calibration": calibration, "score_stats": stats, "scores": records,
        "preprocessing": {"roi": roi, "source_images_already_cropped": not args.images_uncropped,
                          "image_size": args.image_size, "resize": "PIL.BILINEAR",
                          "input": "float32 NCHW RGB [0,1]", "normalization_in_model": True},
        "detector_config": settings,
        "evaluation_note": "Calibration uses all supplied images, including training images. "
                           "These are NOT held-out accuracy results; validate with fresh images.",
    }
    metadata_path = model_path.with_name("calibration.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stats": stats, "calibration": calibration}, indent=2))
    print(metadata["evaluation_note"])
    print(f"Saved {metadata_path}")
    print("Paste under models.detector in config/config.yaml (not edited automatically):")
    print(yaml.safe_dump(settings, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=["patchcore", "padim", "efficientad"], default="patchcore")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Override for trainable models; EfficientAd needs more than one epoch")
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--images-uncropped", action="store_true",
                        help="Apply config ROI; omit for photos saved by tools/capture.py")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    args = parser.parse_args()
    if args.image_size < 1 or args.epochs < 1:
        parser.error("--image-size and --epochs must be positive")
    # Validate margin without importing/starting any models.
    try:
        suggest_threshold([0.0], margin=args.margin)
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        train(args, config)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Training error: {exc}\n")


if __name__ == "__main__":
    main()
