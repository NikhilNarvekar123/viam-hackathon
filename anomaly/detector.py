"""OpenVINO adapter. SDK import is lazy; hardware execution is unverified."""
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly.preprocess import image_tensor
from core.types import Detection, Frame

logger = logging.getLogger(__name__)


def select_device(requested, available):
    if requested in available:
        return requested
    logger.warning("Device %r unavailable (%s); falling back to CPU", requested, available)
    if "CPU" not in available:
        raise ValueError("Requested device unavailable and CPU is not available")
    return "CPU"


def parse_outputs(score, heatmap):
    score = np.asarray(score)
    if score.size != 1 or not np.isfinite(score).all():
        raise ValueError("Expected one finite anomaly score for one item")
    if heatmap is not None:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        # Strip batch/channel singleton axes only; retain 1-pixel spatial axes.
        while heatmap.ndim > 2 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
            raise ValueError("Expected a finite two-dimensional anomaly heatmap")
        heatmap = heatmap.copy()
    return float(score.reshape(-1)[0]), heatmap


class OpenVINODetector:
    def __init__(self, settings: dict, root: Path):
        self.settings = settings
        self.threshold = float(settings["threshold"])
        if not math.isfinite(self.threshold):
            raise ValueError("Threshold must be finite")
        # Build/validate input before importing a heavy dependency.
        warmup = image_tensor(Image.new("RGB", (settings["image_size"],) * 2), settings)
        import openvino as ov

        self.core = ov.Core()
        cache = root / settings["cache_dir"]
        cache.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache)})
        self.device = select_device(settings.get("device"), self.core.available_devices)
        model = self.core.read_model(str(root / settings["path"]))
        if len(model.inputs) != 1:
            raise ValueError("Anomaly model must have one image input")
        port = model.input(0)
        if tuple(port.shape) != warmup.shape:
            raise ValueError(f"Configured tensor {warmup.shape} does not match exported input {tuple(port.shape)}")
        if port.element_type.get_type_name() != "f32":
            raise ValueError("This adapter expects an exported float32 NCHW image input")
        # Do not reshape an export to hide a preprocessing/config mismatch.
        # V2 graph verified: input NCHW RGB [0,1], embedded resize to 256x256,
        # then ImageNet normalization. Color semantics still require training provenance.
        self.compiled = self.core.compile_model(model, self.device, {"PERFORMANCE_HINT": "LATENCY"})
        self.input = self.compiled.input(0)
        # VERIFY: export output names; do not silently assume port order.
        self.score_output = self.compiled.output(settings["score_output"])
        map_name = settings.get("heatmap_output")
        self.heatmap_output = self.compiled.output(map_name) if map_name else None
        self._infer(warmup)  # Compile/warm up once, never per item.

    def _infer(self, tensor):
        result = self.compiled({self.input: tensor})
        return parse_outputs(result[self.score_output],
                             result[self.heatmap_output] if self.heatmap_output is not None else None)

    def detect(self, frame: Frame) -> Detection:
        score, heatmap = self._infer(image_tensor(frame.image, self.settings))
        # Return the heatmap aligned to the original crop for future overlays.
        if heatmap is not None and heatmap.shape != (frame.image.height, frame.image.width):
            width, height = frame.image.size
            side = max(width, height)
            square = Image.fromarray(heatmap).resize((side, side), Image.Resampling.BILINEAR)
            left, top = (side-width)//2, (side-height)//2
            heatmap = np.asarray(square.crop((left, top, left+width, top+height)), dtype=np.float32)
        return Detection(score, score >= self.threshold, heatmap)
