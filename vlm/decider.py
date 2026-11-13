"""Lazy OpenVINO GenAI VLM adapter. SDK calls are unverified until NUC testing."""
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from anomaly.detector import select_device
from core.types import Decision, Detection, Frame
from vlm.overlay import build_overlay
from vlm.parse import parse_decision
from vlm.prompt import build_prompt


class OpenVINODecider:
    implementation = "openvino"

    def __init__(self, settings: dict, defect_types, allowed_actions, root: Path):
        self.settings = settings
        self.allowed_actions = allowed_actions
        self.prompt = build_prompt(settings["prompt"], defect_types, allowed_actions)
        self.last_latency_ms = 0.0
        limit = settings["max_new_tokens"]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        size = settings["warmup_image_size"]
        if not isinstance(size, int) or size < 1:
            raise ValueError("warmup_image_size must be a positive integer")
        model_path = root / settings["path"]
        if not model_path.is_dir():
            raise ValueError(f"VLM model directory not found: {model_path}")
        import openvino as ov
        import openvino_genai as genai

        self.ov = ov
        core = ov.Core()
        self.device = select_device(settings.get("device"), core.available_devices)
        cache = root / settings["cache_dir"]
        cache.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        # VERIFY: VLMPipeline kwargs for CACHE_DIR/PERFORMANCE_HINT on pinned GenAI.
        # VERIFY: supported model architecture/device (especially NPU static shapes).
        # Do not claim NPU/GPU support until compile + CPU-output comparisons pass.
        self.pipeline = genai.VLMPipeline(str(model_path), self.device,
                                         CACHE_DIR=str(cache), PERFORMANCE_HINT="LATENCY")
        # VERIFY: GenerationConfig supports these fields and greedy decoding.
        self.generation_config = genai.GenerationConfig()
        self.generation_config.max_new_tokens = limit
        self.generation_config.do_sample = False
        self.generation_config.num_beams = 1
        warmup = build_overlay(Image.new("RGB", (size, size)), None,
                               **settings["overlay"])
        self._generate(warmup)  # Warm once at startup; no per-item pipeline reload.

    def _generate(self, image: Image.Image) -> str:
        # VERIFY: VLM images use uint8 NHWC batch tensors, not normalized NCHW.
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8)[None])
        tensor = self.ov.Tensor(pixels)
        # No chat session: every item is independent; do not retain previous decisions.
        # VERIFY: images= and generation_config= kwargs and returned .texts shape.
        output = self.pipeline.generate(self.prompt, images=[tensor],
                                        generation_config=self.generation_config)
        if isinstance(output, str):
            return output
        texts = getattr(output, "texts", None)
        if texts and isinstance(texts[0], str):
            return texts[0]
        raise ValueError("Unexpected VLM result format; VERIFY GenAI result.texts API")

    def decide(self, frame: Frame, detection: Detection) -> Decision:
        start = perf_counter()
        try:
            image = build_overlay(frame.image, detection.heatmap, **self.settings["overlay"])
            raw = self._generate(image)
            return parse_decision(raw, self.allowed_actions)
        finally:
            self.last_latency_ms = (perf_counter() - start) * 1000
