"""Build a side-by-side original crop and red anomaly heatmap overlay."""
import math

import numpy as np
from PIL import Image


def build_overlay(image: Image.Image, heatmap: np.ndarray | None, alpha=0.5,
                  value_min=0.0, value_max=1.0) -> Image.Image:
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be in [0, 1]")
    if not all(math.isfinite(v) for v in (value_min, value_max)) or value_max <= value_min:
        raise ValueError("Heatmap range must be finite and increasing")
    crop = image.convert("RGB")
    right = crop.copy()
    if heatmap is not None:
        values = np.asarray(heatmap, dtype=np.float32)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("Heatmap must be a nonempty finite 2D array")
        values = np.clip((values - value_min) / (value_max - value_min), 0, 1)
        values = np.asarray(Image.fromarray(values).resize(crop.size, Image.Resampling.BILINEAR))
        weight = values[..., None] * alpha
        rgb = np.asarray(crop, dtype=np.float32)
        right = Image.fromarray(np.rint(rgb * (1 - weight) +
                               np.array([255, 0, 0]) * weight).astype(np.uint8))
    # Missing heatmap means duplicate views, never fabricated anomaly evidence.
    result = Image.new("RGB", (crop.width * 2, crop.height))
    result.paste(crop, (0, 0))
    result.paste(right, (crop.width, 0))
    return result
