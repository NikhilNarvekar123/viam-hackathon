"""Shared RGB/PIL resize contract for training staging and inference."""
import numpy as np
from PIL import Image

from robot.viam_hw import crop_image


def prepare_image(image: Image.Image, image_size: int, roi=None) -> Image.Image:
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    return crop_image(image.convert("RGB"), roi).resize(
        (image_size, image_size), Image.Resampling.BILINEAR)


def image_tensor(image: Image.Image, settings: dict) -> np.ndarray:
    # Frame.image is already cropped by the camera. Do NOT reapply config ROI.
    image = prepare_image(image, settings["image_size"])
    values = np.asarray(image, dtype=np.float32) / 255.0
    if not settings["normalization_in_model"]:
        mean = np.asarray(settings["mean"], dtype=np.float32)
        std = np.asarray(settings["std"], dtype=np.float32)
        if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("Normalization requires three finite means and positive stds")
        values = (values - mean) / std
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])
