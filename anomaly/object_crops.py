"""Heuristic candidate crops on a plain tabletop; not semantic object detection."""
from PIL import ImageFilter
from robot.locate import locate_objects


def automatic_crops(image, settings):
    radius = settings["background_blur_radius_px"]
    if not isinstance(radius, (int, float)) or not 1 <= radius <= 200:
        raise ValueError("background_blur_radius_px must be between 1 and 200")
    maximum = settings["max_area_px"]
    if not isinstance(maximum, int) or maximum < settings["min_area_px"]:
        raise ValueError("max_area_px must be >= min_area_px")
    # Smooth local illumination estimate. Edges, texture and shadows can also respond.
    reference = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))
    result, mask = locate_objects(image, reference, settings)
    accepted, rejected = [], []
    for obj in result["objects"]:
        if obj["area_px"] > maximum or obj["warnings"]:
            rejected.append(obj)
            continue
        accepted.append(obj)
    for index, obj in enumerate(accepted, 1):
        obj["id"] = index
    result["objects"] = accepted
    result["rejected_candidates"] = rejected
    result["method"] = "local contrast against Gaussian-smoothed frame; NOT a trained detector"
    return [obj["crop_bbox_xyxy"] for obj in accepted], result, mask
