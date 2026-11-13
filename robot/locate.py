"""Pixel-only background subtraction. Not a grasp planner or robot calibration."""
from collections import deque
import math

import numpy as np
from PIL import Image, ImageDraw


def morphology(mask, radius, dilate):
    if radius == 0:
        return mask.copy()
    if not isinstance(radius, int) or not 0 <= radius <= 5:
        raise ValueError("Morphology radius must be an integer from 0 to 5")
    size = 2 * radius + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(mask, radius, constant_values=False), (size, size))
    return windows.any(axis=(-2, -1)) if dilate else windows.all(axis=(-2, -1))


def components(mask):
    """8-connected components, returned as arrays of (row, column) pixels."""
    remaining = mask.copy()
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        queue = deque([(int(y), int(x))])
        points = []
        while queue:
            row, col = queue.popleft()
            points.append((row, col))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = row + dy, col + dx
                    if 0 <= ny < height and 0 <= nx < width and remaining[ny, nx]:
                        remaining[ny, nx] = False
                        queue.append((ny, nx))
        yield np.asarray(points, dtype=int)


def locate_objects(image: Image.Image, background: Image.Image, settings: dict):
    if image.size != background.size:
        raise ValueError("Frame/background dimensions differ; recapture the empty-table reference")
    width, height = image.size
    roi = settings["workspace_roi"] or [0, 0, width, height]
    if len(roi) != 4 or any(not isinstance(v, int) for v in roi):
        raise ValueError("workspace_roi must be four integer pixel coordinates")
    left, top, right, bottom = roi
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("workspace_roi must fit inside the full camera frame")
    threshold = settings["difference_threshold"]
    minimum = settings["min_area_px"]
    ratio = settings["orientation_min_ratio"]
    padding = settings["crop_padding_px"]
    fraction_limit = settings["max_foreground_fraction"]
    if not math.isfinite(threshold) or not 0 < threshold < 255:
        raise ValueError("difference_threshold must be in (0, 255)")
    if not isinstance(minimum, int) or minimum < 2:
        raise ValueError("min_area_px must be an integer >= 2")
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError("orientation_min_ratio must be greater than 1")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("crop_padding_px must be a nonnegative integer")
    if not math.isfinite(fraction_limit) or not 0 < fraction_limit <= 1:
        raise ValueError("max_foreground_fraction must be in (0, 1]")
    current = np.asarray(image.convert("RGB"), dtype=np.int16)[top:bottom, left:right]
    reference = np.asarray(background.convert("RGB"), dtype=np.int16)[top:bottom, left:right]
    mask = np.max(np.abs(current - reference), axis=2) >= threshold
    fraction = float(mask.mean())
    if fraction > fraction_limit:
        raise ValueError(f"{fraction:.1%} of workspace changed: check lighting, camera movement, "
                         "hands, or reference image before trusting detections")
    opening, closing = settings["opening_radius_px"], settings["closing_radius_px"]
    mask = morphology(morphology(mask, opening, False), opening, True)
    mask = morphology(morphology(mask, closing, True), closing, False)
    # Raw changes at an edge must not disappear from the edge-warning decision.
    objects, accepted_mask = [], np.zeros((height, width), dtype=bool)
    for points in components(mask):
        if len(points) < minimum:
            continue
        xy = points[:, ::-1].astype(float) + [left, top]
        center = xy.mean(axis=0)
        covariance = np.cov(xy, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        anisotropy = float(eigenvalues[-1] / max(eigenvalues[0], 1e-9))
        angle = float(math.degrees(math.atan2(direction[1], direction[0])) % 180)
        if anisotropy < ratio:
            angle = None  # Round/symmetric objects do not have a reliable long axis.
        x0, y0 = xy.min(axis=0).astype(int)
        x1, y1 = xy.max(axis=0).astype(int) + 1
        near_edge = (x0 <= left + closing or y0 <= top + closing
                     or x1 >= right - closing or y1 >= bottom - closing)
        bounds = [int(x0), int(y0), int(x1), int(y1)]
        crop_bounds = [max(left, int(x0) - padding), max(top, int(y0) - padding),
                       min(right, int(x1) + padding), min(bottom, int(y1) + padding)]
        objects.append({
            "center_uv": center.tolist(), "bbox_xyxy": bounds,
            "crop_bbox_xyxy": crop_bounds, "area_px": len(points),
            "image_angle_deg": angle, "axis_ratio": anisotropy,
            "warnings": ["touches_workspace_boundary"] if near_edge else [],
        })
        accepted_mask[points[:, 0] + top, points[:, 1] + left] = True
    objects.sort(key=lambda obj: (obj["center_uv"][1], obj["center_uv"][0]))
    for index, obj in enumerate(objects, 1):
        obj["id"] = index  # Frame-local identifier, NOT cross-frame tracking.
    return {"objects": objects, "workspace_roi": roi, "foreground_fraction": fraction,
            "coordinate_system": "full-image pixels: u right, v down; angles clockwise from +u, modulo 180",
            "robot_targets_valid": False,
            "limitations": "Touching objects can merge; shadows/glare can be false objects. "
                           "Centers/axes are not validated grasp points or robot coordinates."}, accepted_mask


def annotate(image, result, mask):
    output = np.asarray(image.convert("RGB")).copy()
    boundary = mask & ~morphology(mask, 1, False)
    output[boundary] = [0, 255, 0]
    annotated = Image.fromarray(output)
    draw = ImageDraw.Draw(annotated)
    for obj in result["objects"]:
        x0, y0, x1, y1 = obj["bbox_xyxy"]
        u, v = obj["center_uv"]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="yellow", width=2)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill="red")
        angle = obj["image_angle_deg"]
        if angle is not None:
            length = max(x1 - x0, y1 - y0) / 2
            dx, dy = length * math.cos(math.radians(angle)), length * math.sin(math.radians(angle))
            draw.line((u - dx, v - dy, u + dx, v + dy), fill="cyan", width=2)
        draw.text((x0, max(0, y0 - 12)), str(obj["id"]), fill="yellow", stroke_width=1, stroke_fill="black")
    return annotated
