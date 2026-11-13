"""Pick target selection. Vision localization is a fake-only placeholder."""
from dataclasses import dataclass
import math

from core.types import Frame


@dataclass(frozen=True)
class PickTarget:
    pose_name: str | None = None
    xyz: tuple[float, float, float] | None = None
    is_stub: bool = False


def get_pick_target(frame: Frame, config: dict) -> PickTarget:
    mode = config["pick_mode"]
    if mode == "fixed":
        return PickTarget(pose_name=config["pick_pose"])
    if mode == "vision":
        # TODO(event): replace with pixel-to-arm calibration from task 8.
        # No inference from the image yet; these coordinates are simulation only.
        values = config["vision_stub_xyz_mm"]
        if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("vision_stub_xyz_mm must contain three finite coordinates")
        return PickTarget(xyz=tuple(float(v) for v in values), is_stub=True)
    raise ValueError(f"Unknown pick_mode: {mode}")
