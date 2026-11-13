"""One shared movement plan for fake execution, real execution, and dry runs."""
from dataclasses import dataclass
import math

from core.types import Arm, Frame, Gripper
from robot.pick import get_pick_target


@dataclass(frozen=True)
class Step:
    operation: str
    target: str | tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SortPlan:
    steps: tuple[Step, ...]
    uses_stub_target: bool = False


def build_sort_plan(frame: Frame, destination: str, config: dict) -> SortPlan:
    target = get_pick_target(frame, config)
    if target.xyz is None:
        above = Step("move_to", config["above_pick_pose"])
        pick = Step("move_to", target.pose_name)
    else:
        offset = float(config["above_pick_z_offset_mm"])
        if not math.isfinite(offset) or offset <= 0:
            raise ValueError("above_pick_z_offset_mm must be finite and positive")
        x, y, z = target.xyz
        above = Step("move_to_xyz", (x, y, z + offset))
        pick = Step("move_to_xyz", target.xyz)
    return SortPlan((
        Step("move_to", config["home_pose"]),
        Step("open"),  # Ensure the gripper is open before approaching the item.
        above, pick, Step("grab"), above,
        Step("move_to", config["above_bin_poses"][destination]),
        Step("open"), Step("move_to", config["home_pose"]),
    ), target.is_stub)


def describe_plan(plan: SortPlan) -> str:
    return " -> ".join(f"{step.operation}({step.target})" if step.target is not None
                       else f"{step.operation}()" for step in plan.steps)


async def sort_item(arm: Arm, gripper: Gripper, plan: SortPlan, *, allow_stub=False):
    if plan.uses_stub_target and not allow_stub:
        raise ValueError("Vision pick is a fake-only stub until calibration is implemented")
    for step in plan.steps:
        if step.operation == "move_to":
            await arm.move_to(step.target)
        elif step.operation == "move_to_xyz":
            await arm.move_to_xyz(*step.target)
        elif step.operation == "open":
            await gripper.open()
        elif step.operation == "grab":
            await gripper.grab()
        else:
            raise ValueError(f"Unknown sorting operation: {step.operation}")
