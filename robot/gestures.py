"""Hello and dance: whimsical joint-space routines around the home pose.

Only base yaw (1), forearm roll (4), wrist pitch (5) and wrist roll (6) move, by small angles from the start
pose, so the tool stays about 300 mm above the table and far from the configured walls; shoulder and elbow
never move. Every routine ends exactly where it started, with the jaws open.
"""
import asyncio

from viam.proto.component.arm import JointPositions

LIMITS = {1: 20.0, 4: 30.0, 5: 30.0, 6: 75.0}     # max |offset| in degrees; joints not listed never move

# (what, {joint: offset from the start pose in degrees}, gripper: "open" | "close" | None)
HELLO = [
    ("hand up", {5: -25}, "open"),
    ("wave", {5: -25, 4: 22}, "close"),
    ("wave", {5: -25, 4: -22}, "open"),
    ("wave", {5: -25, 4: 22}, "close"),
    ("wave", {5: -25, 4: -22}, "open"),
    ("tilt head, curious", {5: -10, 6: 35}, None),
    ("tilt the other way", {5: -10, 6: -35}, None),
    ("bow", {5: 25}, "close"),
    ("hold the bow", {5: 28}, None),
    ("straighten up", {}, "open"),
]
DANCE = [
    ("sway left, twist", {1: 15, 6: -45}, None),
    ("sway right, twist", {1: -15, 6: 45}, None),
    ("sway left, twist", {1: 15, 6: -45}, None),
    ("sway right, twist", {1: -15, 6: 45}, None),
    ("bob down, clap", {5: 18}, "close"),
    ("bob up", {5: -18}, "open"),
    ("bob down, clap", {5: 18}, "close"),
    ("bob up", {5: -18}, "open"),
    ("arms out left", {4: 28, 6: 60}, None),
    ("arms out right", {4: -28, 6: -60}, None),
    ("twirl", {6: 75}, None),
    ("twirl back", {6: -75}, None),
    ("shimmy", {1: 6, 4: 8}, None),
    ("shimmy", {1: -6, 4: -8}, None),
    ("shimmy", {1: 6, 4: 8}, None),
    ("shimmy", {1: -6, 4: -8}, None),
    ("big finish, reach up", {5: -28, 6: 30}, "close"),
    ("take a bow", {5: 28}, "open"),
    ("back to home", {}, None),
]
ROUTINES = {"hello": HELLO, "dance": DANCE}


def check(steps):
    """Refuse a routine that moves any joint other than LIMITS, or past its limit."""
    for what, offsets, grip in steps:
        for joint, offset in offsets.items():
            if joint not in LIMITS or abs(offset) > LIMITS[joint]:
                raise ValueError(f"{what}: joint {joint} offset {offset} outside the gesture limits")
        if grip not in (None, "open", "close"):
            raise ValueError(f"{what}: unknown gripper action {grip}")


for _steps in ROUTINES.values():
    check(_steps)


async def perform(arm, gripper, name, log=print):
    """Run a routine from the arm's current pose and always return to it, jaws open."""
    steps = ROUTINES[name]
    check(steps)
    start = list((await arm.get_joint_positions()).values)
    try:
        for what, offsets, grip in steps:
            target = list(start)
            for joint, offset in offsets.items():
                target[joint - 1] += offset
            log(f"  {name}: {what}")
            moves = [arm.move_to_joint_positions(JointPositions(values=target))]
            if grip == "open":
                moves.append(gripper.open())
            elif grip == "close":
                moves.append(gripper.grab())
            await asyncio.gather(*moves)
    finally:
        await arm.move_to_joint_positions(JointPositions(values=start))
        await gripper.open()
    return start
