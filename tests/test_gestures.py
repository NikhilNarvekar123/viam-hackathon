import asyncio

import pytest

from robot.gestures import LIMITS, ROUTINES, check, perform
from tools.gesture import FakeArm, FakeGripper


@pytest.mark.parametrize("name", sorted(ROUTINES))
def test_routine_stays_in_limits_and_ends_where_it_started(name):
    arm, seen = FakeArm(), []
    start = list(arm.joints)
    move = arm.move_to_joint_positions

    async def record(positions):
        seen.append(list(positions.values))
        await move(positions)
    arm.move_to_joint_positions = record
    asyncio.run(perform(arm, FakeGripper(), name, log=lambda *_: None))
    assert arm.joints == start
    for joints in seen:
        for j, (a, b) in enumerate(zip(joints, start), 1):
            assert abs(a - b) <= LIMITS.get(j, 0.0) + 1e-9, (j, a, b)


def test_shoulder_or_elbow_offsets_are_refused():
    with pytest.raises(ValueError):
        check([("reach", {2: 5}, None)])
