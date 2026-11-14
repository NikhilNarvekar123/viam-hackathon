import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from robot.gestures import FakeArm, FakeGripper, FakeSwitch, perform, run_gesture

ROOT = Path(__file__).resolve().parents[1]


def setup():
    settings = yaml.safe_load((ROOT/'config/gestures.yaml').read_text())
    settings['settle_s'] = 0
    home = yaml.safe_load((ROOT/'config/config.yaml').read_text())['overview']['home']
    degrees = {'home-pose': [0.]*6, 'left_wave': [10., 2., 3., 4., 5., 6.],
               'right_wave': [-10., 2., 3., 4., 5., 6.], 'up_wave': [0., 2., 3., 20., 5., 6.],
               'down_wave': [0., 2., 3., -20., 5., 6.]}
    switches = {k: FakeSwitch(v) for k,v in degrees.items()}
    return settings, home, degrees, switches


@pytest.mark.parametrize('name,sequence', [
    ('hello', ['left_wave', 'right_wave', 'left_wave', 'right_wave']),
    ('dance', ['left_wave', 'up_wave', 'right_wave', 'down_wave']),
])
def test_exact_taught_sequence_then_home_without_recording_or_gripper_commands(name, sequence):
    settings, home, degrees, switches = setup()
    arm, gripper, preflight = FakeArm(), FakeGripper(), AsyncMock()
    result = asyncio.run(perform(arm, gripper, switches, name, settings, home,
                                preflight=preflight, fake=True))
    assert result == dict(status='completed', visited=['home-pose', *sequence, 'home-pose'])
    assert len(arm.moves) == len(sequence)+1
    for actual, resource in zip(arm.moves, sequence+['home-pose']):
        assert actual == pytest.approx(degrees[resource])
    assert arm.joints == [0.]*6
    # Fakes have no gripper.open/grab or switch.set_position at all: calls would fail.
    assert preflight.await_count == len(sequence)+2


@pytest.mark.parametrize('joints', [[], [0.]*5, [0.]*7, [float('nan')]*6, [True]*6, None])
def test_all_saved_poses_validated_before_motion(joints):
    settings, home, _, switches = setup();arm = FakeArm()
    switches['down_wave'] = SimpleNamespace(do_command=AsyncMock(return_value={'joints': joints}))
    with pytest.raises(ValueError, match='saved pose'):
        asyncio.run(perform(arm, FakeGripper(), switches, 'dance', settings, home,
                            preflight=AsyncMock(), fake=True))
    assert not arm.moves


def test_real_gesture_requires_explicit_authorization():
    settings, home, _, switches = setup();arm = FakeArm()
    with pytest.raises(ValueError, match='confirm-motion'):
        asyncio.run(perform(arm, FakeGripper(), switches, 'hello', settings, home, preflight=AsyncMock()))
    assert not arm.moves


def test_held_object_refuses_gesture_without_opening_or_homing():
    settings, home, _, switches = setup();arm = FakeArm()
    gripper = SimpleNamespace(is_holding_something=AsyncMock(return_value=SimpleNamespace(is_holding_something=True)))
    with pytest.raises(ValueError, match='empty gripper'):
        asyncio.run(perform(arm, gripper, switches, 'hello', settings, home, preflight=AsyncMock(), fake=True))
    assert not arm.moves


def test_busy_controller_preflight_refuses_motion():
    settings, home, _, switches = setup();arm = FakeArm()
    with pytest.raises(ValueError, match='teleop'):
        asyncio.run(perform(arm, FakeGripper(), switches, 'hello', settings, home,
                            preflight=AsyncMock(side_effect=ValueError('teleop active')), fake=True))
    assert not arm.moves


def test_stop_between_waypoints_does_not_auto_home():
    settings, home, degrees, switches = setup();arm = FakeArm()
    result = asyncio.run(perform(arm, FakeGripper(), switches, 'hello', settings, home,
        preflight=AsyncMock(), fake=True, checkpoint=lambda: bool(arm.moves)))
    assert result['status'] == 'interrupted' and result['visited'] == ['home-pose', 'left_wave']
    assert len(arm.moves) == 1 and arm.joints == pytest.approx(degrees['left_wave'])


def test_failed_move_stops_without_recovery_home_or_release():
    settings, home, _, switches = setup();arm = FakeArm();attempted = []
    async def fail(positions):
        attempted.append(list(positions.values));raise ValueError('move failed')
    arm.move_to_joint_positions = fail
    with pytest.raises(ValueError, match='move failed'):
        asyncio.run(perform(arm, FakeGripper(), switches, 'hello', settings, home,
                            preflight=AsyncMock(), fake=True))
    assert arm.stopped and len(attempted) == 1


def test_wrong_arrival_stops_without_next_pose():
    settings, home, _, switches = setup();arm = FakeArm()
    arm.move_to_joint_positions = AsyncMock()  # Deliberately leaves actual joints at home.
    with pytest.raises(ValueError, match='did not reach'):
        asyncio.run(perform(arm, FakeGripper(), switches, 'hello', settings, home,
                            preflight=AsyncMock(), fake=True))
    assert arm.stopped and arm.move_to_joint_positions.await_count == 1


def test_fake_runner_never_connects_and_uses_configured_paths():
    config = yaml.safe_load((ROOT/'config/config.yaml').read_text())
    with patch('robot.viam_hw.connect', side_effect=AssertionError('No real hardware')):
        result = asyncio.run(run_gesture('dance', config, ROOT, fake=True))
    assert result['visited'] == ['home-pose', 'left_wave', 'up_wave', 'right_wave', 'down_wave', 'home-pose']


def test_unconfirmed_runner_refuses_before_connection_or_pose_read():
    with patch('robot.viam_hw.connect', side_effect=AssertionError('No connection')):
        with pytest.raises(ValueError, match='confirm-motion'):
            asyncio.run(run_gesture('hello', {}, ROOT))
