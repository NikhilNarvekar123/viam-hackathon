import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from robot.home import replay_home, home_before_capture, open_for_capture


def test_gripper_open_and_verify():
    gripper=SimpleNamespace(open=AsyncMock(),is_moving=AsyncMock(return_value=False),
        do_command=AsyncMock(return_value={'pos':850}),stop=AsyncMock())
    settings=dict(timeout_s=1,settle_s=0,gripper_open_position=850,gripper_open_tolerance=30)
    asyncio.run(open_for_capture(gripper,settings))
    gripper.open.assert_awaited_once()
    gripper.do_command.assert_awaited_once_with({'get':True})
    gripper.stop.assert_not_awaited()


@pytest.mark.parametrize('position',[0,400,float('nan')])
def test_gripper_not_open_stops(position):
    gripper=SimpleNamespace(open=AsyncMock(),is_moving=AsyncMock(return_value=False),
        do_command=AsyncMock(return_value={'pos':position}),stop=AsyncMock())
    settings=dict(timeout_s=1,settle_s=0,gripper_open_position=850,gripper_open_tolerance=30)
    with pytest.raises(ValueError): asyncio.run(open_for_capture(gripper,settings))
    gripper.stop.assert_awaited_once()


def devices():
    arm=SimpleNamespace(do_command=AsyncMock(),is_moving=AsyncMock(return_value=False),
        get_joint_positions=AsyncMock(side_effect=[SimpleNamespace(values=[0,0]),SimpleNamespace(values=[0,90])]),stop=AsyncMock())
    switch=SimpleNamespace(do_command=AsyncMock(return_value={'joints':[0,math.pi/2]}),set_position=AsyncMock())
    settings=dict(speed_deg_s=5,acceleration_deg_s2=10,joint_tolerance_deg=1,timeout_s=1,settle_s=0)
    return arm,switch,settings


def test_live_home_cannot_be_skipped(tmp_path):
    config={'overview':{'home':{'switch_name':'home-pose'}}}
    with pytest.raises(ValueError,match='cannot be skipped'):
        asyncio.run(home_before_capture(config,tmp_path,False,False))


def test_home_go_to_not_save():
    arm,switch,settings=devices()
    asyncio.run(replay_home(arm,switch,settings))
    switch.set_position.assert_awaited_once_with(2)
    arm.do_command.assert_awaited_once_with({'set_speed':5.0,'set_acceleration':10.0})
    arm.stop.assert_not_awaited()


def test_already_home_no_motion():
    arm,switch,settings=devices()
    arm.get_joint_positions.side_effect=None
    arm.get_joint_positions.return_value=SimpleNamespace(values=[0,90])
    asyncio.run(replay_home(arm,switch,settings))
    switch.set_position.assert_not_awaited()
    arm.do_command.assert_not_awaited()


def test_home_failure_stops():
    arm,switch,settings=devices()
    switch.set_position.side_effect=RuntimeError('failed')
    with pytest.raises(RuntimeError): asyncio.run(replay_home(arm,switch,settings))
    arm.stop.assert_awaited_once()


def test_home_wrong_final_pose_stops():
    arm,switch,settings=devices()
    arm.get_joint_positions.side_effect=None
    arm.get_joint_positions.return_value=SimpleNamespace(values=[0,0])
    with pytest.raises(ValueError,match='did not reach'): asyncio.run(replay_home(arm,switch,settings))
    arm.stop.assert_awaited_once()
