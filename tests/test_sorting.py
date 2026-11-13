import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

from PIL import Image
import pytest
import yaml

from core.factory import build_components
from core.types import Frame
from robot.fakes import FakeArm, FakeGripper
from robot.pick import get_pick_target
from robot.sorting import build_sort_plan, describe_plan, sort_item

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())


@pytest.fixture
def frame():
    return Frame("test", Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("mode", ["fixed", "vision"])
@pytest.mark.parametrize("destination", ["box", "reject_bin"])
def test_full_sequence(config, frame, mode, destination, caplog):
    settings = config["sorting"]
    settings["pick_mode"] = mode
    poses = json.loads((ROOT / config["poses_file"]).read_text())
    arm, gripper = FakeArm(poses), FakeGripper()
    plan = build_sort_plan(frame, destination, settings)
    with caplog.at_level(logging.INFO, logger="robot.fakes"):
        asyncio.run(sort_item(arm, gripper, plan, allow_stub=True))
    if mode == "fixed":
        above, pick = "above_pick", "pick"
    else:
        x, y, z = settings["vision_stub_xyz_mm"]
        above, pick = (x, y, z + settings["above_pick_z_offset_mm"]), (x, y, z)
    assert arm.calls == ["home", above, pick, above,
                         settings["above_bin_poses"][destination], "home"]
    assert gripper.calls == ["open", "grab", "open"]
    # Verify cross-device call order, not just separate per-device histories.
    operations = [record.message.split("(")[0] for record in caplog.records]
    move = "FakeArm.move_to" if mode == "fixed" else "FakeArm.move_to_xyz"
    assert operations == ["FakeArm.move_to", "FakeGripper.open", move, move,
                          "FakeGripper.grab", move, "FakeArm.move_to",
                          "FakeGripper.open", "FakeArm.move_to"]
    assert len(caplog.records) == 9
    assert settings["above_bin_poses"][destination] in describe_plan(plan)


def test_vision_stub_cannot_move_real_hardware(config, frame):
    config["sorting"]["pick_mode"] = "vision"
    plan = build_sort_plan(frame, "box", config["sorting"])
    arm, gripper = AsyncMock(), AsyncMock()
    with pytest.raises(ValueError, match="fake-only"):
        asyncio.run(sort_item(arm, gripper, plan))
    assert not arm.mock_calls and not gripper.mock_calls


def test_fixed_target(config, frame):
    target = get_pick_target(frame, config["sorting"])
    assert target.pose_name == "pick"
    assert target.xyz is None and not target.is_stub


@pytest.mark.parametrize("offset", [0, -1, float("inf"), float("nan")])
def test_invalid_clearance(config, frame, offset):
    config["sorting"].update(pick_mode="vision", above_pick_z_offset_mm=offset)
    with pytest.raises(ValueError, match="positive"):
        build_sort_plan(frame, "box", config["sorting"])


@pytest.mark.parametrize("xyz", [[1, 2], [1, 2, float("nan")]])
def test_invalid_xyz(config, frame, xyz):
    config["sorting"].update(pick_mode="vision", vision_stub_xyz_mm=xyz)
    with pytest.raises(ValueError, match="three finite"):
        get_pick_target(frame, config["sorting"])


def test_unknown_mode(config, frame):
    config["sorting"]["pick_mode"] = "typo"
    with pytest.raises(ValueError, match="Unknown pick_mode"):
        get_pick_target(frame, config["sorting"])


def test_custom_bin_from_config(config, frame):
    config["sorting"]["above_bin_poses"]["review_bin"] = "above_review"
    plan = build_sort_plan(frame, "review_bin", config["sorting"])
    assert plan.steps[-3].target == "above_review"


def test_missing_approach_pose_rejected(config, tmp_path):
    poses = json.loads((ROOT / config["poses_file"]).read_text())
    del poses["above_pick"]
    path = tmp_path / "poses.json"
    path.write_text(json.dumps(poses))
    config["poses_file"] = str(path)
    with pytest.raises(ValueError, match="Missing taught"):
        build_components(config, ROOT, fake=True)
