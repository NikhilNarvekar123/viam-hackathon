import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
import yaml

from robot.fakes import FakeArm
from robot import teach
from robot.viam_hw import ViamArm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    cfg["fake"]["arm_delay_s"] = 0
    return cfg


def test_merge_backup_and_replace(tmp_path):
    path = tmp_path / "poses.json"
    original = '{"_TODO": "keep", "home": {"joint_positions_deg": [1, 2]}}\n'
    path.write_text(original)
    backup = teach.save_pose(path, "pick", [3, 4], {"x": 1}, fake=True)
    assert backup.read_text() == original
    poses = teach.read_poses(path)
    assert poses["home"]["joint_positions_deg"] == [1, 2]
    assert poses["_TODO"] == "keep"
    assert poses["pick"]["source"] == "fake"
    before = path.read_bytes()
    second_backup = teach.save_pose(path, "pick", [5, 6], None, fake=True)
    assert backup != second_backup
    assert second_backup.read_bytes() == before
    assert teach.read_poses(path)["pick"]["joint_positions_deg"] == [5, 6]


def test_create_missing_file(tmp_path):
    path = tmp_path / "nested/poses.json"
    assert teach.save_pose(path, "home", [0, 0], None, fake=True) is None
    assert "home" in teach.read_poses(path)


@pytest.mark.parametrize("name,joints", [("", [1]), ("_TODO", [1]),
    ("home", []), ("home", [float("nan")]), ("home", [float("inf")])])
def test_bad_input_does_not_write(tmp_path, name, joints):
    path = tmp_path / "poses.json"
    with pytest.raises(ValueError):
        teach.save_pose(path, name, joints, None, fake=True)
    assert not path.exists()


def test_corrupt_file_not_overwritten(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text("broken")
    with pytest.raises(ValueError):
        teach.save_pose(path, "home", [1], None, fake=True)
    assert path.read_text() == "broken"


def test_atomic_replace_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "poses.json"
    path.write_text('{}\n')
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(teach.os, "replace", fail)
    with pytest.raises(OSError):
        teach.save_pose(path, "home", [1], None, fake=True)
    assert path.read_text() == '{}\n'
    assert len(list(tmp_path.iterdir())) == 2  # original and backup, no temp file


def test_fake_arm_readback():
    arm = FakeArm({"home": {"joint_positions_deg": [1, 2],
                            "end_position": {"x": 3, "y": 4, "z": 5}}})
    asyncio.run(arm.move_to("home"))
    joints = asyncio.run(arm.get_joint_positions())
    assert joints == [1, 2]
    joints[0] = 999
    assert asyncio.run(arm.get_joint_positions()) == [1, 2]
    assert asyncio.run(arm.get_end_position())["z"] == 5
    asyncio.run(arm.move_to_xyz(10, 20, 30))
    assert asyncio.run(arm.get_end_position()) == {"x": 10, "y": 20, "z": 30}
    with pytest.raises(ValueError, match="no IK"):
        asyncio.run(arm.get_joint_positions())


def test_save_only_reads_arm(config, tmp_path, monkeypatch):
    arm = AsyncMock()
    arm.get_joint_positions.return_value = [1, 2]
    arm.get_end_position.return_value = {"x": 3}
    @asynccontextmanager
    async def context(*args):
        yield arm
    monkeypatch.setattr(teach, "teaching_arm", context)
    asyncio.run(teach.teach(config, tmp_path / "poses.json", name="home", fake=True))
    assert [call[0] for call in arm.mock_calls] == ["get_joint_positions", "get_end_position"]


def test_goto_gate_before_connect(config, tmp_path, monkeypatch):
    connect = AsyncMock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(teach, "connect", connect)
    with pytest.raises(ValueError, match="confirm-motion"):
        asyncio.run(teach.teach(config, tmp_path / "poses.json", goto="home"))
    connect.assert_not_called()


def test_real_adapter_rejects_fake_taught_pose():
    device = AsyncMock()
    arm = ViamArm(device, {"home": {"source": "fake", "joint_positions_deg": [0]}}, {}, True)
    with pytest.raises(ValueError, match="fake-taught"):
        asyncio.run(arm.move_to("home"))
    assert not device.mock_calls


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_teaching_cleanup(config, monkeypatch, confirmed, failure):
    device, machine = AsyncMock(), AsyncMock()
    monkeypatch.setattr(teach, "load_credentials", lambda _: {})
    monkeypatch.setattr(teach, "connect", AsyncMock(return_value=machine))
    monkeypatch.setitem(sys.modules, "viam.components.arm", SimpleNamespace(
        Arm=SimpleNamespace(from_robot=lambda *args: device)))
    async def exercise():
        async with teach.teaching_arm(config, {}, False, confirmed):
            raise failure()
    with pytest.raises(failure):
        asyncio.run(exercise())
    assert device.stop.await_count == int(confirmed)
    machine.close.assert_awaited_once()


def test_cli_fake_save_list_goto(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text('{"existing": {"joint_positions_deg": [1, 2]}}')
    base = [sys.executable, str(ROOT / "robot/teach.py"), "--fake", "--poses", str(path)]
    def run(*args):
        return subprocess.run([*base, *args], cwd=tmp_path, capture_output=True, text=True)
    saved = run("home")
    assert saved.returncode == 0, saved.stderr
    assert "FAKE: saved home" in saved.stdout
    assert "Backup:" in saved.stdout
    listed = run("--list")
    assert listed.returncode == 0
    assert "home:" in listed.stdout and "existing:" in listed.stdout
    denied = run("--goto", "home")
    assert denied.returncode != 0 and "requires --confirm-motion" in denied.stderr
    moved = run("--goto", "home", "--confirm-motion")
    assert moved.returncode == 0, moved.stderr
    assert "FAKE: moved to home" in moved.stdout
    missing = run("--goto", "absent", "--confirm-motion")
    assert missing.returncode != 0 and "Unknown pose" in missing.stderr
