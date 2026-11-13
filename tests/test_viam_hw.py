import asyncio
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from PIL import Image
import pytest
import yaml

import core.factory as factory
import main
from core.types import Components, Detection, Frame
from robot import viam_hw

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    cfg["poses_file"] = str(ROOT / cfg["poses_file"])
    cfg["logs_dir"] = "logs"
    cfg["fake"] = {key: 0 for key in cfg["fake"]}
    return cfg


def test_credentials_missing(monkeypatch, tmp_path):
    for name in ("VIAM_ADDRESS", "VIAM_API_KEY", "VIAM_API_KEY_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="missing .env values: VIAM_ADDRESS"):
        viam_hw.load_credentials(tmp_path / ".env")


def test_dotenv_and_environment_precedence(monkeypatch, tmp_path):
    for name in ("VIAM_ADDRESS", "VIAM_API_KEY", "VIAM_API_KEY_ID"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("VIAM_ADDRESS=local\nVIAM_API_KEY=test\nVIAM_API_KEY_ID=id\n")
    monkeypatch.setenv("VIAM_ADDRESS", "override")
    assert viam_hw.load_credentials(env) == {
        "VIAM_ADDRESS": "override", "VIAM_API_KEY": "test", "VIAM_API_KEY_ID": "id"}


def test_depth_and_crop():
    payload = b"DEPTHMAP" + (2).to_bytes(8, "big") + (1).to_bytes(8, "big")
    depth = viam_hw.decode_depth(payload + b"\x00\x01\x01\x00")
    np.testing.assert_array_equal(depth, [[1, 256]])
    assert viam_hw.crop_image(Image.new("RGB", (10, 10)), [1, 2, 5, 6]).size == (4, 4)
    with pytest.raises(ValueError, match="ROI"):
        viam_hw.crop_image(Image.new("RGB", (10, 10)), [0, 0, 11, 10])


@pytest.mark.parametrize("payload", [b"", b"bad header" * 3,
    b"DEPTHMAP" + (2).to_bytes(8, "big") + (2).to_bytes(8, "big")])
def test_bad_depth(payload):
    with pytest.raises(ValueError):
        viam_hw.decode_depth(payload)


def test_camera_color_and_optional_depth():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    color = SimpleNamespace(data=buffer.getvalue(), mime_type="image/png", name="color")
    depth_bytes = b"DEPTHMAP" + (1).to_bytes(8, "big") * 2 + b"\x00\x05"
    depth_image = SimpleNamespace(data=depth_bytes, mime_type="image/vnd.viam.dep", name="depth")
    # Depth first ensures the wrapper selects by MIME, not list position.
    camera = SimpleNamespace(get_images=AsyncMock(return_value=([depth_image, color], None)))
    depth = SimpleNamespace(get_images=AsyncMock(return_value=([depth_image], None)))
    wrapper = viam_hw.ViamCamera(camera, depth, [0, 0, 4, 4])
    frame = asyncio.run(wrapper.get_frame())
    assert frame.image.size == (4, 4)
    assert frame.ground_truth is None
    np.testing.assert_array_equal(frame.depth, [[5]])
    assert asyncio.run(viam_hw.ViamCamera(camera).get_frame()).depth is None


def test_camera_raw_rgba():
    payload = b"RGBA" + (2).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([255, 0, 0, 255]) * 2
    image = SimpleNamespace(data=payload, mime_type="image/vnd.viam.rgba", name="overview")
    camera = SimpleNamespace(get_images=AsyncMock(return_value=([image], None)))
    frame = asyncio.run(viam_hw.ViamCamera(camera).get_frame())
    assert frame.image.size == (2, 1) and frame.image.getpixel((0, 0)) == (255, 0, 0)
    camera.get_images.assert_awaited_once_with()
    image.data = payload[:-1]
    with pytest.raises(ValueError, match="payload length"):
        viam_hw.decode_color(image)


@pytest.mark.parametrize("images", [[], [SimpleNamespace(name="depth", mime_type="image/vnd.viam.dep")],
    [SimpleNamespace(name="a", mime_type="image/png"), SimpleNamespace(name="b", mime_type="image/jpeg")]])
def test_camera_missing_or_ambiguous_color(images):
    camera = SimpleNamespace(get_images=AsyncMock(return_value=(images, None)))
    with pytest.raises(ValueError, match="Expected one color"):
        asyncio.run(viam_hw.ViamCamera(camera).get_frame())


def test_adapters_deny_motion_before_sdk_import():
    arm_device, grip_device = AsyncMock(), AsyncMock()
    arm = viam_hw.ViamArm(arm_device, {}, {})
    gripper = viam_hw.ViamGripper(grip_device)
    for call in (arm.move_to("home"), arm.move_to_xyz(1, 2, 3),
                 gripper.open(), gripper.grab()):
        with pytest.raises(PermissionError, match="confirm-motion"):
            asyncio.run(call)
    assert not arm_device.mock_calls and not grip_device.mock_calls


def test_adapters_with_mock_sdk(monkeypatch):
    # These stand-ins validate our plumbing, not actual SDK compatibility.
    monkeypatch.setitem(sys.modules, "viam.proto.component.arm",
                        SimpleNamespace(JointPositions=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "viam.proto.common",
                        SimpleNamespace(Pose=lambda **kw: kw))
    device = AsyncMock()
    device.get_joint_positions.return_value = SimpleNamespace(values=[1, 2])
    device.get_end_position.return_value = SimpleNamespace(
        x=1, y=2, z=3, o_x=0, o_y=0, o_z=-1, theta=0)
    orientation = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
    arm = viam_hw.ViamArm(device, {"home": {"joint_positions_deg": [1, 2]}},
                         {"down_orientation": orientation}, True)
    asyncio.run(arm.move_to("home"))
    device.move_to_joint_positions.assert_awaited_once_with({"values": [1, 2]})
    asyncio.run(arm.move_to_xyz(1, 2, 3))
    device.move_to_position.assert_awaited_once_with(dict(x=1, y=2, z=3, **orientation))
    assert asyncio.run(arm.get_joint_positions()) == [1, 2]
    assert asyncio.run(arm.get_end_position())["z"] == 3
    asyncio.run(arm.stop())
    device.stop.assert_awaited_once()
    gripper = viam_hw.ViamGripper(device, True)
    asyncio.run(gripper.open())
    asyncio.run(gripper.grab())
    device.open.assert_awaited_once()
    device.grab.assert_awaited_once()


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_connection_cleanup(config, tmp_path, monkeypatch, confirm, failure):
    machine, arm = AsyncMock(), AsyncMock()
    monkeypatch.setattr(viam_hw, "load_credentials", lambda _: {})
    monkeypatch.setattr(viam_hw, "build_hardware", AsyncMock(
        return_value=(object(), arm, object(), machine)))

    async def exercise():
        async with factory.managed_components(config, tmp_path, False, confirm):
            if failure:
                raise failure()

    if failure:
        with pytest.raises(failure):
            asyncio.run(exercise())
    else:
        asyncio.run(exercise())
    assert arm.stop.await_count == int(confirm and failure is not None)
    machine.close.assert_awaited_once()


def test_dry_run_has_no_arm_or_gripper_calls(config, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    arm, gripper = AsyncMock(), AsyncMock()
    components = Components(
        SimpleNamespace(get_frame=AsyncMock(return_value=Frame("test", Image.new("RGB", (4, 4))))),
        SimpleNamespace(detect=lambda _: Detection(0.1, False)),
        object(), arm, gripper, real_hardware=True)
    asyncio.run(main.run_loop(config, 1, components))
    assert not arm.mock_calls and not gripper.mock_calls
    import json
    record = json.loads(next((tmp_path / "logs").glob("*.jsonl")).read_text())
    assert not record["motion_executed"]
    assert record["final_bin"] == config["sorting"]["pass_bin"]


def test_cli_missing_credentials_no_traceback(tmp_path):
    # Run copied entry point with real project imports but isolated .env location.
    script = tmp_path / "main.py"
    script.write_text((ROOT / "main.py").read_text())
    env = {key: value for key, value in os.environ.items() if not key.startswith("VIAM_")}
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run([sys.executable, str(script), "--config",
                             str(ROOT / "config/config.yaml")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing .env values" in result.stderr
    assert "Traceback" not in result.stderr


def test_fake_mode_never_connects(config, tmp_path, monkeypatch):
    connect = AsyncMock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(viam_hw, "connect", connect)

    async def exercise():
        async with factory.managed_components(config, tmp_path, True, True) as components:
            assert not components.real_hardware

    asyncio.run(exercise())
    connect.assert_not_called()
