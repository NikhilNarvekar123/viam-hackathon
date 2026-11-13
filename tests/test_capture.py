import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from PIL import Image
import pytest
import yaml

from core.types import Frame
from robot import camera as camera_module
from robot.fakes import FakeCamera
from tools.capture import capture, save_images

ROOT = Path(__file__).resolve().parents[1]


def test_fake_capture_crop_and_no_overwrite(tmp_path):
    camera = FakeCamera(tmp_path / "missing-samples")
    out = tmp_path / "images"
    first = asyncio.run(capture(camera, out, [10, 20, 50, 70], n=2, interval=0))
    original = first[0].read_bytes()
    second = asyncio.run(capture(camera, out, [10, 20, 50, 70], n=1, interval=0))
    assert [p.name for p in first + second] == [
        "image_000001.png", "image_000002.png", "image_000003.png"]
    assert first[0].read_bytes() == original
    for path in first + second:
        with Image.open(path) as image:
            assert image.size == (40, 50)


def test_preview_same_frame_and_exactly_one_crop(tmp_path):
    image = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    camera = SimpleNamespace(get_frame=AsyncMock(return_value=Frame("test", image)))
    paths = asyncio.run(capture(camera, tmp_path, [2, 3, 6, 7], n=20, preview=True))
    camera.get_frame.assert_awaited_once()
    with Image.open(paths[0]) as full, Image.open(paths[1]) as cropped:
        assert full.size == (8, 8)
        assert cropped.size == (4, 4)
        np.testing.assert_array_equal(cropped, np.asarray(image)[3:7, 2:6])


def test_preview_collision_keeps_existing_pair(tmp_path):
    occupied = tmp_path / "preview_000001_cropped.png"
    occupied.write_bytes(b"keep")
    image = Image.new("RGB", (4, 4))
    paths = save_images(tmp_path, {"_uncropped": image, "_cropped": image}, "preview")
    assert all("000002" in p.name for p in paths)
    assert occupied.read_bytes() == b"keep"
    assert not (tmp_path / "preview_000001_uncropped.png").exists()


def test_image_write_failure_removes_partial_pair(tmp_path):
    class BrokenImage:
        def save(self, *args, **kwargs):
            raise OSError("disk error")
    with pytest.raises(OSError):
        save_images(tmp_path, {"_full": Image.new("RGB", (4, 4)), "_crop": BrokenImage()})
    assert not list(tmp_path.iterdir())


def test_interactive_waits_before_each_frame(tmp_path):
    events = []
    class Camera:
        async def get_frame(self):
            events.append("capture")
            return Frame("test", Image.new("RGB", (4, 4)))
    async def wait(prompt):
        events.append("enter")
    asyncio.run(capture(Camera(), tmp_path, None, n=3, wait=wait))
    assert events == ["enter", "capture"] * 3


def test_interval_between_frames(tmp_path):
    camera = FakeCamera(tmp_path / "missing")
    wait, sleep = AsyncMock(), AsyncMock()
    asyncio.run(capture(camera, tmp_path / "out", None, n=3, interval=0.25,
                        wait=wait, sleep=sleep))
    wait.assert_not_called()
    assert sleep.await_count == 2
    sleep.assert_awaited_with(0.25)


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_bad_interval(tmp_path, interval):
    camera = SimpleNamespace(get_frame=AsyncMock())
    with pytest.raises(ValueError, match="interval"):
        asyncio.run(capture(camera, tmp_path, None, interval=interval))
    camera.get_frame.assert_not_called()


def test_invalid_roi_saves_nothing(tmp_path):
    camera = FakeCamera(tmp_path / "missing")
    with pytest.raises(ValueError, match="ROI"):
        asyncio.run(capture(camera, tmp_path / "out", [0, 0, 999, 999], n=1, interval=0))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_camera_only_cleanup(monkeypatch, failure):
    machine = AsyncMock()
    lookups = []
    def lookup(machine, name):
        lookups.append(name)
        return object()
    monkeypatch.setattr(camera_module, "load_credentials", lambda _: {})
    monkeypatch.setattr(camera_module, "connect", AsyncMock(return_value=machine))
    monkeypatch.setitem(sys.modules, "viam.components.camera", SimpleNamespace(
        Camera=SimpleNamespace(from_robot=lookup)))
    config = {"hardware": {"viam": {"camera": "test-camera"}}}
    async def exercise():
        async with camera_module.capture_camera(config, ROOT, False) as camera:
            assert camera.roi is None
            if failure:
                raise failure()
    if failure:
        with pytest.raises(failure):
            asyncio.run(exercise())
    else:
        asyncio.run(exercise())
    assert lookups == ["test-camera"]
    machine.close.assert_awaited_once()


def test_fake_camera_ignores_roi_until_capture(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["camera"]["roi"] = [0, 0, 8, 8]
    connect = AsyncMock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(camera_module, "connect", connect)
    async def exercise():
        async with camera_module.capture_camera(config, tmp_path, True) as camera:
            frame = await camera.get_frame()
            assert frame.image.size == (128, 128)
    asyncio.run(exercise())
    connect.assert_not_called()


def test_cli_batch_preview_interactive_and_eof(tmp_path):
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["camera"]["roi"] = [10, 10, 30, 40]
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    out = tmp_path / "images"
    base = [sys.executable, str(ROOT / "tools/capture.py"), "--fake", "--config", str(cfg),
            "--out", str(out)]
    def run(args, text=""):
        return subprocess.run(base + args, input=text, capture_output=True, text=True,
                              cwd=tmp_path, timeout=15)
    batch = run(["--n", "2", "--interval", "0"])
    assert batch.returncode == 0, batch.stderr
    preview = run(["--preview"])
    assert preview.returncode == 0, preview.stderr
    interactive = run(["--n", "2"], "\n\n")
    assert interactive.returncode == 0, interactive.stderr
    closed = run(["--n", "1"])
    assert closed.returncode == 1 and "stdin closed" in closed.stderr
    assert len(list(out.glob("image_*.png"))) == 4
    assert len(list(out.glob("preview_*.png"))) == 2
