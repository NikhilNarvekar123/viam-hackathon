import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest
import yaml

from robot.locate import locate_objects, morphology
from tools.locate import save_results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    values = yaml.safe_load((ROOT / "config/localization.yaml").read_text())
    values.update(min_area_px=20, closing_radius_px=0, crop_padding_px=3)
    return values


def scene():
    background = Image.new("RGB", (200, 160), "white")
    image = background.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 25, 70, 33), fill="gray")
    draw.rectangle((110, 40, 118, 95), fill="gray")
    draw.ellipse((45, 100, 65, 120), fill="gray")
    return background, image


def test_centers_axes_and_round_object(settings):
    background, image = scene()
    result, mask = locate_objects(image, background, settings)
    objects = result["objects"]
    assert len(objects) == 3
    assert objects[0]["center_uv"] == [45.0, 29.0]
    assert objects[0]["image_angle_deg"] == pytest.approx(0)
    assert objects[1]["image_angle_deg"] == pytest.approx(90)
    assert objects[2]["image_angle_deg"] is None
    assert mask.shape == (160, 200) and not result["robot_targets_valid"]


def test_workspace_offset_and_crop(settings):
    background, image = scene()
    settings["workspace_roi"] = [90, 20, 150, 120]
    settings["max_foreground_fraction"] = 1
    result, _ = locate_objects(image, background, settings)
    obj = result["objects"][0]
    assert obj["center_uv"] == [114.0, 67.5]
    assert obj["bbox_xyxy"] == [110, 40, 119, 96]
    assert obj["crop_bbox_xyxy"] == [107, 37, 122, 99]


def test_empty_scene_noise_and_large_change(settings):
    background, _ = scene()
    result, _ = locate_objects(background, background, settings)
    assert result["objects"] == []
    noisy = background.copy()
    noisy.putpixel((30, 30), (0, 0, 0))
    assert not locate_objects(noisy, background, settings)[0]["objects"]
    with pytest.raises(ValueError, match="workspace changed"):
        locate_objects(Image.new("RGB", background.size, "black"), background, settings)


def test_size_mismatch(settings):
    with pytest.raises(ValueError, match="dimensions differ"):
        locate_objects(Image.new("RGB", (2, 2)), Image.new("RGB", (3, 3)), settings)


def test_diagonal_orientation(settings):
    background = Image.new("RGB", (100, 100), "white")
    image = background.copy()
    ImageDraw.Draw(image).line((20, 20, 70, 70), fill="gray", width=7)
    obj = locate_objects(image, background, settings)[0]["objects"][0]
    assert obj["image_angle_deg"] == pytest.approx(45)


def test_touching_items_merge_and_boundary_warns(settings):
    background = Image.new("RGB", (100, 100), "white")
    image = background.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 10, 20, 20), fill="gray")
    draw.rectangle((20, 10, 40, 20), fill="gray")
    result, _ = locate_objects(image, background, settings)
    assert len(result["objects"]) == 1  # Explicit limitation, not separate grasp targets.
    assert "touches_workspace_boundary" in result["objects"][0]["warnings"]


def test_morphology():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True
    mask[4, 4] = False
    closed = morphology(morphology(mask, 1, True), 1, False)
    assert closed[4, 4]
    with pytest.raises(ValueError):
        morphology(mask, 9, True)


def test_saved_artifacts(settings, tmp_path):
    background, image = scene()
    folder = save_results(image, background, settings, tmp_path, "offline-test")
    result = json.loads((folder / "objects.json").read_text())
    assert len(result["objects"]) == 3
    assert (folder / "annotated.png").exists()
    for obj in result["objects"]:
        with Image.open(folder / obj["crop_path"]) as crop:
            bounds = obj["crop_bbox_xyxy"]
            assert crop.size == (bounds[2] - bounds[0], bounds[3] - bounds[1])
            np.testing.assert_array_equal(crop, image.crop(bounds))


def test_cli_offline_and_reference_not_overwritten(tmp_path):
    background, image = scene()
    bg, frame = tmp_path / "bg.png", tmp_path / "frame.png"
    background.save(bg)
    image.save(frame)
    result = subprocess.run([sys.executable, str(ROOT / "tools/locate.py"),
        "--image", str(frame), "--background", str(bg), "--out", str(tmp_path / "results")],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "Detected 3 candidate objects" in result.stdout
    protected = subprocess.run([sys.executable, str(ROOT / "tools/locate.py"),
        "--image", str(frame), "--save-background", str(bg)], capture_output=True, text=True)
    assert protected.returncode == 1 and "already exists" in protected.stderr
