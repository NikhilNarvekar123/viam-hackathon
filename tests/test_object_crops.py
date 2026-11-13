from pathlib import Path

from PIL import Image, ImageDraw
import pytest
import yaml

from anomaly.object_crops import automatic_crops, extract_candidate_crops, square_cutout
import numpy as np


def settings(size=(240,240)):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/wrist_inspection.yaml').read_text())['auto_crop']
    cfg['reference_size_px'] = list(size)
    return cfg


def test_empty_table():
    boxes, result, mask = automatic_crops(Image.new('RGB', (200, 200), 'gray'), settings((200,200)))
    assert boxes == []
    assert not mask.any()


def test_two_separate_objects_and_edge_clutter():
    image = Image.new('RGB', (240, 240), 'gray')
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 50, 115, 80), fill='black')
    draw.ellipse((150, 150, 175, 175), fill='white')
    draw.rectangle((0, 10, 10, 100), fill='black')
    boxes, result, mask = automatic_crops(image, settings())
    assert len(boxes) == 2
    assert len(result['rejected_candidates']) >= 1
    for box in boxes:
        assert all(isinstance(x, int) for x in box)
        assert box[2]-box[0] == box[3]-box[1] > 0


def test_bad_config():
    cfg = settings()
    cfg['background_blur_radius_px'] = 0
    with pytest.raises(ValueError):
        automatic_crops(Image.new('RGB', (100,100)), cfg)
