from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.inspect_wrist import inspect, validate_rois


@pytest.mark.parametrize('box', [[-1,0,4,4], [0,0,9,4], [4,0,4,4], [0,0,4], [0,0,4.0,4]])
def test_invalid_roi(box):
    with pytest.raises(ValueError):
        validate_rois(Image.new('RGB', (8,8)), [box])


def test_missing_roi():
    with pytest.raises(ValueError):
        validate_rois(Image.new('RGB',(8,8)), [])


@pytest.mark.parametrize('score,threshold,label', [(0.4,0.4,'anomalous'), (0.39,0.4,'normal'), (0.6,None,'unclassified')])
def test_inspection(tmp_path, score, threshold, label):
    def detect(frame):
        assert frame.image.size == (4,4)
        return SimpleNamespace(score=score, heatmap=np.zeros((4,4), dtype=np.float32))
    records=inspect(Image.new('RGB',(8,8)), [[0,0,4,4]], SimpleNamespace(detect=detect), threshold, tmp_path)
    assert records[0]['classification'] == label
    assert (tmp_path/'object_001_overlay.png').exists()
    assert np.load(tmp_path/'object_001_map.npy').shape == (4,4)
