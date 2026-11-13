import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import yaml

from anomaly.detector import OpenVINODetector, parse_outputs, select_device
from anomaly.preprocess import image_tensor, prepare_image
from anomaly.train import discover_images, export_xml_path, stage_dataset
from core.factory import build_detector
from core.types import Frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())


def test_shared_preprocessing(config, tmp_path):
    settings = config["models"]["detector"]
    settings["image_size"] = 4
    raw = Image.fromarray(np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3))
    source = tmp_path / "source.png"
    raw.save(source)
    roi = [2, 3, 8, 9]
    stage_dataset({"good": [source], "defect": []}, tmp_path / "staged", 4, roi)
    with Image.open(tmp_path / "staged/good/000000.png") as staged:
        # Training staging and an already-cropped camera frame agree exactly.
        np.testing.assert_array_equal(image_tensor(staged, settings),
                                      image_tensor(raw.crop(roi), settings))
    tensor = image_tensor(raw.crop(roi), settings)
    assert tensor.shape == (1, 3, 4, 4) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor.min() >= 0 and tensor.max() <= 1


def test_normalization(config):
    settings = config["models"]["detector"]
    settings.update(image_size=2, normalization_in_model=False,
                    mean=[0.5] * 3, std=[0.5] * 3)
    np.testing.assert_array_equal(image_tensor(Image.new("L", (2, 2), 255), settings),
                                  np.ones((1, 3, 2, 2), dtype=np.float32))
    settings["std"] = [0, 1, 1]
    with pytest.raises(ValueError, match="Normalization"):
        image_tensor(Image.new("RGB", (2, 2)), settings)


def test_bad_resize_and_roi():
    with pytest.raises(ValueError):
        prepare_image(Image.new("RGB", (2, 2)), 0)
    with pytest.raises(ValueError, match="ROI"):
        prepare_image(Image.new("RGB", (2, 2)), 4, [0, 0, 3, 3])


def test_device_fallback(caplog):
    assert select_device("NPU", ["CPU", "NPU"]) == "NPU"
    assert select_device("GPU", ["CPU"]) == "CPU"
    assert "falling back to CPU" in caplog.text
    assert select_device("GPU.0", ["CPU", "GPU.0"]) == "GPU.0"
    with pytest.raises(ValueError, match="CPU"):
        select_device("NPU", ["GPU"])


def test_output_validation():
    score, heatmap = parse_outputs(np.array([0.5]), np.zeros((1, 1, 2, 3)))
    assert score == 0.5 and heatmap.shape == (2, 3)
    assert parse_outputs([0.2], None)[1] is None
    assert parse_outputs([0.2], np.zeros((1, 1, 1, 3)))[1].shape == (1, 3)
    for score, heatmap in [([1, 2], None), ([np.nan], None), ([1], [np.nan]),
                           ([1], np.zeros((2, 3, 4)))]:
        with pytest.raises(ValueError):
            parse_outputs(score, heatmap)


@pytest.mark.parametrize("device", ["CPU", "NPU"])
def test_mocked_openvino_compile_once_and_warmup(config, tmp_path, monkeypatch, device):
    # This is a fake runtime, not an OpenVINO execution or hardware benchmark.
    settings = config["models"]["detector"]
    settings.update(device=device, image_size=4)
    model = Mock()
    model.inputs = ["input"]
    model.input.return_value = "input"
    compiled = Mock()
    compiled.input.return_value = "input"
    compiled.output.side_effect = lambda name: name
    compiled.return_value = {"pred_score": np.array([0.7]),
                             "anomaly_map": np.ones((1, 1, 4, 4))}
    core = Mock()
    core.available_devices = ["CPU", "NPU"]
    core.read_model.return_value = model
    core.compile_model.return_value = compiled
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    detector = OpenVINODetector(settings, tmp_path)
    assert compiled.call_count == 1  # startup warm-up
    frame = Frame("test", Image.new("RGB", (8, 6)))
    result = detector.detect(frame)
    detector.detect(frame)
    assert compiled.call_count == 3
    core.compile_model.assert_called_once_with(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    assert model.reshape.call_count == int(device == "NPU")
    if device == "NPU":
        model.reshape.assert_called_once_with({"input": [1, 3, 4, 4]})
    assert result.is_anomalous and result.heatmap.shape == (6, 8)
    assert (tmp_path / settings["cache_dir"]).is_dir()


def test_detector_config_selection(config, tmp_path, monkeypatch):
    assert type(build_detector(config, tmp_path)).__name__ == "StubDetector"
    config["models"]["detector"]["implementation"] = "openvino"
    constructor = Mock(return_value="mock-detector")
    monkeypatch.setattr("anomaly.detector.OpenVINODetector", constructor)
    assert build_detector(config, tmp_path) == "mock-detector"
    config["models"]["detector"]["implementation"] = "typo"
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector(config, tmp_path)


def test_data_discovery_and_export_paths(tmp_path):
    with pytest.raises(ValueError, match="two good"):
        discover_images(tmp_path)
    (tmp_path / "good").mkdir()
    for index in range(2):
        Image.new("RGB", (4, 4)).save(tmp_path / f"good/{index}.png")
    assert len(discover_images(tmp_path)["good"]) == 2
    assert discover_images(tmp_path)["defect"] == []
    xml = tmp_path / "model.xml"
    xml.write_text("mock")
    assert export_xml_path(xml) == xml
    assert export_xml_path(tmp_path) == xml
    with pytest.raises(ValueError):
        export_xml_path(None)


def test_lightweight_imports_and_fake_run_without_ml(tmp_path):
    # Block heavy imports even though an earlier session installed those packages.
    # Run a subprocess so test plugin imports cannot mask missing dependencies.
    script = '''
import importlib.abc, sys, asyncio
from pathlib import Path
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','anomalib','openvino','openvino_genai','viam'}:
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockML())
import anomaly.train, anomaly.detector, vlm.decider, vlm.try_vlm
import main, yaml
cfg = yaml.safe_load(Path('config/config.yaml').read_text())
cfg['logs_dir'] = sys.argv[1]
asyncio.run(main.run(cfg, 1, True))
cfg['models']['decider']['implementation'] = 'rules'
asyncio.run(main.run(cfg, 2, True))
'''
    # Use a project-relative log directory because main prints paths relative to ROOT.
    result = subprocess.run([sys.executable, "-c", script, "logs/lightweight-test"],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
