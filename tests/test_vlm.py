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

from core.factory import build_decider
from core.types import Detection, Frame
from vlm.decider import OpenVINODecider
from vlm.overlay import build_overlay
from vlm.prompt import build_prompt
from vlm.rules import RulesDecider

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())


def test_overlay_pixels_and_input_unchanged():
    image = Image.new("RGB", (2, 2), (100, 100, 100))
    heatmap = np.array([[0, 1], [0, 1]], dtype=np.float32)
    output = np.asarray(build_overlay(image, heatmap, alpha=1))
    assert output.shape == (2, 4, 3)
    np.testing.assert_array_equal(output[:, :2], np.asarray(image))
    np.testing.assert_array_equal(output[0, 2], [100, 100, 100])
    np.testing.assert_array_equal(output[0, 3], [255, 0, 0])
    assert image.getpixel((1, 1)) == (100, 100, 100)
    np.testing.assert_array_equal(heatmap, [[0, 1], [0, 1]])


def test_missing_constant_and_resized_heatmaps():
    image = Image.new("L", (4, 3), 100)
    missing = np.asarray(build_overlay(image, None))
    np.testing.assert_array_equal(missing[:, :4], missing[:, 4:])
    zero = np.asarray(build_overlay(image, np.zeros((1, 1))))
    np.testing.assert_array_equal(zero, missing)
    high = np.asarray(build_overlay(image, np.full((1, 1), 20), alpha=1,
                                   value_min=10, value_max=20))
    assert np.all(high[:, 4:] == [255, 0, 0])


@pytest.mark.parametrize("kwargs", [{"alpha": -1}, {"alpha": float("nan")},
    {"value_min": 1, "value_max": 1}])
def test_invalid_overlay_settings(kwargs):
    with pytest.raises(ValueError):
        build_overlay(Image.new("RGB", (2, 2)), None, **kwargs)


@pytest.mark.parametrize("heatmap", [np.zeros((1, 2, 3)), np.array([[np.nan]]), np.empty((0, 2))])
def test_invalid_heatmap(heatmap):
    with pytest.raises(ValueError):
        build_overlay(Image.new("RGB", (2, 2)), heatmap)


def test_prompt_json_braces_and_required_instructions():
    template = 'Types: {defect_types}, actions: {allowed_actions}. Example schema: {"action":"reject"}'
    prompt = build_prompt(template, ["scratch"], ["reject"])
    assert '{"action":"reject"}' in prompt
    assert '{defect_types}' not in prompt and '{allowed_actions}' not in prompt
    assert 'JSON only' in prompt and '15 words' in prompt
    example = json.loads(prompt.split('prediction): ')[1])
    assert example["action"] == "reject" and example["defect_type"] == "scratch"


@pytest.mark.parametrize("template,types,actions", [
    ("missing placeholders", ["scratch"], ["reject"]),
    ("{defect_types} {allowed_actions}", [], ["reject"]),
    ("{defect_types} {allowed_actions}", ["scratch"], ["accept"]),
])
def test_invalid_prompt(template, types, actions):
    with pytest.raises(ValueError):
        build_prompt(template, types, actions)


def test_rules_decider():
    decider = RulesDecider(["reject", "review"], "unknown")
    decision = decider.decide(Frame("test", Image.new("RGB", (2, 2))), Detection(0.9, True))
    assert decision.valid and decision.action == "reject"
    assert decision.defect_type == "unknown"
    assert decider.last_latency_ms >= 0
    assert json.loads(decision.raw_output)["action"] == "reject"
    with pytest.raises(ValueError):
        RulesDecider(["accept"], "unknown")


def test_factory_selection_and_load_failure(config, tmp_path, monkeypatch, caplog):
    assert build_decider(config, tmp_path).implementation == "stub"
    settings = config["models"]["decider"]
    settings["implementation"] = "rules"
    assert build_decider(config, tmp_path).implementation == "rules"
    settings["implementation"] = "openvino"
    constructor = Mock(side_effect=RuntimeError("compile failed"))
    monkeypatch.setattr("vlm.decider.OpenVINODecider", constructor)
    assert build_decider(config, tmp_path).implementation == "rules"
    assert "compile failed" in caplog.text and "using rules" in caplog.text
    constructor.side_effect = None
    constructor.return_value = SimpleNamespace(implementation="openvino")
    assert build_decider(config, tmp_path).implementation == "openvino"
    settings["implementation"] = "typo"
    with pytest.raises(ValueError):
        build_decider(config, tmp_path)


def test_invalid_prompt_is_not_silently_fallback(config, tmp_path):
    config["models"]["decider"].update(implementation="openvino", prompt="bad config")
    with pytest.raises(ValueError, match="template"):
        build_decider(config, tmp_path)


@pytest.mark.parametrize("raw,valid", [
    ('{"defect_type":"scratch","action":"reject"}', True),
    ('broken', False),
    ('{"defect_type":"scratch","action":"unsafe"}', False),
])
def test_mocked_genai_warmup_greedy_and_parsing(config, tmp_path, monkeypatch, raw, valid):
    settings = config["models"]["decider"]
    settings.update(path="vlm", device="NPU")
    (tmp_path / "vlm").mkdir()
    core = Mock()
    core.available_devices = ["CPU"]
    tensors = []
    def tensor(array):
        tensors.append(array)
        return array
    pipeline = Mock()
    pipeline.generate.return_value = SimpleNamespace(texts=[raw])
    constructor = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core, Tensor=tensor))
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(
        VLMPipeline=constructor, GenerationConfig=SimpleNamespace))
    decider = OpenVINODecider(settings, config["defect_types"], config["allowed_actions"], tmp_path)
    assert pipeline.generate.call_count == 1  # warmup
    assert decider.device == "CPU"
    assert decider.generation_config.do_sample is False
    assert decider.generation_config.num_beams == 1
    assert decider.generation_config.max_new_tokens == settings["max_new_tokens"]
    frame = Frame("test", Image.new("RGB", (4, 3)))
    decision = decider.decide(frame, Detection(0.9, True, np.ones((3, 4))))
    assert decision.raw_output == raw and decision.valid is valid and decision.action == "reject"
    assert tensors[-1].shape == (1, 3, 4, 3) and tensors[-1].dtype == np.uint8
    assert pipeline.generate.call_count == 2
    constructor.assert_called_once()
    assert decider.last_latency_ms >= 0
    assert (tmp_path / settings["cache_dir"]).is_dir()


def test_try_vlm_cli_rules_and_stub(config, tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (8, 8)).save(image)
    cfg = tmp_path / "config.yaml"
    for implementation in ("stub", "rules"):
        config["models"]["decider"]["implementation"] = implementation
        cfg.write_text(yaml.safe_dump(config))
        result = subprocess.run([sys.executable, str(ROOT / "vlm/try_vlm.py"),
                                 "--config", str(cfg), "--image", str(image)],
                                cwd=tmp_path, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert f"actual={implementation}" in result.stdout
        assert "Prompt:" in result.stdout and "Raw output:" in result.stdout
        assert "Parsed decision:" in result.stdout and "Latency ms:" in result.stdout
        assert "No VLM inference" in result.stdout
