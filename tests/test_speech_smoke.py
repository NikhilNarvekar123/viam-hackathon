"""Dependency-free fake tests; no OpenVINO or robot imports."""
import argparse
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

from tools import speech_smoke as smoke
from tools import fetch_speech_model as fetcher


class SpeechSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wav = self.root / "command.wav"
        self.write_wav()

    def write_wav(self, rate=16000, channels=1, frames=160):
        with wave.open(str(self.wav), "wb") as wav:
            wav.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<h", -16384) * frames * channels)

    def test_audio_normalization(self):
        self.assertEqual(smoke.read_audio(self.wav), [-0.5] * 160)

    def test_bad_audio(self):
        for rate, channels, frames in [(8000, 1, 160), (16000, 2, 160), (16000, 1, 0), (16000, 1, 480001)]:
            with self.subTest(rate=rate, channels=channels, frames=frames):
                self.write_wav(rate, channels, frames)
                with self.assertRaises(ValueError):
                    smoke.read_audio(self.wav)

    def setup_benchmark(self, fail=False):
        model = self.root / "model"
        model.mkdir()
        for name in ("openvino_encoder_model.xml", "openvino_encoder_model.bin", "openvino_decoder_model.xml", "openvino_decoder_model.bin", "generation_config.json"):
            (model / name).write_text("fake")
        calls = []

        class Pipeline:
            def __init__(self, path, device, **kwargs):
                calls.append((device, kwargs))
                if fail:
                    raise RuntimeError("NPU compile failed")

            def get_generation_config(self):
                return types.SimpleNamespace()

            def generate(self, audio, config):
                calls.append("generate")
                return types.SimpleNamespace(texts=["pause"])

        args = argparse.Namespace(model=model, wav=self.wav, cache=self.root / "cache", device="NPU", runs=3, max_new_tokens=64)
        return args, types.SimpleNamespace(WhisperPipeline=Pipeline), calls

    def test_fake_benchmark_reports_unverified(self):
        args, genai, calls = self.setup_benchmark()
        report = {"environment": {"devices": {"NPU": "fake"}}}
        smoke.benchmark(args, None, genai, report)
        self.assertEqual(calls[0][0], "NPU")
        self.assertIn("CACHE_DIR", calls[0][1])
        self.assertEqual(calls.count("generate"), 4)
        self.assertEqual(len(report["runs"]), 3)
        self.assertEqual(report["runs"][0]["transcripts"], ["pause"])
        self.assertFalse(report["execution_evidence"]["physical_execution_verified"])
        self.assertIn("load_compile_or_cache_import_ms", report)
        self.assertIn("warmup_ms", report)

    def test_compile_failure_never_retries(self):
        args, genai, calls = self.setup_benchmark(fail=True)
        with self.assertRaisesRegex(RuntimeError, "NPU compile failed"):
            smoke.benchmark(args, None, genai, {"environment": {"devices": {"NPU": "fake"}}})
        self.assertEqual(len(calls), 1)

    def test_unavailable_device_fails_before_constructing(self):
        args, genai, calls = self.setup_benchmark()
        with self.assertRaisesRegex(RuntimeError, "refusing fallback"):
            smoke.benchmark(args, None, genai, {"environment": {"devices": {"CPU": "fake"}}})
        self.assertEqual(calls, [])

    def test_large_download_refused_before_creating_directory(self):
        metadata = {"sha": fetcher.REVISION, "siblings": [{"rfilename": "model.bin", "size": fetcher.LIMIT + 1}]}
        import io
        target = self.root / "download"
        with patch.object(fetcher.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(metadata))):
            with self.assertRaisesRegex(ValueError, "approval"):
                fetcher.fetch(fetcher.REPO, fetcher.REVISION, target)
        self.assertFalse(target.exists())

    def test_download_requires_pinned_revision(self):
        with self.assertRaises(ValueError):
            fetcher.fetch(fetcher.REPO, "main", self.root / "download")


if __name__ == "__main__":
    unittest.main()
