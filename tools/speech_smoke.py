#!/usr/bin/env python3
"""Local-only recorded speech test. No robot, microphone, network or motion imports."""
import argparse
from array import array
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import sys
import time
import wave


def read_audio(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Expected uncompressed 16 kHz mono signed 16-bit PCM WAV; no implicit resampling")
        count = wav.getnframes()
        if not 0 < count <= 30 * 16000:
            raise ValueError("Clip must be nonempty and at most 30 seconds")
        raw = wav.readframes(count)
        if len(raw) != count * 2:
            raise ValueError("Truncated WAV")
    samples = array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return [x / 32768.0 for x in samples]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(ov, genai):
    core = ov.Core()
    devices = {}
    for device in core.available_devices:
        try:
            devices[device] = str(core.get_property(device, "FULL_DEVICE_NAME"))
        except Exception as exc:
            devices[device] = f"query unavailable: {exc}"
    versions = {}
    for name in ("openvino", "openvino-genai", "openvino-tokenizers", "optimum-intel", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cls = getattr(genai, "WhisperPipeline", None)
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "versions": versions, "devices": devices,
        "whisper_api": None if cls is None else {
            "constructor_doc": cls.__init__.__doc__,
            "generate_doc": cls.generate.__doc__,
            "public_members": [name for name in dir(cls) if not name.startswith("_")],
        },
    }


def benchmark(args, ov, genai, report):
    # Explicit single-device selection only: never AUTO/HETERO/MULTI or retry on CPU.
    if args.device not in ("NPU", "CPU"):
        raise ValueError("Only explicit NPU or CPU is supported")
    if args.device not in report["environment"]["devices"]:
        raise RuntimeError(f"Requested {args.device} is unavailable; refusing fallback")
    audio = read_audio(args.wav)
    model = args.model.resolve()
    required = ("openvino_encoder_model.xml", "openvino_encoder_model.bin",
                "openvino_decoder_model.xml", "openvino_decoder_model.bin", "generation_config.json")
    for name in required:
        if not (model / name).is_file():
            raise ValueError(f"Missing model artifact: {name}")
    report["model"] = {"path": str(model), "sha256": {
        p.name: file_hash(p) for p in sorted(model.iterdir()) if p.is_file()
    }}
    report["audio"] = {"path": str(args.wav), "sha256": file_hash(args.wav), "seconds": len(audio) / 16000}
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    report["cache"] = {"path": str(cache), "files_before": sum(p.is_file() for p in cache.rglob("*"))}
    report["execution_evidence"] = {
        "requested_device": args.device,
        "automatic_device_selection": False,
        "application_cpu_retry": False,
        "physical_execution_verified": False,
        "pipeline_execution_devices": None,
        "limitation": "WhisperPipeline does not expose its internal compiled models through the documented API. Device detection and successful construction are not physical execution proof. CPU preprocessing is expected. Correlate external NPU telemetry with this run; do not infer absence of internal fallback.",
    }
    report["stage"] = "pipeline_construction"
    start = time.perf_counter()
    pipe = genai.WhisperPipeline(str(model), args.device, CACHE_DIR=str(cache))
    report["load_compile_or_cache_import_ms"] = (time.perf_counter() - start) * 1000
    # This includes loading and cache import; the API does not isolate compiler time.
    config = pipe.get_generation_config()
    config.max_new_tokens = args.max_new_tokens
    report["generation"] = {"max_new_tokens": args.max_new_tokens}
    report["stage"] = "warmup"
    start = time.perf_counter()
    pipe.generate(audio, config)
    report["warmup_ms"] = (time.perf_counter() - start) * 1000
    report["stage"] = "warm_inference"
    runs = []
    for _ in range(args.runs):
        start = time.perf_counter()
        result = pipe.generate(audio, config)
        elapsed = (time.perf_counter() - start) * 1000
        runs.append({"wall_ms": elapsed, "transcripts": list(result.texts)})
    report["runs"] = runs
    report["warm_median_ms"] = statistics.median(r["wall_ms"] for r in runs)
    report["warm_realtime_factor"] = report["warm_median_ms"] / (len(audio) / 16)
    report["cache"]["files_after"] = sum(p.is_file() for p in cache.rglob("*"))
    report["status"] = "transcribed_device_execution_unverified"
    report["stage"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Inspect installed API/devices only")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--device", choices=("NPU", "CPU"), default="NPU")
    parser.add_argument("--cache", type=Path, default=Path("models/speech-cache"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-device-evidence", action="store_true",
                        help="Exit 2 if transcription succeeds but physical execution remains unverified")
    args = parser.parse_args(argv)
    if not args.probe and (args.model is None or args.wav is None):
        parser.error("--model and --wav are required unless --probe")
    if args.runs < 1 or args.max_new_tokens < 1:
        parser.error("--runs and --max-new-tokens must be positive")
    report = {"status": "failed", "stage": "imports", "started_unix": time.time()}
    code = 0
    try:
        import openvino as ov
        import openvino_genai as genai
        report["environment"] = inventory(ov, genai)
        if args.probe:
            report["status"] = "probe_only"
        else:
            benchmark(args, ov, genai, report)
            if args.require_device_evidence:
                code = 2
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    report["finished_unix"] = time.time()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
