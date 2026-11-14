# Standalone local speech smoke test

No robot imports, connection, motion, microphone capture, action dispatch, or runtime
network access. Do not integrate with the controller until the speech tests and
command-policy fake tests pass. Voice stop is not a physical emergency stop.

## What was actually checked

Development workspace: ARM64 AWS / Neoverse-N1, no OpenVINO, GenAI, NumPy,
Transformers, Optimum, or Torch installed; no exposed `/dev/accel` or `/dev/dri`.
This is **not** the user's Intel robot machine. The user's reported 2026.4.1
runtime and NPU detection have not been independently inspected here.

Downloaded the pre-exported `OpenVINO/whisper-tiny.en-fp16-ov` candidate,
revision `82b00e6b7ce190802f654f891257ccdbc50faadb`, to the ignored
`models/whisper-tiny.en-fp16-ov/` directory (84,227,705 artifact bytes).
No packages were installed or upgraded. **Export/runtime compatibility is
unvalidated.** No command recording was supplied; no real inference, compilation,
accuracy, latency, or NPU execution has been measured. Base.en is deferred.

Seven dependency-free fake tests passed. The standalone probe correctly failed
with `ModuleNotFoundError: No module named 'openvino'`. The existing
`python3 main.py --fake` was attempted but is blocked by missing `yaml` on this
workspace; the full robot pipeline has not been tested here.

## Run on the Intel robot PC

Use its existing working Python environment. Do not install this repository's
requirements over that environment or blindly upgrade OpenVINO. These scripts
add no dependencies beyond the existing OpenVINO and GenAI packages.

First inspect the **installed** versions, device names, constructor/generate
docstrings, and public methods:

```bash
python tools/speech_smoke.py --probe --report logs/speech/probe.json
```

Review this report before inference. The implementation uses `WhisperPipeline`,
`CACHE_DIR`, `get_generation_config()`, `max_new_tokens`, `generate(audio, config)`
and `result.texts`. It deliberately does not assume master-only `ASRPipeline`,
static/stateful selectors, or NPU tuning flags. Missing/unsupported APIs fail
rather than selecting a different implementation or falling back. We could not
validate these APIs against installed 2026.4.1 here. Upstream master sources are
only reference material, not evidence of version compatibility:

- https://github.com/openvinotoolkit/openvino.genai/blob/master/src/python/py_whisper_pipeline.cpp
- https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/automatic_speech_recognition
- https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/

Copy the downloaded model directory to the PC, or provision it there once while
online (inference itself is entirely offline):

```bash
python tools/fetch_speech_model.py
```

The fetcher uses only Python's standard library, pins an immutable revision,
checks total advertised size before downloading, bounds each file's transfer,
and refuses totals over 1 GB. It never overwrites an existing destination.
Failed downloads leave a partial directory; inspect it and choose a new output
path for a retry. No executable model code is downloaded/run. The report records
SHA256 hashes of all local model artifacts and source provenance.

Provide an existing recording, or record a short known command with the system's
already-installed recording tools. Input must be **16 kHz, mono, signed 16-bit
PCM WAV, nonempty, at most 30 seconds**. No automatic resampling or capture-device
selection occurs. Start with push-to-talk recordings, including negatives,
negations, ambiguous speech, and floor noise. Transcription alone does not
implement safe command rejection or prove accuracy.

```bash
python tools/speech_smoke.py \
  --model models/whisper-tiny.en-fp16-ov \
  --wav data/speech/pause.wav --device NPU \
  --cache models/speech-cache/tiny-npu \
  --runs 5 --report logs/speech/tiny-npu.json \
  --require-device-evidence
```

- Exit **1**: failed imports, input/model validation, compilation, or generation.
  Stage and error are recorded; there is no application-level CPU retry.
- Exit **2** with `--require-device-evidence`: transcription completed but physical
  execution is unverified. This is expected with the currently documented Python
  API. It is **not an NPU verification pass**.
- Exit **0** without that flag: probe or transcription completed, **not** a claim
  of physical execution. The report still explicitly marks execution unverified.

Only explicit `NPU` and `CPU` targets are accepted; no AUTO, MULTI, HETERO or GPU.
This diagnostic intentionally overrides the general repo CPU-fallback convention:
a strict NPU test must not pass on CPU. Device and cache paths are CLI-configurable.

### Timing and execution evidence

`load_compile_or_cache_import_ms` measures the entire pipeline constructor:
model loading plus compilation or cache import, **not isolated compiler time**.
`warmup_ms` is one separate generation (which may include deferred compilation).
`runs[].wall_ms` measures complete warm generations, including CPU feature
processing and decoding overhead, not only accelerator kernels. Each run records
its transcript; median latency and real-time factor are reported. A token cap may
truncate speech; inspect transcripts and use short command clips.

Use a fresh cache directory to measure a cold constructor. Re-run with that same
directory to measure cache reuse; file counts are recorded but are not proof of a
cache hit. Keep model/runtime/device caches separate.

Device enumeration, the requested target, and lack of application retry are
reported, but `WhisperPipeline`'s documented API does not expose internal encoder
and decoder compiled-model `EXECUTION_DEVICES`. Do not compile unrelated IRs and
claim their device properties establish this pipeline's execution. Do not treat
CPU-fallback upstream tests as proof. Collect timestamp-correlated NPU telemetry
using tools already available on the PC, and/or version-matched runtime profiling
that identifies encoder and decoder execution. No driver or system changes are
part of this test. CPU capture/preprocessing is expected even with NPU inference.

After the NPU attempt succeeds, run the identical clip as a CPU reference:

```bash
python tools/speech_smoke.py \
  --model models/whisper-tiny.en-fp16-ov \
  --wav data/speech/pause.wav --device CPU \
  --cache models/speech-cache/tiny-cpu \
  --runs 5 --report logs/speech/tiny-cpu.json
```

Compare transcripts against manually labeled commands, not just against each
other. Then obtain a revision-pinned base.en candidate (the fetcher accepts
`--repo`, `--revision`, `--output`), validate compatibility, and repeat the same
clips. If export is necessary, use a separate export venv with CPU Torch and
version-compatible Optimum tooling; no exporter has been installed or validated
in this workstream. Ask before any download exceeding 1 GB.

Benchmark under existing vision load only after standalone validation. Starting
vision/robot code is not part of these scripts and must not accidentally authorize
motion. Preserve existing ENTER/Q pauses and `--confirm-motion` gates.

## Fake checks here

```bash
python3 -m unittest discover -s tests -p test_speech_smoke.py -v
```

These test PCM validation, normalization, unavailable NPU rejection, compile
failure without retry, timing/report separation with a fake pipeline, pinned
revision enforcement, and the download size guard. They do not validate OpenVINO.

## Deferred integration

Microphone/activation → separate ASR process → finalized transcript → configurable
allowlisted action map → expiring/deduplicated command queue → sole robot motion
controller. Reject unknown, ambiguous and negated commands. Keep ASR off the robot
asyncio event loop. Pause only at defined safe cooperative checkpoints, including
between batch objects; status does not move; inspect-again queues when safe;
resume respects paused-state and authorization gates; software stop latches
stopped state without automatic restart. No action mapping or robot changes are
included in this standalone work.
