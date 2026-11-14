# Isolated device benchmarks

These standalone scripts do not import production robot code or change `main.py`,
production configs, model weights, packages, or drivers. Run with the existing
project virtual environment. Do not run inspections or browser VLM polling during
measurement. No camera captures or robot motion are requested.

## Files

- `benchmark_anomalib_devices.py`: CPU/GPU/NPU compile and inference testing on one
  saved real crop. CPU reference comparisons, separate worker processes, timeout,
  RAM/disk headroom monitoring, and production file hash verification. **GPU uses
  explicit FP32** (`INFERENCE_PRECISION_HINT=f32`): default precision produced a
  large anomaly-map disagreement in testing. CPU/NPU precision is left unchanged.
- `benchmark_vlm.py`: benchmark the existing GPU-hosted Viam VLM using sequential
  saved-image requests with the production prompt/preprocessing. First request,
  three additional warm-ups, then 30 timed requests. Stops on request errors without
  retries. It does not restart or reconfigure the service.
- `benchmark_vlm_npu.py`: separate direct GenAI NPU load/compile and one-generation
  attempt using the saved GPU benchmark image/prompt. Five-minute timeout and
  headroom monitoring. It leaves the GPU service alone. Success alone does not
  independently verify placement of every pipeline component.
- `test_device_benchmark.py`: mocked harness tests, not hardware benchmarks.

## Setup

From this repository checkout, copy just the scripts to a separate working area:

```bash
mkdir -p ~/viam-device-bench
cp benchmarks/benchmark*.py ~/viam-device-bench/
cd ~/viam-device-bench
```

Activate the existing production venv if needed; **do not install or upgrade
packages**. Use the project path for your machine in the initialization command:

```bash
python benchmark_anomalib_devices.py init --root /home/intel/viam
python benchmark_anomalib_devices.py run CPU
python benchmark_anomalib_devices.py run GPU
python benchmark_anomalib_devices.py run NPU
python benchmark_anomalib_devices.py verify
```

Initialization reads the configured anomaly export and selects a saved crop from
a real inspection. It snapshots production source/config, model and input hashes.
If a baseline already exists, initialization refuses to overwrite it. Existing
state/reference files from the nano walkthrough are supported. Do not delete or
replace a baseline simply to suppress a file-integrity failure.

For the GPU VLM service baseline, followed by the isolated NPU attempt:

```bash
python benchmark_vlm.py
python benchmark_vlm_npu.py
```

The VLM scripts expect the default `~/viam-device-bench` working area, the Anomalib
baseline/verifier, and—for the NPU test—a completed GPU VLM report. Credentials are
read from the existing environment/project `.env`, never printed. VLM model assets
are hashed before/after; weights are not copied. All generated reports, reference
outputs and image payloads belong outside this repository.

## Interpretation and limits

- A failed compile is a result for this export/runtime/driver combination, not
  proof that every model is unsupported on that device. A timeout is not proof of
  incompatibility. Native compiler crashes are recorded as aborted workers.
- Anomalib timing is model inference, excluding crop preparation and robot stages.
  Persistent OpenVINO caching is disabled; OS/driver caches are not globally reset.
- GPU VLM timing includes service transport and generation; the service is already
  loaded, so this is not a cold-start benchmark. Direct GenAI NPU first-generation
  timing is a different measurement, not a warmed service-latency comparison.
- Repeated inference on one crop measures speed/repeatability, **not accuracy**.
  Valid-response rate and repeated action consistency do not establish correctness.
- RSS does not include all device/driver memory. Headroom polling is a precaution,
  not a hard memory reservation or an absolute OOM guarantee.
- A small compressed CPU output reference is saved (16 MiB uncompressed cap).
  Worker temporary/cache directories are removed after process exit. Model loading
  and compiler temporary storage can still consume RAM/disk during a run.
- On VLM service timeout, stop and investigate: the server request may continue
  even though the benchmark stopped. Do not start overlapping retries.

Offline harness test:

```bash
python -m pytest benchmarks/test_device_benchmark.py -q
```

Keep `.env`, saved crops, weights, raw outputs and machine logs private. None are
included in this folder.
