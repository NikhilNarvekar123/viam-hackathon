# AGENTS.md

## Project

Entry for Viam's "Robot Hackathon: Edge of Intelligence", Track 3: **Spot it, sort it**.

A robot station that inspects items, detects defective ones, decides how to handle each defect, and physically sorts them: good items go into the box, defective items go into the reject bin.

The scenario is scattered fasteners (screws, bolts, nuts, etc.). Tape of any color or marker marks are defects. Keep implementations item-agnostic: item names, defect types, allowed actions, thresholds, camera crop, and arm poses live in config, never in code.

Judging criteria: detection accuracy, correct responses (VLM decisions), latency, how work is split across NPU / GPU / CPU, and a short design explanation.

## Pipeline

1. **Capture**: get a color frame (and optionally depth) from the Viam camera, then crop to the item region of interest (ROI).
2. **Detect**: an Anomalib anomaly model, trained on good images only and exported to OpenVINO IR. Outputs an anomaly score, a heatmap, and pass/fail against a threshold.
3. **Decide**: runs only for flagged items. A VLM (`openvino_genai.VLMPipeline`) receives the crop plus the heatmap overlay and returns JSON with the defect type and an action chosen from the fixed list in config. If the output doesn't parse or the action isn't in the list, use `"reject"`.
4. **Sort**: the Viam Python SDK moves the arm through taught poses: home → pick → box or reject bin → home.

`main.py` runs this loop. Each stage lives in its own module behind a small interface so it can be swapped or faked.

## Hardware and runtime

- Target machine: a Linux mini PC (Intel NUC-style) with an Intel Core Ultra Series 3 processor, which has a CPU, an integrated GPU, and an NPU.
- The arm, gripper, and depth camera are controlled by `viam-server` running on that machine, configured at app.viam.com.
- **All model inference runs locally through OpenVINO on that machine.** No cloud inference APIs, no CUDA (there is no NVIDIA GPU).
- Development also happens on laptops without an NPU. Code must run on CPU anywhere and use GPU/NPU when available.

## Repo layout

```
anomaly/   train + export Anomalib model to OpenVINO, threshold calibration, inference wrapper
vlm/       VLM wrapper, prompt template, JSON parsing + validation
robot/     Viam connection, camera/arm/gripper wrappers, pose teaching tool, fakes.py
bench/     device benchmark: latency of each model on CPU / GPU / NPU
config/    config.yaml (item-specific settings), poses.json (taught arm poses)
main.py    capture → detect → decide → sort loop
data/      training/sample images (gitignored)
models/    exported OpenVINO models (gitignored)
logs/      per-item JSONL logs (gitignored)
```

## Commands

These are the intended entry points. Create them if they don't exist yet.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py --fake            # full loop with fake camera + fake arm
python main.py                   # real hardware (see Safety rules)
python anomaly/train.py --data data/<item> --model patchcore   # train, threshold, export
python bench/devices.py          # latency table for every model on every device
python robot/teach.py <name>     # save the arm's current pose to config/poses.json
```

## Conventions

- Python 3.10+. Keep dependencies minimal, and add every new one to `requirements.txt`.
- **Config, not constants.** `config/config.yaml` holds Viam resource names, ROI, model paths, the device for each model, the anomaly threshold, defect types, allowed actions, and the VLM prompt template.
- **Devices.** Never hardcode `"NPU"` or `"GPU"`. Read the device for each model from config and fall back to CPU if it's missing from `ov.Core().available_devices`, logging a warning.
- **Fake hardware.** `robot/fakes.py` provides `FakeCamera` (cycles through `data/samples/good/` and `data/samples/defect/`) and `FakeArm` / `FakeGripper` (log moves, no motion). The `--fake` flag selects them. Every piece of hardware-touching code needs a fake path.
- **Viam SDK** is async (`asyncio`). Read credentials from environment variables or `.env` (`VIAM_ADDRESS`, `VIAM_API_KEY`, `VIAM_API_KEY_ID`). Never commit them.
- **Logging.** For every item, append one JSON line to `logs/`: item id, timestamp, anomaly score, pass/fail, raw VLM output, parsed action, final bin, and per-stage latency in ms (capture, detect, decide, sort). These logs feed the accuracy and latency numbers in the design explanation.

## OpenVINO notes

- Set `CACHE_DIR` on `ov.Core()` so NPU and GPU models don't recompile on every start.
- The NPU needs static input shapes. Reshape models before compiling for NPU.
- Compile and warm up every model once at startup. Never reload a model per item.
- Use `PERFORMANCE_HINT: LATENCY` for per-item inference.
- VLM: keep `max_new_tokens` small, and prompt for JSON only.
- Before claiming a model runs on NPU or GPU, confirm it compiles on that device and that its outputs match CPU.

## Anomalib notes

- Use the Anomalib 2.x API at the version pinned in `requirements.txt`. Many online examples use the old 0.x/1.x workflow (`config.yaml` + `tools/` scripts), so don't copy those.
- Install CPU PyTorch. Never install CUDA builds.
- Use the `Folder` dataset format: `data/<item>/good/` and optionally `data/<item>/defect/`.
- Candidate models: Patchcore (default), Padim, EfficientAd.
- With few defect samples, set the threshold from the distribution of scores on good images (for example, the maximum good score plus a margin) and record how it was chosen.
- Anomaly models are sensitive to background, lighting, and position. Always crop to the ROI and use identical preprocessing for training and inference.

## Verifying work

- After changing pipeline or hardware code, run `python main.py --fake` and report the real output.
- Don't claim something works on NPU/GPU or on real hardware unless it was actually run there.

## Safety rules

- **Never run code that moves the real arm without explicit confirmation from the user in the current session.** Default to `--fake`.
- Keep arm speeds conservative and move through known waypoints. Always be able to return to the home pose.
- Ask before downloading large models (>1 GB) or installing system packages or drivers.
- Don't modify the `viam-server` config or organizer-installed system software.
- Never commit `data/`, `models/`, `logs/`, `.venv/`, or `.env`.
- Commit, or tell the user to commit, before large refactors.

## Current status

Scenario: scattered fasteners, with tape or marker marks as defects. Teammates are handling Anomalib and the vision-language model; this workstream handles localization and robot integration.

Reference hardware resource names: arm (xArm6), gripper (UFactory), overview-cam (separate webcam), cam (arm-mounted RealSense). Models/SDK adapters are still not fully hardware-verified; poses remain placeholders.

`tools/locate.py` and `robot/locate.py` implement pixel-only background subtraction from the overview camera. Settings are in `config/localization.yaml`. Outputs are frame-local IDs, full-image pixel centers, image-plane long-axis angles, bounding boxes, and individual crops. These are NOT calibrated robot targets or validated grasp points. Touching objects can merge; lighting changes, glare and shadows can create false detections. Real camera accuracy remains to be verified.

Organizer workshop files exist separately under ~/nyc_workshop and ~/workshop, with an Intel-XPU/Anomalib/OpenVINO environment. Their documented robot demo uses simulated SO-101 arms and blocks, not this physical xArm/fastener scenario. Do not overwrite their environment or checkpoints.
