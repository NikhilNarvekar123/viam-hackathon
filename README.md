# Viam Fastener Inspection Hackathon

Work-in-progress local fastener inspection for Robot Hackathon: Edge of Intelligence,
Track 3: Spot it, sort it. **Model accuracy and autonomous sorting are not validated.**

## Current pipeline

The default `main.py` loop repeats until Ctrl+C:

1. Reach/verify the organizer's saved `home-pose`.
2. Fully open the gripper and verify its reported position.
3. Capture `overview-cam` and count candidate objects.
4. Capture wrist camera `cam`, detect contours, isolate square object crops, and run
   the local OpenVINO Anomalib model once per candidate.
5. Scores **below 0.75** route to GOOD; other candidates escalate to the VLM stage.
   VLM is enabled in this snapshot; unavailable/invalid responses remain HOLD.
6. Use supplied object ROIs and shared stationary RGB/depth to locate pickup targets.
   Process the batch tallest measured height first, using the original verdicts:
   ACCEPT → GOOD area; REJECT → pick up and return to the pickup location.
   HOLD or unusable geometry → no grasp. No second neural classification pass.
7. Save artifacts and pause for Enter after the complete batch.

The anomaly model is compiled once per run on CPU. The optional Stage 2 service is
Viam `mainVLM`, running Qwen2.5-VL-7B-Instruct INT4 locally through OpenVINO GenAI on
Intel GPU. The configured prompt checks surface condition, foreign material and
physical damage, returning ordered `observation` and boolean `defect` fields.
Valid true -> REJECT; valid false -> GOOD; invalid/error responses -> HOLD.

**The default live loop now attempts physical batch picking/sorting.**
Use `--inspect-only` to suppress picking. `--n 1` means one complete batch, not one
object. The latest direct-ROI/tallest-first update is offline-tested but has **not
been deployed to the station or physically tested**. See [INSPECTED_PICKING.md](INSPECTED_PICKING.md).
Overview/wrist IDs are independent and frame-local. A GOOD decision is a model
output, not a verified ground-truth label. The 0.75 threshold is an operating choice,
not a validated calibration. Both false positives and false accepts have occurred.

## Image and model contract

- Segmentation thresholds use a canonical 640x480 resolution. Border, texture and
  fragment filters reduce false candidates; touching objects can still merge.
- Centered square crop side:
  `max(int(36*w/640), int(max(bbox_width,bbox_height)*1.45))`.
- Other detected contours, including interior holes, are filled with median tabletop
  color after four-reference-pixel dilation. Target pixels are protected.
- Boundary-crossing square windows stop inspection rather than adding an outer pad.
- Automatic crop sharpening is **disabled** after comparison showed no separation
  benefit. A noise-gated Gaussian unsharp helper remains for controlled experiments.
- Anomalib: RGB /255, float32 NCHW `[1,3,128,128]`. The current export embeds resize
  to 256x256 and ImageNet normalization. Shape/dtype mismatches are rejected rather
  than hidden by reshaping the model. Check your own export before using this setup.
- VLM: the isolated crop is resized to 224x224 with OpenCV `INTER_LANCZOS4`, then
  encoded as lossless PNG. No heatmap, annotation or anomaly score is supplied.
  Upsampling does not recover detail missing from the original capture.
- Scores/maps may be normalized and clamped by the export. A score of 1 is not a
  probability and does not imply every pixel in the anomaly map is saturated.

## Safe quick start — no robot or weights required

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow opencv-python-headless pyyaml pytest python-dotenv viam-sdk
python main.py --fake --n 2
pytest
```

Fake mode uses synthetic scenes and stub inference, NOT real model predictions.
Legacy sorting simulation remains available with `python main.py --stage sort --fake --n 10`.

## Hardware setup

Keep a separate project environment; preserve organizer environments and `/opt/viam`.
Install OpenVINO for real anomaly inference. Full `requirements.txt` also includes
training and local GenAI dependencies; it is not the lightweight fake-only install.
Tested hardware environment: Python 3.12, viam-sdk 0.85.2, OpenVINO 2026.4.1.

Copy `.env.example` to `.env` and supply your own credentials. Review resource names
and paths in `config/config.yaml`, `config/wrist_inspection.yaml`, and
`config/vlm_service.yaml`. Supply privately:

- An empty-table overview reference at the configured path, with matching viewpoint
  and lighting.
- Your OpenVINO `.xml`/`.bin` anomaly export under `models/`.
- An organizer `home-pose` switch (`erh:vmodutils:arm-position-saver`), arm and gripper.
- For Stage 2: a configured local Viam vision service and downloaded Qwen model.
  Set `enabled: true` only after checking service health and validating behavior.

With a clear workspace, no competing controller, and emergency-stop access:

```bash
python main.py --confirm-motion --n 1
```

**This homes/opens the gripper and can pick/place multiple objects.** Home replay may use direct
joint motion rather than collision planning. Requests use 5 deg/s speed and
10 deg/s² acceleration. Validate the saved home/path first. Default live mode does
not use the placeholder project poses and will not skip mandatory home verification.
Omit `--n` for repeated batches with Enter between batches. Keep the scene unchanged
within a batch and check/clear the GOOD area between batches. Errors stop the loop.
`--pick-target-source picker` retains the older independent picker selection mode.

## Saved-frame diagnostics — no motion

```bash
python tools/replay_wrist.py --image /path/to/frame.png
python tools/inspect_anomaly_tensors.py --image /path/to/frame.png
python tools/ablate_anomaly.py --image /path/to/frame.png
python tools/diagnose_anomaly.py --labels /path/to/labels.json
# Optional explicit reference images; no dataset or ground-truth labels are embedded:
python tools/diagnose_anomaly.py --sample /path/to/reference.png --labels /path/to/labels.json
```

`replay_wrist.py --vlm` invokes the decision stage, honoring the configured enabled flag.
The tensor inspector records direct exported tensors before heatmap visualization.
The ablation tool compares masking/sharpening combinations. Diagnostic output stays
under gitignored `logs/`; calibration proposals never overwrite config automatically.

Other tools:
- `tools/inspect_wrist.py`: camera capture/preview or offline inference, automatic/manual ROIs.
- `tools/test_vlm.py`: standalone inspection using the same gating and routing as main.
- `tools/locate.py` / `tools/capture.py`: pixel localization and camera-only capture.
- `tools/gripper_demo.py`: fake-by-default gripper demo with explicit motion opt-in.
- `tools/overview_hover.py`: **experimental, unvalidated** mapping and hover planning;
  a physical test went to the wrong location. Not used by the main loop.

## Safety and limitations

- VLM and anomaly accuracy remain unvalidated across representative parts/views.
  Clean and defective anomaly scores can overlap; do not hide this by threshold tuning.
- Large/boundary/low-texture candidates may be excluded; clutter may still be counted.
- `config/poses.json` contains placeholders. Do not run the legacy real sorter with them.
- Model collision checks are not personnel-safety systems. Add parts only during safe
  paused/loading windows. Never run competing motion controllers.
- A browser's live vision polling can accumulate outstanding Viam requests and hit
  concurrency limits. This is not a local model token quota. Stop excess polling
  and investigate service responsiveness rather than blindly increasing timeouts.
- Monitor disk space; per-cycle image artifacts accumulate.

**297 tests passed** for this snapshot. Tests are primarily synthetic/mocked; real
OpenVINO inference and Viam integration were tested separately. A finished autonomous
inspection-and-sorting robot is not claimed.

See [NUC_START.md](NUC_START.md), [LOCALIZATION.md](LOCALIZATION.md), and
[OVERVIEW_HOVER.md](OVERVIEW_HOVER.md). This sanitized public snapshot excludes
credentials, weights, images, labels/calibration artifacts, runtime logs, virtual
environments, and private Git history.
