# Viam Fastener Inspection Hackathon

Work-in-progress fastener inspection/sorting project for Robot Hackathon: Edge of
Intelligence, Track 3: Spot it, sort it. Tape and marker marks represent defects.

## Current pipeline

The default `main.py` loop repeats until Ctrl+C:

1. Reach/verify the organizer's saved `home-pose`.
2. Fully open the gripper and verify its reported position.
3. Capture `overview-cam`, detect/count candidate objects, and save annotations.
4. Capture wrist camera `cam`, automatically crop candidate objects, and run the
   local OpenVINO anomaly model. Save scores, heatmaps, and overlays.
5. Wait the configured interval and repeat with fresh images.

The model is compiled once per run. Current inference device is CPU; GPU/NPU
benchmarking is pending. Initial anomaly threshold is 0.4. Scores are not probabilities.
VLM decisions, cross-camera object matching, and physical sorting are **not**
integrated into this loop. Wrist/overview IDs are independent and frame-local.

## Safe quick start (no robot or model required)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml pytest python-dotenv
python main.py --fake --n 2
pytest
```

Fake mode uses synthetic scenes and stub anomaly scores, NOT real model inference.
Omit `--n` for a continuous loop. Legacy simulation remains available via:

```bash
python main.py --stage sort --fake --n 10
```

## Hardware setup

In a separate project environment, install `viam-sdk` and `openvino`. Tested on the
NUC with Python 3.12, viam-sdk 0.85.2, and OpenVINO 2026.4.1.
Copy `.env.example` to `.env` and enter your own credentials. Configure resource
names in `config/config.yaml` and `config/wrist_inspection.yaml`.

The repository excludes all runtime data. Supply:
- An empty-table overview reference at the configured `overview.background` path,
  captured with the current fixed camera viewpoint, lighting, and exposure.
- Your OpenVINO `.xml`/`.bin` model pair under `models/`. Update the model path.
  The team's v2 export expects float32 RGB NCHW 1x3x128x128, returning pred_score
  and anomaly_map. Current external preprocessing is RGB / 255; confirm the full
  preprocessing contract with the trainer.
- An organizer-configured `home-pose` switch implementing
  `erh:vmodutils:arm-position-saver`, plus arm and gripper resources.

After setup, with the workspace clear, competing teleop stopped and emergency stop
available, a supervised one-cycle run is:

```bash
python main.py --confirm-motion --n 1
```

This DOES move the arm home and open the gripper. The organizer switch replays
joints directly when no motion service is configured on it; it is not guaranteed
collision-planned. The script requests 5 deg/s speed and 10 deg/s² acceleration.
It never uses the placeholder project poses for this default home stage.
Omit `--n` for continuous operation. Live mode refuses to skip homing; fake and
offline overview analysis remain available. Errors exit rather than advancing.

## Standalone tools

- `tools/inspect_wrist.py`: wrist capture/inference, automatic or manual ROI crops,
  offline images, and preview-only capture.
- `tools/locate.py`: overview background subtraction, crops, and pixel coordinates.
- `tools/capture.py`: camera-only capture helper.
- `tools/gripper_demo.py`: fake-by-default gripper open/close/open demo.
- `tools/overview_hover.py`: **experimental** provisional overview-to-robot mapping
  and hover planning. A physical test was reported to land in the wrong place.
  Treat this path as unvalidated and do NOT use it for unattended motion.

## Limitations and safety

- Automated cropping/counting is heuristic: shadows/clutter can be false positives;
  touching objects can merge, and large/boundary candidates may be rejected.
- Live accuracy is not established; real captures have produced false positives.
  V2 and the 0.4 threshold need labeled validation, not just successful inference.
- The overview mapping is provisional. Its calibration artifacts are excluded,
  and the main loop does NOT use it to move toward objects.
- `config/poses.json` contains placeholders. Do not execute the legacy real sorter
  until all poses, paths and model settings have been independently validated.
- Model collision checks are not personnel-safety systems. Add/rearrange parts only
  during a safe paused/loading phase. Do not run competing controllers.
- Logs/images accumulate each cycle; monitor disk space during long runs.

174 tests passed on the NUC at this snapshot. Tests are primarily synthetic/mocked;
model loading/inference was also tested on saved images. A finished autonomous
inspection-and-sorting robot is not claimed.

See [NUC_START.md](NUC_START.md), [LOCALIZATION.md](LOCALIZATION.md), and
[OVERVIEW_HOVER.md](OVERVIEW_HOVER.md). Credentials, weights, images, calibration
captures, logs, virtual environments, and private Git history are excluded.
