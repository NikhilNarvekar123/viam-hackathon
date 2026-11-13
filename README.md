# Viam Fastener Inspection Hackathon

Work-in-progress entry for **Robot Hackathon: Edge of Intelligence**, Track 3:
**Spot it, sort it**. Inspect scattered fasteners for tape or marker marks and
sort defective items from clean ones.

This is a public snapshot of the project code on the team's NUC. It is **not a
finished autonomous robot controller**. Credentials, images, models, environments,
and private repository history are excluded.

## Quick start (no robot required)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml pytest python-dotenv
python main.py --fake --n 10
pytest
```

## Components

- `robot/locate.py`, `tools/locate.py`: overview-camera background subtraction,
  per-object crops, pixel centers and image-plane orientations.
- `robot/viam_hw.py`: Viam camera, arm and gripper adapters.
- `robot/teach.py`: pose teaching with backups and explicit motion gating.
- `tools/gripper_demo.py`: fake-by-default open/close/open demonstration.
- `anomaly/`: stubs, preprocessing, threshold calibration, training/export and
  OpenVINO detector scaffolding.
- `vlm/`: prompt building, heatmap overlays, JSON validation, rules fallback,
  and OpenVINO GenAI scaffolding.
- `main.py`: sequential capture -> detect -> decide -> sort loop.

## Current design

Use the overview camera to locate objects, then a calibrated robot approach to
obtain top-down wrist-camera images for defect inspection. The calibration and
approach integration are not yet implemented. Localization is standalone, not yet
wired into the main pipeline. Background subtraction can merge nearby objects and
mistake shadows or moving background objects for fasteners.

The fake pipeline and unit tests have been run on the NUC. Model inference,
calibration, and end-to-end physical sorting have not been validated. Read `VERIFY:`
comments before using real integrations.

## Safety

**Checked-in poses are placeholders. Do not execute the real sorting loop with them.**
Never run competing teleop/control programs. Validate calibration, workspace,
speeds, tool offsets, and emergency-stop procedures before any physical motion.
No camera-derived pixel coordinate is a valid robot target by itself.

See [NUC_START.md](NUC_START.md) and [LOCALIZATION.md](LOCALIZATION.md).
