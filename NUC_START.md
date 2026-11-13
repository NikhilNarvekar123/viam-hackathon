# NUC setup

Keep this project separate from organizer-managed software and `/opt/viam`.

## Lightweight simulation environment

```bash
cd ~/viam-hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml pytest python-dotenv
python main.py --fake --n 10
pytest
```

If system Python lacks venv/ensurepip, use an organizer-approved Python interpreter
to create an isolated venv. Do not overwrite the organizer's ML environment.

## Hardware integration

Install `viam-sdk` in the project environment when ready for camera or hardware work.
Copy `.env.example` to `.env` and enter your own machine address and API credentials.
Never commit credentials. The public snapshot contains none.

Camera-only object localization is documented in LOCALIZATION.md. Confirm camera
resource names for your station. Its output is image pixels, not robot targets.

## Safety and current limitations

- Models default to stubs. Real ML integrations have VERIFY markers.
- All checked-in arm poses are placeholders. Do not run the main pipeline with
  --confirm-motion until real poses and safety settings have been validated.
- Automatic camera-to-robot calibration and scattered-object grasping are unfinished.
- The gripper demo is separate, fake by default, and requires explicit flags for
  jaw movement. Stop teleop and clear the gripper before any real test.
- Use organizer-approved speeds, a clear workspace, and an accessible emergency stop.
- Do not run our full requirements install into an existing organizer environment;
  its versions may differ from the project's requirements.

Images, logs, models, .venv, and .env are gitignored.
