# NUC setup

Keep this project separate from organizer-managed software and `/opt/viam`.

## Lightweight simulation environment

```bash
cd ~/viam-hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pillow opencv-python-headless pyyaml pytest python-dotenv viam-sdk
python main.py --fake --n 2
pytest
```

If system Python lacks venv/ensurepip, use an organizer-approved Python interpreter
to create an isolated venv. Do not overwrite the organizer's ML environment.

## Hardware integration

Install `openvino` in the project environment for real Anomalib inference. The Viam SDK is included above because integration tests also import it. The VLM service is enabled in `config/vlm_service.yaml`; confirm service availability before live use. Invalid/unavailable responses remain HOLD.
Copy `.env.example` to `.env` and enter your own machine address and API credentials.
Never commit credentials. The public snapshot contains none.

Camera-only object localization is documented in LOCALIZATION.md. Confirm camera
resource names for your station. Its output is image pixels, not robot targets.

## Safety and current limitations

- Fake mode and the legacy sorter use stubs. The default live loop uses the
  separately supplied OpenVINO model specified in config/wrist_inspection.yaml.
- All checked-in project poses are placeholders. Do not execute the legacy real
  sorter with them. The default loop instead uses the organizer home switch;
  validate that saved home/path before using --confirm-motion.
- The default main loop now attempts whole-batch picking after inspection; `--n 1` means one batch. Use `--inspect-only` to suppress picking. See INSPECTED_PICKING.md. The latest direct-ROI/tallest-first changes have not been deployed to the station or physically validated.
- The gripper demo is separate, fake by default, and requires explicit flags for
  jaw movement. Stop teleop and clear the gripper before any real test.
- Use organizer-approved speeds, a clear workspace, and an accessible emergency stop.
- Do not run our full requirements install into an existing organizer environment;
  its versions may differ from the project's requirements.

Images, logs, models, .venv, and .env are gitignored.
