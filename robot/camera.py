"""Camera-only connection for capture tools: never constructs an arm/gripper."""
from contextlib import asynccontextmanager
import logging

from robot.fakes import FakeCamera
from robot.viam_hw import ViamCamera, connect, disconnect, load_credentials


@asynccontextmanager
async def capture_camera(config, root, fake):
    # Always uncropped here: capture/preview applies ROI exactly once.
    if fake:
        yield FakeCamera(root / config["camera"]["samples"],
                         config["fake"]["camera_delay_s"], roi=None)
        return
    credentials = load_credentials(root / ".env")
    from viam.components.camera import Camera

    machine = await connect(credentials)
    try:
        # VERIFY: camera resource name and get_image MIME support on event SDK.
        camera = Camera.from_robot(machine, config["hardware"]["viam"]["camera"])
        yield ViamCamera(camera, roi=None)
    finally:
        try:
            await disconnect(machine)
        except Exception:
            logging.exception("Camera disconnect failed")
