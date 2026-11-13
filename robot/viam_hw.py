"""Viam adapters. SDK imports are lazy; live APIs remain unverified."""
import asyncio
import logging
from io import BytesIO
import os
from pathlib import Path

import numpy as np
from PIL import Image

from core.types import Frame

logger = logging.getLogger(__name__)


def load_credentials(env_file: Path) -> dict:
    from dotenv import load_dotenv

    load_dotenv(env_file, override=False)
    names = ("VIAM_ADDRESS", "VIAM_API_KEY", "VIAM_API_KEY_ID")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("missing .env values: " + ", ".join(missing))
    return {name: os.environ[name].strip() for name in names}


async def connect(credentials):
    from viam.robot.client import RobotClient

    # VERIFY: Options.with_api_key and at_address signatures on organizer SDK.
    options = RobotClient.Options.with_api_key(
        api_key=credentials["VIAM_API_KEY"], api_key_id=credentials["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(credentials["VIAM_ADDRESS"], options)


def crop_image(image, roi):
    if roi is None:
        return image
    left, top, right, bottom = roi
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError("ROI must fit inside the camera image")
    return image.crop(roi)


def decode_depth(payload: bytes) -> np.ndarray:
    # VERIFY: organizer camera supports image/vnd.viam.dep, DEPTHMAP header,
    # big-endian uint64 width/height followed by uint16 millimeter depths.
    if len(payload) < 24 or payload[:8] != b"DEPTHMAP":
        raise ValueError("Unsupported Viam depth encoding; verify camera MIME format")
    width = int.from_bytes(payload[8:16], "big")
    height = int.from_bytes(payload[16:24], "big")
    if not width or not height or len(payload) != 24 + width * height * 2:
        raise ValueError("Invalid Viam depth dimensions or payload length")
    return np.frombuffer(payload, dtype=">u2", offset=24).reshape(height, width).copy()


def select_image(images, *, depth=False):
    supported = {"image/vnd.viam.dep"} if depth else {
        "image/png", "image/jpeg", "image/vnd.viam.rgba"}
    matches = [image for image in images
               if str(image.mime_type).removesuffix("+lazy") in supported]
    if len(matches) != 1:
        streams = [(getattr(image, "name", ""), str(image.mime_type)) for image in images]
        raise ValueError(f"Expected one {'depth' if depth else 'color'} stream; got {streams}")
    return matches[0]


def decode_color(image):
    mime = str(image.mime_type).removesuffix("+lazy")
    if mime == "image/vnd.viam.rgba":
        data = image.data
        if len(data) < 12 or data[:4] != b"RGBA":
            raise ValueError("Invalid Viam RGBA header")
        width = int.from_bytes(data[4:8], "big")
        height = int.from_bytes(data[8:12], "big")
        if not width or not height or len(data) != 12 + width * height * 4:
            raise ValueError("Invalid Viam RGBA payload length")
        return Image.frombytes("RGBA", (width, height), data[12:]).convert("RGB")
    with Image.open(BytesIO(image.data)) as encoded:
        return encoded.convert("RGB")


class ViamCamera:
    def __init__(self, camera, depth_camera=None, roi=None):
        self.camera = camera
        self.depth_camera = depth_camera
        self.roi = roi
        self.index = 0

    async def get_frame(self) -> Frame:
        # viam-sdk 0.85.2: get_images returns (Sequence[NamedImage], metadata).
        # Select by MIME rather than assuming the first stream is color.
        images, _metadata = await self.camera.get_images()
        image = decode_color(select_image(images))
        depth = None
        if self.depth_camera is not None:
            depth_images = images
            if self.depth_camera is not self.camera:
                depth_images, _ = await self.depth_camera.get_images()
            depth = decode_depth(select_image(depth_images, depth=True).data)
            # Depth is deliberately uncropped: alignment to RGB is not assumed.
            # VERIFY: depth units, RGB registration, and timestamp synchronization.
        image = crop_image(image, self.roi)
        self.index += 1
        return Frame(f"item-{self.index:06d}", image, depth, None)


class ViamArm:
    def __init__(self, arm, poses, settings, confirm_motion=False):
        self.arm = arm
        self.poses = poses
        self.settings = settings
        self.confirm_motion = confirm_motion

    def _require_motion(self):
        if not self.confirm_motion:
            raise PermissionError("Arm motion requires --confirm-motion")

    async def move_to(self, pose_name):
        self._require_motion()
        pose = self.poses[pose_name]
        if pose.get("source") == "fake":
            raise ValueError("Cannot move real hardware to a fake-taught pose")
        from viam.proto.component.arm import JointPositions

        joints = pose["joint_positions_deg"]
        # VERIFY: joint angles are degrees; supported speed controls are driver-specific.
        # Conservative velocity must be configured/verified on hardware before use.
        await self.arm.move_to_joint_positions(JointPositions(values=joints))

    async def move_to_xyz(self, x, y, z):
        self._require_motion()
        from viam.proto.common import Pose

        orientation = self.settings["down_orientation"]
        # VERIFY: base-frame millimeters, orientation convention, tool offset,
        # downward tool axis, and collision-free path for the actual arm.
        pose = Pose(x=x, y=y, z=z, **orientation)
        await self.arm.move_to_position(pose)

    async def get_joint_positions(self):
        # VERIFY: return value is JointPositions with .values in degrees.
        return list((await self.arm.get_joint_positions()).values)

    async def get_end_position(self):
        # VERIFY: get_end_position returns a Pose in the arm base frame.
        pose = await self.arm.get_end_position()
        return {key: getattr(pose, key) for key in
                ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}

    async def stop(self):
        await self.arm.stop()


class ViamGripper:
    def __init__(self, gripper, confirm_motion=False):
        self.gripper = gripper
        self.confirm_motion = confirm_motion

    def _require_motion(self):
        if not self.confirm_motion:
            raise PermissionError("Gripper motion requires --confirm-motion")

    async def open(self):
        self._require_motion()
        await self.gripper.open()

    async def grab(self):
        self._require_motion()
        await self.gripper.grab()


async def disconnect(machine):
    # Bound cleanup so an unreachable machine cannot hang shutdown indefinitely.
    await asyncio.wait_for(machine.close(), timeout=5)


async def build_hardware(config, poses, credentials, confirm_motion):
    from viam.components.arm import Arm
    from viam.components.camera import Camera
    from viam.components.gripper import Gripper

    machine = await connect(credentials)
    try:
        settings = config["hardware"]["viam"]
        # VERIFY: resource names match app.viam.com and from_robot is synchronous.
        depth = (Camera.from_robot(machine, settings["depth_camera"])
                 if settings["depth_camera"] else None)
        camera = ViamCamera(Camera.from_robot(machine, settings["camera"]), depth,
                            config["camera"]["roi"])
        arm = ViamArm(Arm.from_robot(machine, settings["arm"]), poses, settings,
                      confirm_motion)
        gripper = ViamGripper(Gripper.from_robot(machine, settings["gripper"]),
                              confirm_motion)
        return camera, arm, gripper, machine
    except BaseException:
        await disconnect(machine)
        raise
