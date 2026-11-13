"""Fake-only hardware. Generated images stay in memory, never overwrite data."""
import asyncio
import logging
from pathlib import Path

from PIL import Image, ImageDraw

from core.types import Frame

logger = logging.getLogger(__name__)


class FakeCamera:
    def __init__(self, samples: Path, delay: float = 0, roi=None):
        self.delay = delay
        self.roi = roi
        self.index = 0
        groups = {}
        for label in ("good", "defect"):
            paths = sorted(p for p in (samples / label).glob("*")
                           if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
            if paths:
                groups[label] = [(path, label) for path in paths]
            else:
                groups[label] = [(self._synthetic(label, i), label) for i in range(3)]
        # Interleave classes so short demos exercise both, without dropping images.
        self.samples = []
        for i in range(max(map(len, groups.values()))):
            for group in groups.values():
                if i < len(group):
                    self.samples.append(group[i])

    @staticmethod
    def _synthetic(label, variant):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((24 + variant, 24, 104, 104), fill="gray")
        if label == "defect":
            if variant % 2:
                draw.ellipse((58, 58, 72, 72), fill="black")
            else:
                draw.line((40, 40, 88, 88), fill="black", width=4)
        return image

    async def get_frame(self) -> Frame:
        await asyncio.sleep(self.delay)
        source, label = self.samples[self.index % len(self.samples)]
        if isinstance(source, Path):
            with Image.open(source) as image:
                image = image.convert("RGB")
        else:
            image = source.copy()
        if self.roi is not None:
            left, top, right, bottom = self.roi
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("ROI must fit inside the camera image")
            image = image.crop(self.roi)
        self.index += 1
        return Frame(f"item-{self.index:06d}", image, ground_truth=label)


class FakeArm:
    def __init__(self, poses: dict, delay: float = 0):
        self.poses = poses
        self.delay = delay
        self.calls = []
        # Synthetic readback only; no forward/inverse kinematics are simulated.
        self.joint_positions = [0.0] * 6
        self.end_position = None

    async def move_to(self, pose_name: str) -> None:
        pose = self.poses[pose_name]
        self.joint_positions = list(pose["joint_positions_deg"])
        self.end_position = pose.get("end_position")
        self.calls.append(pose_name)
        logger.info("FakeArm.move_to(%s): %s", pose_name, pose)
        await asyncio.sleep(self.delay)


    async def move_to_xyz(self, x: float, y: float, z: float) -> None:
        self.joint_positions = None  # Cannot infer joint angles without IK.
        self.end_position = {"x": x, "y": y, "z": z}
        self.calls.append((x, y, z))
        logger.info("FakeArm.move_to_xyz(%s, %s, %s)", x, y, z)
        await asyncio.sleep(self.delay)

    async def get_joint_positions(self) -> list[float]:
        self.calls.append("get_joint_positions")
        logger.info("FakeArm.get_joint_positions()")
        await asyncio.sleep(self.delay)
        if self.joint_positions is None:
            raise ValueError("Fake joint readback unavailable after XYZ motion (no IK)")
        return list(self.joint_positions)

    async def get_end_position(self) -> dict | None:
        self.calls.append("get_end_position")
        logger.info("FakeArm.get_end_position()")
        await asyncio.sleep(self.delay)
        return dict(self.end_position) if self.end_position is not None else None

    async def stop(self) -> None:
        self.calls.append("stop")
        logger.info("FakeArm.stop()")
        await asyncio.sleep(self.delay)


class FakeGripper:
    def __init__(self, delay: float = 0):
        self.delay = delay
        self.calls = []

    async def open(self) -> None:
        self.calls.append("open")
        logger.info("FakeGripper.open()")
        await asyncio.sleep(self.delay)

    async def grab(self) -> None:
        self.calls.append("grab")
        logger.info("FakeGripper.grab()")
        await asyncio.sleep(self.delay)
