"""Shared contracts; implementations must not leak into the pipeline loop."""
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from PIL import Image


@dataclass
class Frame:
    item_id: str
    image: Image.Image
    depth: np.ndarray | None = None
    ground_truth: Literal["good", "defect"] | None = None


@dataclass
class Detection:
    score: float
    is_anomalous: bool
    heatmap: np.ndarray | None = None


@dataclass
class Decision:
    defect_type: str | None
    action: str
    raw_output: str
    valid: bool


class Camera(Protocol):
    async def get_frame(self) -> Frame: ...


class Detector(Protocol):
    def detect(self, frame: Frame) -> Detection: ...


class Decider(Protocol):
    def decide(self, frame: Frame, detection: Detection) -> Decision: ...


class Arm(Protocol):
    async def move_to(self, pose_name: str) -> None: ...
    async def move_to_xyz(self, x: float, y: float, z: float) -> None: ...
    async def get_joint_positions(self) -> list[float]: ...
    async def get_end_position(self) -> dict | None: ...
    async def stop(self) -> None: ...


class Gripper(Protocol):
    async def open(self) -> None: ...
    async def grab(self) -> None: ...


@dataclass
class Components:
    camera: Camera
    detector: Detector
    decider: Decider
    arm: Arm
    gripper: Gripper
    real_hardware: bool = False
