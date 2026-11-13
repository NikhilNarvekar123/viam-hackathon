"""Implementation wiring belongs here, not in the orchestration loop."""
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from anomaly.stub import StubDetector
from core.types import Components
from robot.fakes import FakeArm, FakeCamera, FakeGripper
from vlm.stub import StubDecider


def build_detector(config: dict, root: Path):
    settings = config["models"]["detector"]
    implementation = settings.get("implementation", "stub")
    if implementation == "stub":
        return StubDetector(settings["threshold"], **config["stub"]["detector"])
    if implementation == "openvino":
        from anomaly.detector import OpenVINODetector
        return OpenVINODetector(settings, root)
    raise ValueError(f"Unknown detector implementation: {implementation}")


def build_decider(config: dict, root: Path):
    from vlm.prompt import build_prompt
    from vlm.rules import RulesDecider

    settings = config["models"]["decider"]
    allowed, defects = config["allowed_actions"], config["defect_types"]
    implementation = settings.get("implementation", "stub")
    if implementation == "stub":
        return StubDecider(allowed, defects, config["stub"]["decider"])
    if implementation == "rules":
        return RulesDecider(allowed, settings["rules_defect_type"])
    if implementation != "openvino":
        raise ValueError(f"Unknown decider implementation: {implementation}")
    # Invalid policy/prompt config must be fixed, not hidden by a model fallback.
    build_prompt(settings["prompt"], defects, allowed)
    fallback = RulesDecider(allowed, settings["rules_defect_type"])
    try:
        from vlm.decider import OpenVINODecider
        return OpenVINODecider(settings, defects, allowed, root)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "VLM load/warm-up failed (%s: %s); using rules decider", type(exc).__name__, exc)
        return fallback


def build_components(config: dict, root: Path, fake: bool) -> Components:
    if not fake:
        raise ValueError("Use managed_components for real hardware")
    with (root / config["poses_file"]).open() as stream:
        poses = json.load(stream)
    sorting = config["sorting"]
    allowed = config["allowed_actions"]
    routes = config["action_to_bin"]
    if "reject" not in allowed or routes.get("reject") != sorting["reject_bin"]:
        raise ValueError("Safe reject must be allowed and route to the reject bin")
    for action in allowed:
        if action not in routes or routes[action] not in sorting["above_bin_poses"]:
            raise ValueError(f"Missing action/bin mapping: {action}")
    if sorting["pick_mode"] not in ("fixed", "vision"):
        raise ValueError("pick_mode must be fixed or vision")
    required = [sorting["home_pose"], *sorting["above_bin_poses"].values()]
    if sorting["pick_mode"] == "fixed":
        required.extend([sorting["pick_pose"], sorting["above_pick_pose"]])
    if any(name not in poses for name in required):
        raise ValueError("Missing taught/placeholder pose")
    if sorting["pass_bin"] not in sorting["above_bin_poses"]:
        raise ValueError("Missing pass-bin pose")
    delays = config["fake"]
    if any(delay < 0 for delay in delays.values()):
        raise ValueError("Fake delays must be nonnegative")
    return Components(
        FakeCamera(root / config["camera"]["samples"], delays["camera_delay_s"],
                   config["camera"]["roi"]),
        build_detector(config, root),
        build_decider(config, root),
        FakeArm(poses, delays["arm_delay_s"]),
        FakeGripper(delays["gripper_delay_s"]),
    )


@asynccontextmanager
async def managed_components(config, root, fake, confirm_motion=False):
    # --fake always wins. No --fake explicitly selects real mode, regardless
    # of the default backend used by other tools.
    from robot.viam_hw import build_hardware, disconnect, load_credentials

    credentials = None if fake else load_credentials(root / ".env")
    components = build_components(config, root, fake=True)
    machine = None
    try:
        if not fake:
            backend = config["hardware"]["real_backend"]
            if backend != "viam":
                raise ValueError(f"Unsupported real hardware backend: {backend}")
            with (root / config["poses_file"]).open() as stream:
                poses = json.load(stream)
            camera, arm, gripper, machine = await build_hardware(
                config, poses, credentials, confirm_motion)
            components.camera, components.arm, components.gripper = camera, arm, gripper
            components.real_hardware = True
        yield components
    except BaseException:
        if machine is not None and confirm_motion:
            try:
                await asyncio.wait_for(components.arm.stop(), timeout=5)
            except BaseException:
                logging.exception("Emergency arm stop failed; use physical emergency stop")
        raise
    finally:
        if machine is not None:
            try:
                await disconnect(machine)
            except Exception:
                logging.exception("Viam disconnect failed")
