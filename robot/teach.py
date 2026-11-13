"""Teach named joint poses. Saving reads position; only --goto causes motion."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile

# Support the documented `python robot/teach.py` invocation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from robot.fakes import FakeArm
from robot.viam_hw import ViamArm, connect, disconnect, load_credentials

logger = logging.getLogger(__name__)


def read_poses(path: Path) -> dict:
    if not path.exists():
        return {}
    poses = json.loads(path.read_text())
    if not isinstance(poses, dict):
        raise ValueError("Poses file must contain a JSON object")
    return poses


def validate_name(name: str):
    if not name.strip() or name.startswith("_"):
        raise ValueError("Pose name must be nonempty and cannot start with '_'")


def save_pose(path: Path, name: str, joints, end_position, *, fake: bool) -> Path | None:
    """Merge with latest file, back up its exact bytes, then atomically replace.

    Single-operator tool: do not run concurrent writers against the same file.
    """
    validate_name(name)
    if not joints or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in joints):
        raise ValueError("Joint positions must be a nonempty list of finite numbers")
    poses = read_poses(path)
    poses[name] = {
        "joint_positions_deg": list(joints),
        "end_position": end_position,
        "source": "fake" if fake else "viam",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = json.dumps(poses, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        with backup.open("xb") as stream:
            stream.write(path.read_bytes())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return backup


@asynccontextmanager
async def teaching_arm(config, poses, fake, confirm_motion):
    # Connect only to the arm; teaching must not require cameras, models, or all
    # future waypoints to exist (we are creating those waypoints here).
    if fake:
        yield FakeArm(poses, config["fake"]["arm_delay_s"])
        return
    credentials = load_credentials(ROOT / ".env")
    from viam.components.arm import Arm

    machine = await connect(credentials)
    arm = None
    try:
        settings = config["hardware"]["viam"]
        # VERIFY: from_robot resource lookup and joint readback on organizer SDK.
        arm = ViamArm(Arm.from_robot(machine, settings["arm"]), poses, settings,
                      confirm_motion)
        yield arm
    except BaseException:
        if confirm_motion and arm is not None:
            try:
                await asyncio.wait_for(arm.stop(), timeout=5)
            except BaseException:
                logger.exception("Emergency stop failed; use the physical emergency stop")
        raise
    finally:
        try:
            await disconnect(machine)
        except Exception:
            logger.exception("Viam disconnect failed")


async def teach(config, path, *, name=None, goto=None, fake=False, confirm_motion=False):
    poses = read_poses(path)
    if goto is not None:
        if not confirm_motion:
            raise ValueError("--goto requires --confirm-motion (including --fake)")
        validate_name(goto)
        if goto not in poses or not isinstance(poses[goto], dict):
            raise ValueError(f"Unknown pose: {goto}")
        if not fake and poses[goto].get("source") == "fake":
            raise ValueError("Cannot move real hardware to a fake-taught pose")
    else:
        validate_name(name)
    async with teaching_arm(config, poses, fake, confirm_motion and goto is not None) as arm:
        if goto is not None:
            await arm.move_to(goto)
            print(f"{'FAKE' if fake else 'REAL'}: moved to {goto}")
        else:
            joints = await arm.get_joint_positions()
            end = await arm.get_end_position()
            backup = save_pose(path, name, joints, end, fake=fake)
            print(f"{'FAKE' if fake else 'REAL'}: saved {name} to {path}")
            if fake:
                print("Simulation readback only; not a calibrated hardware pose.")
            if backup is not None:
                print(f"Backup: {backup}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--goto")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--poses", type=Path, help="Override pose file (useful for fake practice)")
    args = parser.parse_args()
    if sum((args.name is not None, args.list, args.goto is not None)) != 1:
        parser.error("Choose exactly one of <name>, --list, or --goto <name>")
    if args.goto is not None and not args.confirm_motion:
        parser.error("--goto requires --confirm-motion (including --fake)")
    try:
        with args.config.open() as stream:
            config = yaml.safe_load(stream)
        path = args.poses if args.poses is not None else ROOT / config["poses_file"]
        if args.list:
            poses = read_poses(path)
            for name, pose in sorted(poses.items()):
                if not name.startswith("_"):
                    print(f"{name}: {json.dumps(pose)}")
            if not any(not name.startswith("_") for name in poses):
                print("No saved poses.")
            return
        asyncio.run(teach(config, path, name=args.name, goto=args.goto,
                          fake=args.fake, confirm_motion=args.confirm_motion))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; cleanup attempted.\n")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Pose teaching error: {exc}\n")


if __name__ == "__main__":
    main()
