"""Hello / dance on the arm.

python tools/gesture.py hello --fake               # log the moves, no machine
python tools/gesture.py dance --confirm-motion     # real arm: preflight, home, dance, back home
"""
import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.gestures import ROUTINES, perform  # noqa: E402


class FakeArm:
    def __init__(self):
        self.joints = [2.8, -26.8, -55.7, -2.8, 86.7, 1.4]

    async def get_joint_positions(self):
        return SimpleNamespace(values=list(self.joints))

    async def move_to_joint_positions(self, positions):
        self.joints = list(positions.values)


class FakeGripper:
    async def open(self):
        pass

    async def grab(self):
        return False


async def run(name, fake, config):
    if fake:
        await perform(FakeArm(), FakeGripper(), name)
        return
    from viam.components.arm import Arm
    from viam.components.gripper import Gripper
    from viam.components.switch import Switch
    from robot.home import open_for_capture, replay_home
    from robot.inspection_pick import verify_main_pick_start
    from robot.viam_hw import connect, disconnect, load_credentials
    home = config["overview"]["home"]
    await verify_main_pick_start(config, ROOT)       # teleop stopped, arm still, gripper empty
    machine = await connect(load_credentials(ROOT / ".env"))
    try:
        arm, gripper = Arm.from_robot(machine, home["arm_name"]), Gripper.from_robot(machine, home["gripper_name"])
        await replay_home(arm, Switch.from_robot(machine, home["switch_name"]), home)
        await open_for_capture(gripper, home)
        await arm.do_command({"set_speed": home["speed_deg_s"], "set_acceleration": home["acceleration_deg_s2"]})
        try:
            await perform(arm, gripper, name)
        except BaseException:
            try:
                await asyncio.wait_for(arm.stop(), 5)
            except Exception:
                print("Stop failed: use machine Stop / physical E-stop.")
            raise
    finally:
        await disconnect(machine)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gesture", choices=sorted(ROUTINES))
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true", help="required to move the real arm")
    args = parser.parse_args()
    if not args.fake and not args.confirm_motion:
        parser.error("real arm motion needs --confirm-motion (or use --fake)")
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    try:
        asyncio.run(run(args.gesture, args.fake, config))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Gesture error: {exc}\n")


if __name__ == "__main__":
    main()
