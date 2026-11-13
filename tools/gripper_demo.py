"""One open -> close -> open cycle. Fake by default; never commands the arm.

Real use requires an empty gripper, clear fingers/workspace, and stopped teleop.
"""
import argparse
import asyncio
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.fakes import FakeGripper
from robot.viam_hw import connect, disconnect, load_credentials


async def cycle(gripper, delay):
    print("Opening gripper...", flush=True)
    await gripper.open()
    await asyncio.sleep(delay)
    print("Closing gripper (grab)...", flush=True)
    await gripper.grab()
    await asyncio.sleep(delay)
    print("Reopening gripper...", flush=True)
    await gripper.open()
    print("Cycle complete. No arm movement commands were sent.", flush=True)


async def run(args):
    if not args.real:
        print("FAKE MODE: simulated gripper only.")
        await cycle(FakeGripper(), args.delay)
        return
    if not args.confirm_motion:
        raise ValueError("Real gripper control requires both --real and --confirm-motion")
    import yaml
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    name = config["hardware"]["viam"]["gripper"]
    credentials = load_credentials(ROOT / ".env")
    try:
        from viam.components.gripper import Gripper
    except ImportError as exc:
        raise ValueError("viam-sdk is missing. Install it in this project's .venv first.") from exc
    print(f"REAL MODE: gripper '{name}'. Keep fingers clear; stop teleop first.", flush=True)
    print("Starting in 5 seconds; Ctrl+C cancels.", flush=True)
    await asyncio.sleep(5)
    machine = await connect(credentials)
    gripper = None
    attempted_motion = False
    try:
        gripper = Gripper.from_robot(machine, name)
        attempted_motion = True
        # Grab closes the jaws; the returned holding flag is not required for an
        # empty-jaw demonstration. This is not a grasp-success test.
        await cycle(gripper, args.delay)
    except BaseException:
        if gripper is not None and attempted_motion:
            try:
                await asyncio.wait_for(gripper.stop(), timeout=3)
            except BaseException:
                print("Could not confirm gripper stop. Use the physical emergency stop.", file=sys.stderr)
        # Never automatically reopen after an error: an object could be held.
        raise
    finally:
        await disconnect(machine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Connect to the actual gripper")
    parser.add_argument("--confirm-motion", action="store_true", help="Authorize jaw movement")
    parser.add_argument("--delay", type=float, default=2, help="Seconds between actions (default 2)")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    args = parser.parse_args()
    if args.real and not args.confirm_motion:
        parser.error("Real gripper control requires both --real and --confirm-motion")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be finite and nonnegative")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        parser.exit(130, "Cancelled.\n")
    except Exception as exc:
        parser.exit(1, f"Gripper demo failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
