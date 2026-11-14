"""Replay taught hello/dance poses. Real motion requires explicit authorization.

python tools/gesture.py hello --fake
python tools/gesture.py dance --fake
"""
import argparse
import asyncio
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.gestures import ROUTINES, FakeArm, FakeGripper, run_gesture  # noqa: E402


async def run(name, fake, config, confirm_motion=False):
    return await run_gesture(name, config, ROOT, fake=fake, confirm_motion=confirm_motion)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('gesture', choices=ROUTINES)
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--confirm-motion', action='store_true')
    args = parser.parse_args()
    if not args.fake and not args.confirm_motion:
        parser.error('real arm motion needs --confirm-motion (or use --fake)')
    config = yaml.safe_load((ROOT/'config/config.yaml').read_text())
    try:
        asyncio.run(run(args.gesture, args.fake, config, args.confirm_motion))
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted.\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Gesture error: {exc}\n')


if __name__ == '__main__':
    main()
