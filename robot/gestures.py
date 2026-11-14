"""Replay configured, saved joint poses. No pose recording or inferred offsets.

Successful routines go home -> taught sequence -> home. Failures/interruption do
NOT trigger an automatic home/release. Saved endpoints do not prove path clearance.
"""
import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import yaml
from viam.proto.component.arm import JointPositions

ROUTINES = ('hello', 'dance')


def validate(settings, name, home):
    if not settings.get('enabled'):
        raise ValueError('Taught gestures are disabled')
    if name not in ROUTINES:
        raise ValueError('Unknown gesture')
    sequence = settings.get('routines', {}).get(name)
    if (not isinstance(sequence, list) or not 1 <= len(sequence) <= 16
            or any(not isinstance(s, str) or not s.strip() for s in sequence)):
        raise ValueError('Gesture requires a bounded list of saved switch names')
    for key in ('step_timeout_s', 'joint_tolerance_deg'):
        value = float(settings[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid gesture ' + key)
    if not math.isfinite(settings['settle_s']) or settings['settle_s'] < 0:
        raise ValueError('Invalid gesture settle_s')
    if not home.get('switch_name'):
        raise ValueError('Gesture requires a saved home switch')
    for key in ('speed_deg_s', 'acceleration_deg_s2'):
        if not math.isfinite(home[key]) or not 0 < home[key] <= 180:
            raise ValueError('Invalid gesture motion setting: ' + key)
    return [home['switch_name'], *sequence, home['switch_name']]


async def perform(arm, gripper, switches, name, settings, home, *, preflight,
                  fake=False, confirm_motion=False, checkpoint=None, log=print):
    """Snapshot saved radians, command degrees sequentially, and verify each arrival.

    checkpoint is controller-owned and runs only between completed waypoints.
    Returning True cancels remaining waypoints, including the return home.
    """
    if not fake and not confirm_motion:
        raise ValueError('Gesture motion requires --confirm-motion')
    route = validate(settings, name, home)
    timeout = float(settings['step_timeout_s'])
    tolerance = float(settings['joint_tolerance_deg'])
    targets = {}
    for resource in dict.fromkeys(route):
        saved = await asyncio.wait_for(switches[resource].do_command({'cfg': True}), timeout)
        joints = saved.get('joints')
        if (not isinstance(joints, (list, tuple)) or len(joints) != 6
                or any(type(j) not in (int, float) or not math.isfinite(j) for j in joints)):
            raise ValueError('Missing/invalid six-joint saved pose: ' + resource)
        targets[resource] = [math.degrees(j) for j in joints]
        if not all(math.isfinite(j) for j in targets[resource]):
            raise ValueError('Saved joint pose overflows degrees: ' + resource)
    moved = False
    visited = []
    try:
        for resource in route:
            if checkpoint is not None and checkpoint():
                return dict(status='interrupted', visited=visited)
            await asyncio.wait_for(preflight(), timeout)
            holding = await asyncio.wait_for(gripper.is_holding_something(), timeout)
            if holding.is_holding_something is not False:
                raise ValueError('Gesture requires verified empty gripper; no home/open/release')
            # Check again after preflight I/O, before authorizing the next waypoint.
            if checkpoint is not None and checkpoint():
                return dict(status='interrupted', visited=visited)
            target = targets[resource]
            def reached(actual):
                return (len(actual) == 6 and all(math.isfinite(j) for j in actual)
                        and all(abs(a-b) <= tolerance for a,b in zip(actual, target)))
            actual = list((await asyncio.wait_for(arm.get_joint_positions(), timeout)).values)
            if len(actual) != 6 or not all(math.isfinite(j) for j in actual):
                raise ValueError('Cannot verify current six-joint pose')
            log(f'{"FAKE " if fake else ""}{name}: saved pose {resource}')
            if not reached(actual):
                await asyncio.wait_for(arm.do_command(dict(set_speed=home['speed_deg_s'],
                                                         set_acceleration=home['acceleration_deg_s2'])), timeout)
                if checkpoint is not None and checkpoint():
                    return dict(status='interrupted', visited=visited)
                # No switch set_position(1), writes, grasp, or gripper-open commands.
                moved = True
                async def move_and_verify():
                    await arm.move_to_joint_positions(JointPositions(values=target))
                    while await arm.is_moving():
                        await asyncio.sleep(.05)
                    if not reached(list((await arm.get_joint_positions()).values)):
                        raise ValueError('Gesture did not reach saved pose: ' + resource)
                    await asyncio.sleep(settings['settle_s'])
                await asyncio.wait_for(move_and_verify(), timeout)
            visited.append(resource)
        if checkpoint is not None and checkpoint():
            return dict(status='interrupted', visited=visited)
        return dict(status='completed', visited=visited)
    except BaseException:
        if moved:
            try:
                await asyncio.wait_for(arm.stop(), 5)
            except Exception:
                log('Gesture stop failed; use machine Stop / physical E-stop.')
        raise  # Never automatically home/open after failure or cancellation.


class FakeArm:
    def __init__(self):
        self.joints = [0.] * 6
        self.moves = []
        self.stopped = False

    async def get_joint_positions(self):
        return SimpleNamespace(values=list(self.joints))

    async def move_to_joint_positions(self, positions):
        self.joints = list(positions.values)
        self.moves.append(list(self.joints))

    async def is_moving(self):return False
    async def do_command(self, command):return {}
    async def stop(self):self.stopped = True


class FakeGripper:
    async def is_holding_something(self):
        return SimpleNamespace(is_holding_something=False)


class FakeSwitch:
    def __init__(self, degrees):self.joints = [math.radians(j) for j in degrees]
    async def do_command(self, command):
        assert command == {'cfg': True}
        return {'joints': list(self.joints)}


async def run_gesture(name, config, root, *, fake=False, confirm_motion=False, checkpoint=None, log=print):
    if not fake and not confirm_motion:
        raise ValueError('Gesture motion requires --confirm-motion')
    root = Path(root)
    settings = yaml.safe_load((root/config.get('gestures_config', 'config/gestures.yaml')).read_text())
    home = config['overview']['home']
    route = validate(settings, name, home)
    machine = None
    if fake:
        arm, gripper = FakeArm(), FakeGripper()
        # Synthetic targets only; fake mode never connects to read machine poses.
        switches = {s: FakeSwitch([i*5., 0., 0., 0., 0., 0.]) for i,s in enumerate(dict.fromkeys(route))}
        async def preflight():pass
    else:
        from viam.components.arm import Arm
        from viam.components.gripper import Gripper
        from viam.components.switch import Switch
        from viam.services.generic import Generic
        from robot.viam_hw import connect, disconnect, load_credentials
        pick = yaml.safe_load((root/config['pick_bolt_config']).read_text())
        machine = await connect(load_credentials(root/'.env'))
        try:
            arm = Arm.from_robot(machine, home['arm_name'])
            gripper = Gripper.from_robot(machine, home['gripper_name'])
            teleop = Generic.from_robot(machine, pick['resources']['teleop'])
            switches = {s: Switch.from_robot(machine, s) for s in dict.fromkeys(route)}
            async def preflight():
                status = await teleop.do_command({'command': 'status'})
                if status.get('running') is not False or (await arm.is_moving()) is not False:
                    raise ValueError('Gesture requires stopped teleop and idle arm')
        except BaseException:
            await disconnect(machine)
            raise
    try:
        return await perform(arm, gripper, switches, name, settings, home, preflight=preflight,
                             fake=fake, confirm_motion=confirm_motion, checkpoint=checkpoint, log=log)
    finally:
        if machine is not None:
            await disconnect(machine)
