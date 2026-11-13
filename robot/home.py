"""Replay organizer home switch (position 2), never record/overwrite its pose."""
import asyncio
import math
from robot.viam_hw import connect, disconnect, load_credentials


async def replay_home(arm, switch, settings):
    speed=float(settings['speed_deg_s'])
    acceleration=float(settings['acceleration_deg_s2'])
    if not 0<speed<=10 or not 0<acceleration<=30:
        raise ValueError('Invalid conservative homing speed/acceleration')
    saved=await switch.do_command({'cfg':True})
    joints=saved.get('joints')
    if not joints or not all(math.isfinite(float(j)) for j in joints):
        raise ValueError('Organizer home must contain finite saved joint positions')
    target=[math.degrees(float(j)) for j in joints]
    def at_home(actual):
        return (len(actual)==len(target) and all(math.isfinite(a) and
                abs(a-b)<=settings['joint_tolerance_deg'] for a,b in zip(actual,target)))
    if at_home(list((await arm.get_joint_positions()).values)) and not await arm.is_moving():
        print('Already at organizer home; skipping redundant movement')
        await asyncio.sleep(settings['settle_s'])
        return
    moving=False
    try:
        await arm.do_command({'set_speed':speed,'set_acceleration':acceleration})
        moving=True
        async def move_and_wait():
            await switch.set_position(2)  # 1 SAVES a new pose: never use it here.
            while await arm.is_moving():
                await asyncio.sleep(0.2)
            actual=list((await arm.get_joint_positions()).values)
            if not at_home(actual):
                raise ValueError('Arm did not reach organizer home; overview capture cancelled')
            await asyncio.sleep(settings['settle_s'])
        await asyncio.wait_for(move_and_wait(),timeout=settings['timeout_s'])
        moving=False
    finally:
        if moving:
            try:
                await asyncio.wait_for(arm.stop(),timeout=5)
            except Exception:
                print('Home stop failed; use physical emergency stop if necessary')


async def open_for_capture(gripper, settings):
    """Open fully and verify UFactory's reported 0..850 jaw position."""
    async def open_and_verify():
        await gripper.open()
        while await gripper.is_moving():
            await asyncio.sleep(0.2)
        result = await gripper.do_command({'get': True})
        position = float(result['pos'])
        if not math.isfinite(position) or abs(position - settings['gripper_open_position']) > settings['gripper_open_tolerance']:
            raise ValueError(f'Gripper did not reach full-open position (reported {position}); capture cancelled')
        await asyncio.sleep(settings['settle_s'])
    try:
        await asyncio.wait_for(open_and_verify(), timeout=settings['timeout_s'])
    except BaseException:
        try:
            await asyncio.wait_for(gripper.stop(), timeout=5)
        except Exception:
            print('Gripper stop failed; use physical emergency stop if necessary')
        raise


async def home_before_capture(config, root, fake, confirm_motion):
    settings=config['overview']['home']
    if fake:
        print(f"FAKE: simulate organizer {settings['switch_name']} -> fully open gripper -> capture")
        return 'simulated'
    if not confirm_motion:
        raise ValueError('Live loop requires --confirm-motion: home verification cannot be skipped')
    from viam.components.arm import Arm
    from viam.components.switch import Switch
    from viam.components.gripper import Gripper
    machine=await connect(load_credentials(root/'.env'))
    try:
        print(f"Moving to organizer {settings['switch_name']} at {settings['speed_deg_s']} deg/s")
        await replay_home(Arm.from_robot(machine,settings['arm_name']),
                          Switch.from_robot(machine,settings['switch_name']),settings)
        await open_for_capture(Gripper.from_robot(machine,settings['gripper_name']),settings)
        print('Home reached, arm stopped, gripper fully open. Capturing overview next.')
        return 'reached'
    finally:
        await disconnect(machine)
