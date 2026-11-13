"""Live geometry-derived hover and Viam collision-checked plans. No gripper actuation."""
import asyncio
import itertools
import json
import numpy as np
from robot.viam_hw import connect, disconnect, load_credentials


def calculate_hover(endpoint, lowest_z, camera_basis, target_xy, table_z, clearance):
    """Hold current orientation; move optical-axis/table intersection onto target."""
    endpoint = np.asarray(endpoint, dtype=float)
    basis = np.asarray(camera_basis, dtype=float)
    if basis.shape != (4,3) or not np.isfinite(basis).all():
        raise ValueError('Invalid camera transform')
    optical = (basis[3]-basis[0])/100
    if optical[2] > -0.95:
        raise ValueError('Wrist camera must already point approximately downward')
    hover_z = float(endpoint[2] + table_z + clearance - lowest_z)
    camera_at_hover = basis[0]+[0,0,hover_z-endpoint[2]]
    distance = (table_z-camera_at_hover[2])/optical[2]
    if distance <= 0:
        raise ValueError('Table must be in front of camera')
    seen_point = camera_at_hover+distance*optical
    endpoint_xy = endpoint[:2]+np.asarray(target_xy)-seen_point[:2]
    return [float(endpoint_xy[0]),float(endpoint_xy[1]),hover_z]


def box_corners(geometry):
    if not geometry.HasField('box'):
        raise ValueError('Only box tool geometries are supported; refusing to guess mesh extent')
    c=geometry.center
    if abs(c.theta)>1e-6 or abs(c.o_x)>1e-6 or abs(c.o_y)>1e-6 or c.o_z<0.999:
        raise ValueError('Rotated geometry requires explicit corner transformation')
    d=geometry.box.dims_mm
    return [[c.x+sx*d.x/2,c.y+sy*d.y/2,c.z+sz*d.z/2]
            for sx,sy,sz in itertools.product([-1,1],repeat=3)]


async def planned_hover(objects,cfg,folder,root,execute=False):
    from viam.components.arm import Arm
    from viam.components.gripper import Gripper
    from viam.services.motion import MotionClient
    from viam.proto.common import Pose,PoseInFrame
    s=cfg['motion']
    if not 0<s['speed_deg_s']<=10 or not 0<s['acceleration_deg_s2']<=30:
        raise ValueError('Demo speed/acceleration exceed conservative limits')
    machine=await connect(load_credentials(root/'.env'))
    arm=None; moving=False
    try:
        arm=Arm.from_robot(machine,cfg['arm_name'])
        motion=MotionClient.from_robot(machine,'builtin')
        gripper=Gripper.from_robot(machine,cfg['gripper_name'])
        before=list((await arm.get_joint_positions()).values)
        current=(await motion.get_pose(cfg['arm_name'],'world')).pose
        if current.o_z > -0.98:
            raise ValueError('Arm must already point downward; no automatic reorientation')
        ori={key:getattr(current,key) for key in ['o_x','o_y','o_z','theta']}
        endpoint=np.array([current.x,current.y,current.z])
        async def point(frame,xyz):
            out=await machine.transform_pose(PoseInFrame(reference_frame=frame,
                pose=Pose(x=xyz[0],y=xyz[1],z=xyz[2],o_z=1)), 'world')
            return [out.pose.x,out.pose.y,out.pose.z]
        vertices=[]
        for geom in await gripper.get_geometries():
            for vertex in box_corners(geom):
                vertices.append(await point(cfg['gripper_name'],vertex))
        if not vertices: raise ValueError('Missing gripper collision geometry')
        lowest=min(v[2] for v in vertices)
        basis=[await point(cfg['wrist_camera_name'],v) for v in [(0,0,0),(100,0,0),(0,100,0),(0,0,100)]]
        after=list((await arm.get_joint_positions()).values)
        if max(abs(a-b) for a,b in zip(before,after))>0.05:
            raise ValueError('Arm moved while reading geometry; stop competing controllers')
        metadata=dict(endpoint_world_mm=endpoint.tolist(),lowest_gripper_world_z_mm=lowest,
                      camera_basis_world_mm=basis,orientation=ori,assumed_table_top_z_mm=s['assumed_table_top_z_mm'],
                      clearance_mm=s['requested_lowest_gripper_clearance_mm'],joints_deg=after)
        (folder/'geometry.json').write_text(json.dumps(metadata,indent=2))
        # Plan entire route before executing. Subsequent starts use prior planned final joints.
        start_state=None;previous=endpoint;plans=[]
        for obj in objects:
            target=calculate_hover(endpoint,lowest,basis,obj['world_xy_mm'],
                                   s['assumed_table_top_z_mm'],s['requested_lowest_gripper_clearance_mm'])
            if not np.isfinite(target).all(): raise ValueError('Nonfinite target')
            if np.linalg.norm(np.array(target[:2])-endpoint[:2])>s['max_xy_displacement_mm']:
                raise ValueError('Target exceeds demo XY displacement limit')
            # First translate above the target at the higher endpoint height, then descend.
            transit=[target[0],target[1],max(float(previous[2]),target[2])]
            for stage,xyz in [('transit',transit),('hover',target)]:
                if np.linalg.norm(np.array(xyz)-previous)<0.1: continue
                extra={'motion_profile':'linear','collision_buffer_mm':s['collision_buffer_mm']}
                if start_state is not None: extra['start_state']={'configuration':start_state}
                request={'componentName':cfg['arm_name'],
                         'destination':{'referenceFrame':'world','pose':dict(x=xyz[0],y=xyz[1],z=xyz[2],
                          oX=ori['o_x'],oY=ori['o_y'],oZ=ori['o_z'],theta=ori['theta'])},'extra':extra}
                print(f"Planning candidate {obj['id']} {stage}: endpoint XYZ {np.round(xyz,1).tolist()} mm",flush=True)
                response=await motion.do_command({'plan':json.dumps(request)},timeout=60)
                if 'plan_partialwp' in response or not response.get('plan'):
                    raise ValueError('Planner returned partial or empty plan')
                trajectory=response['plan']
                # Only the intended arm may move. Do not accept plans moving leader/other resources.
                if any(set(step)!={cfg['arm_name']} for step in trajectory):
                    raise ValueError('Plan includes unexpected moving resources')
                start_state=trajectory[-1]
                plans.append(dict(id=obj['id'],stage=stage,target=xyz,trajectory=trajectory))
                previous=np.asarray(xyz)
                (folder/'plans.json').write_text(json.dumps(plans,indent=2))
        print(f'Collision-checked {len(plans)} stages. Mode: '+('EXECUTE' if execute else 'PLAN ONLY'),flush=True)
        if not execute: return plans
        if not plans: return []
        if await asyncio.to_thread(input,'Stop teleop; clear workspace; emergency stop ready. Type MOVE to execute reviewed plans: ')!='MOVE':
            print('Cancelled.');return plans
        now=list((await arm.get_joint_positions()).values)
        if max(abs(a-b) for a,b in zip(after,now))>0.05:
            raise ValueError('Arm changed since planning; refusing stale plan')
        speed_response=await arm.do_command({'set_speed':s['speed_deg_s'],'set_acceleration':s['acceleration_deg_s2']})
        (folder/'speed_response.json').write_text(json.dumps(speed_response,indent=2))
        for plan in plans:
            moving=True
            result=await motion.do_command({'execute':plan['trajectory'],'executeCheckStart':0.005},timeout=180)
            if result.get('execute') is not True: raise ValueError('Motion service did not confirm execution')
            moving=False
            print(f"Reached candidate {plan['id']} {plan['stage']}",flush=True)
        return plans
    except BaseException:
        if moving and arm is not None:
            try: await asyncio.wait_for(arm.stop(),timeout=5)
            except Exception: print('Stop failed: use emergency stop')
        raise
    finally:
        await disconnect(machine)
