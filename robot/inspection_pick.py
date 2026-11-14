"""Opt-in bridge: classify the picker's fresh target, then accept/place or reject/return.

This adapter is not physically validated. No inspection-frame IDs are reused across
robot moves. Teammate grasp geometry/planning remains in pick_bolt.py.
"""
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import yaml
import pick_bolt as picker
from anomaly.object_crops import automatic_crops, extract_candidate_crops
from anomaly.wrist import WristInspector
from robot.home import replay_home, open_for_capture
from robot.viam_hw import load_credentials
from tools.inspect_wrist import inspect, annotate_scores
from vlm.service import decide_wrist_objects


def match_target(target, objects, width, settings):
    """Require one RGB candidate consistent with the current depth-selected target."""
    if target.xs is None or target.ys is None or not len(target.xs):
        raise ValueError('Target has no pixel footprint')
    a=[int(min(target.xs)),int(min(target.ys)),int(max(target.xs))+1,int(max(target.ys))+1]
    matches=[]
    for obj in objects:
        b=obj['bbox_xyxy']
        intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        area=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
        distance=math.dist([target.u,target.v],obj['center_uv'])
        if area>0 and intersection/area>=settings['match_min_overlap'] and distance<=settings['match_center_distance_px']*width/640:
            matches.append(obj)
    if len(matches)!=1:
        raise ValueError(f'Expected one RGB/depth target match, found {len(matches)}; no grasp')
    return matches[0]


class InspectionBridge:
    def __init__(self, root, config, settings, fake=False, inspector=None):
        self.root=Path(root);self.settings=settings;self.fake=fake
        self.crop_settings=yaml.safe_load((self.root/config['wrist_inspection_config']).read_text())
        self.vlm_settings=yaml.safe_load((self.root/config['vlm_service_config']).read_text())
        self.inspector=inspector if inspector is not None else WristInspector(self.crop_settings,self.root,fake=fake)

    async def __call__(self, station, item, target, info, objects):
        folder=self.root/'logs/inspection_pick'/datetime.now(timezone.utc).strftime('target_%Y%m%dT%H%M%S_%fZ')
        folder.mkdir(parents=True)
        item['inspection_output']=str(folder)
        try:
            if info.get('bgr') is None: raise ValueError('Missing target color frame')
            if time.monotonic()-info['capture_monotonic']>self.settings['max_frame_age_s']:
                raise ValueError('Target frame already stale')
            if info.get('in_frame',0)<.6: raise ValueError('Point-cloud/color alignment check failed')
            image=Image.fromarray(np.asarray(info['bgr'])[:,:,::-1].copy()).convert('RGB')
            if image.size!=(station.K.width,station.K.height):
                raise ValueError('Color image dimensions do not match depth intrinsics')
            image.save(folder/'frame.png')
            _,result,mask=automatic_crops(image,self.crop_settings['auto_crop'])
            selected=match_target(target,result['objects'],image.width,self.settings)
            index=result['objects'].index(selected)
            crops=extract_candidate_crops(image,result,mask,object_indices=[index])
            records=inspect(image,[selected['crop_bbox_xyxy']],self.inspector.detector,self.inspector.threshold,
                            folder,crops=crops,objects=[selected])
            annotate_scores(image,records).save(folder/'scored_objects.png')
            binding=dict(perception_target_id=target.idx,rgb_candidate_id=selected['id'],
                         source_frame=str(folder/'frame.png'),bbox_xyxy=selected['bbox_xyxy'],
                         target_xy_mm=np.asarray(item['xy']).tolist(),crop_settings=self.crop_settings,
                         captured_monotonic=info['capture_monotonic'])
            (folder/'target_binding.json').write_text(json.dumps(binding,indent=2,allow_nan=False))
            decisions=await decide_wrist_objects(dict(output_dir=str(folder),objects=records),
                                                self.vlm_settings,self.root,fake=self.fake)
            decision=decisions[0]['final_action'];score=records[0]['score']
            if decision=='hold': return 'hold',score,0.0
            if time.monotonic()-info['capture_monotonic']>self.settings['max_frame_age_s']:
                raise ValueError('Inspection exceeded target frame freshness limit; no grasp')
            if not self.fake:
                await station.require_idle_controller()
            item['inspection_action']=decision
            # Legacy slot is not a calibrated model probability.
            return {'accept':'box','reject':'reject'}[decision],score,0.0
        except Exception as exc:
            (folder/'inspection_error.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}',
                'action':'hold','grasp_authorized':False},indent=2))
            print(f'Inspection HOLD: {type(exc).__name__}: {exc}',flush=True)
            return 'hold',None,0.0

class IntegratedStation(picker.Station):
    def __init__(self,args,config):
        super().__init__(args);self.home_settings=config['overview']['home']

    async def require_idle_controller(self):
        from viam.services.generic import Generic
        # Unknown teleop state is NOT interpreted as safe.
        status=await Generic.from_robot(self.robot,picker.TELEOP).do_command({'command':'status'},timeout=10)
        if not isinstance(status.get('running'),bool) or status['running']:
            raise ValueError('Teleop is running or its state is unknown; no pick authorized')
        if await self.arm.is_moving(): raise ValueError('Arm is moving; no pick authorized')

    async def go_home(self,gate=True):
        from viam.components.switch import Switch
        self.moved=True
        await replay_home(self.arm,Switch.from_robot(self.robot,self.home_settings['switch_name']),self.home_settings)
        await open_for_capture(self.gripper,self.home_settings)
        await self.arm.do_command({'set_speed':self.home_settings['speed_deg_s'],
                                   'set_acceleration':self.home_settings['acceleration_deg_s2']})
        self.jaws_open=True;self.theta=picker.NATURAL_THETA
        self.gpos=await self.transform(np.zeros(3),picker.GRIPPER,'world')
        camera=await self.transform(np.zeros(3),picker.CAM,'world')
        self.scan_g=self.gripper_xyz_for_camera([camera[0],camera[1],picker.TABLE_Z_EXPECTED+picker.SCAN_HEIGHT_MM])


async def verify_main_pick_start(config,root):
    """Read-only preflight BEFORE main's home/open, so an existing held item is not dropped."""
    from viam.components.arm import Arm
    from viam.components.gripper import Gripper
    from viam.services.generic import Generic
    from robot.viam_hw import connect,disconnect
    home=config['overview']['home']
    settings=yaml.safe_load((root/config['pick_bolt_config']).read_text())
    robot=await connect(load_credentials(root/'.env'))
    try:
        status=await Generic.from_robot(robot,settings['resources']['teleop']).do_command({'command':'status'},timeout=10)
        if not isinstance(status.get('running'),bool) or status['running']:
            raise ValueError('Stop teleop before starting the main pick pipeline')
        if await Arm.from_robot(robot,home['arm_name']).is_moving():
            raise ValueError('Arm is moving; refusing main home/open')
        if (await Gripper.from_robot(robot,home['gripper_name']).is_holding_something()).is_holding_something:
            raise ValueError('Gripper holds an item; manually resolve before main home/open')
    finally:
        await disconnect(robot)


def setup_good_destination(st,settings):
    """GOOD and (when set) REJECT place areas: every place reachable at PLACE_CARRY_MIN_Z, areas kept apart."""
    for key in ('good_xy','reject_xy'):
        spot=settings[key]
        if spot is not None and not all(picker.reachable([x,y,picker.PLACE_CARRY_MIN_Z]) for x,y in picker.place_grid(spot)):
            raise ValueError(f'{key} place area is not reachable at PLACE_CARRY_MIN_Z')
    good=picker.place_area(settings['good_xy'])
    if settings['reject_xy'] is not None and picker.dist_to_rect(settings['reject_xy'],good)<picker.SPOT_KEEPOUT_MM:
        raise ValueError('reject_xy is too close to the GOOD place area')
    st.box=dict(drop=np.array(settings['good_xy'],float),rim=picker.TABLE_Z_EXPECTED,
                drop_z=float(st.scan_g[2]),bbox=good)


async def execute_result(st,result,skip):
    if result['status']=='held':
        return await picker.deliver(st,result,skip)
    if result['status']=='missed':
        await st.gripper.open();st.jaws_open=True
    if result.get('xy') is not None and result['status'] not in ('none','inspection_hold'):
        skip.append(np.asarray(result['xy']))
    await st.go_scan()
    return f'not delivered: {result["status"]}'


class FakeStation:
    """Fake delivery interface; never creates Viam clients. Its camera sits at the gripper and sees a 20 mm block
    wherever something was let go (self.lying: share it to keep the fake table across stations)."""
    def __init__(self,settings):
        self.args=SimpleNamespace(good_xy=settings['good_xy'],reject_xy=settings['reject_xy'],hold_xy=None)
        self.holding=True  # set False to simulate a part falling out of the jaws on the way
        self.gpos=np.array([400.,0.,350.]);self.scan_g=self.gpos.copy()
        self.K=picker.Intrinsics(600.,600.,320.,240.,640,480);self.lying=[]
        self.placed={'box':0,'reject':0,'hold':0};self.gripper=self;self.actions=[]
        self.released_at=None;self.jaws_open=False
    async def move_gripper(self,xyz,theta,what,**kwargs):
        self.gpos=np.array(xyz,float);self.actions.append(dict(move=self.gpos.tolist(),what=what))
    async def open(self):
        self.actions.append(dict(open_at=self.gpos.tolist()));self.lying.append(self.gpos[:2].tolist())
    async def confirm(self,*args,**kwargs): pass
    async def go_scan(self,*args,**kwargs): self.gpos=self.scan_g.copy()
    async def is_holding_something(self): return SimpleNamespace(is_holding_something=self.holding,meta=None)
    def gripper_xyz_for_camera(self,cam_xyz): return np.asarray(cam_xyz,float)
    async def look(self,tag,closeups=True):
        depth=np.full((480,640),440.)
        for x,y in self.lying:
            v,u=int(240-(x-self.gpos[0])*600/440),int(320-(y-self.gpos[1])*600/440)
            depth[max(v-14,0):max(v+14,0),max(u-14,0):max(u+14,0)]=420.
        info=picker.detect_depth(depth,self.K)[1]
        info['R'],info['t']=np.array([[0.,-1,0],[-1,0,0],[0,0,-1]]),np.array([self.gpos[0],self.gpos[1],440.])
        return [],info


async def run_pick_loop(config,root,fake=False,confirm_motion=False,cycles=None,pause=None,bridge=None,state=None):
    if not fake and not confirm_motion: raise ValueError('Picking requires explicit --confirm-motion')
    settings=yaml.safe_load((root/config['pick_bolt_config']).read_text())
    if settings['hold_xy'] is not None:
        raise ValueError('This integration leaves HOLD ungrasped; hold_xy must be null')
    if settings['resources']!={'arm':picker.ARM,'gripper':picker.GRIPPER,'camera':picker.CAM,
                               'motion':picker.MOTION,'teleop':picker.TELEOP}:
        raise ValueError('Configured resources must match the imported controller resource names')
    if (len(settings['good_xy'])!=2 or not all(math.isfinite(float(x)) for x in settings['good_xy']) or
            type(settings['max_good_places']) is not int or not 1<=settings['max_good_places']<=12 or
            not 0<settings['match_min_overlap']<=1 or
            any(not math.isfinite(float(settings[k])) or settings[k]<=0 for k in
                ['max_frame_age_s','match_center_distance_px'])):
        raise ValueError('Invalid pick settings')
    if cycles is not None and cycles<1: raise ValueError('Pick cycle limit must be positive')
    if not fake and cycles!=1 and pause is None:
        raise ValueError('Real multi-cycle picking requires an operator pause callback')
    bridge=bridge if bridge is not None else InspectionBridge(root,config,settings,fake)
    state=state if state is not None else {}
    skip=state.setdefault('skip',[]);index=0
    report={'status':'capacity','sorting_executed':False}
    folder=root/'logs/inspection_pick'/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True)
    st=None
    try:
        if fake:
            st=FakeStation(settings)
        else:
            credentials=load_credentials(root/'.env')
            args=picker.parse_args([])
            args.address=credentials['VIAM_ADDRESS'];args.mode='pick';args.yes=True;args.home=True
            args.force=False;args.index=None;args.close_up=settings['close_up'];args.frames=settings['frames']
            args.good_xy=settings['good_xy'];args.reject_xy=settings['reject_xy'];args.hold_xy=None;args.anomaly_service=None
            args.record_hz=0;args.debug_dir=str(folder);args.save_raw=False
            # Register Switch before the SDK constructs clients at connect time.
            import viam.components.switch
            st=IntegratedStation(args,config)
            await st.connect();st.inspection_handler=bridge
        st.placed=state.setdefault('placed',{'box':0,'reject':0,'hold':0})
        if fake:st.lying=state.setdefault('fake_table',[])
        while cycles is None or index<cycles:
            if st.placed['box']>=settings['max_good_places']:
                print('GOOD place area capacity reached; stopping.');break
            index+=1
            print(f'\n=== Inspect + pick cycle {index} ({"FAKE" if fake else "REAL MOTION"}) ===',flush=True)
            if fake:
                image=Image.new('RGB',(640,480),'gray');ImageDraw.Draw(image).rectangle((250,210,330,240),fill='black')
                yy,xx=np.mgrid[210:241,250:331]
                target=SimpleNamespace(idx=1,u=290,v=225,xs=xx.ravel(),ys=yy.ravel())
                info=dict(bgr=np.asarray(image)[:,:,::-1].copy(),capture_monotonic=time.monotonic(),in_frame=1)
                result=dict(xy=np.array([400.,0.]),grasp_z=69.,score=1.)
                action,score,_=await bridge(st,result,target,info,[target])
                result.update(status='inspection_hold' if action=='hold' else 'held',decision=action,anomaly=score)
                if action!='hold':st.actions.append({'simulated_pick':result['xy'].tolist()})
            else:
                await st.require_idle_controller()
                if (await st.gripper.is_holding_something()).is_holding_something:
                    raise ValueError('Gripper already holds an item; manually resolve before running')
                await st.go_home()
                setup_good_destination(st,settings)
                await st.go_scan()
                result=await picker.pick_one(st,'pick',skip,force_close_up=True)
            delivery=await execute_result(st,result,skip)
            report=dict(cycle=index,fake=fake,status=result['status'],decision=result.get('decision'),
                        delivery=delivery,inspection_output=result.get('inspection_output'),
                        object_xy_mm=np.asarray(result['xy']).tolist() if result.get('xy') is not None else None,
                        sorting_executed=not fake and result['status']=='held',
                        simulated_actions=st.actions if fake else None)
            (folder/f'cycle_{index:03d}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print('Delivery:',delivery,flush=True)
            if cycles is None or index<cycles:
                if pause is not None and not await pause():break
                if fake and pause is None:await asyncio.sleep(.1)
    except BaseException as exc:
        (folder/'error.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}',
            'cycle':index,'fake':fake,'completed':False},indent=2))
        if st is not None and not fake and getattr(st,'moved',False):
            try:await asyncio.wait_for(st.robot.stop_all(),5)
            except Exception:print('STOP failed: use physical emergency stop.',flush=True)
        raise
    finally:
        if st is not None and not fake and hasattr(st,'robot'):await st.close()
    print('Pick run report:',folder)
    return dict(report,output_dir=str(folder))
