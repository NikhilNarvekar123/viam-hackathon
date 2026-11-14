"""Pass original pipeline boxes/verdicts to the existing picker, once per batch.

A shared stationary RGB/depth capture binds pixel boxes before any camera move.
The picker retains detection for localization/refinement and all grasp/clearance
rules, but cannot substitute a different target or rerun classification.
"""
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import yaml
import pick_bolt as picker
from robot.crop_target_mapping import bind_crops
from robot.roi_pick_target import bind_rois,target_in_view
from robot.inspection_pick import IntegratedStation,FakeStation,setup_good_destination,execute_result
from robot.viam_hw import load_credentials


async def run_external_box_batch(config,root,inspector,fake,confirm_motion,state,checkpoint=None):
    from vlm.service import decide_wrist_objects
    async with ExternalBoxBatch(root,config,fake,confirm_motion,state) as batch:
        wrist=await batch.capture_and_inspect(inspector)
        settings=yaml.safe_load((Path(root)/config['vlm_service_config']).read_text())
        decisions=await decide_wrist_objects(wrist,settings,root,fake=fake)
        report=await batch.execute(decisions,checkpoint=checkpoint)
        return wrist,decisions,report


class ExternalBoxBatch:
    def __init__(self,root,config,fake=False,confirm_motion=False,state=None):
        self.root=Path(root);self.config=config;self.fake=fake
        if not fake and not confirm_motion:raise ValueError('External-box picking requires --confirm-motion')
        self.settings=yaml.safe_load((self.root/config['pick_bolt_config']).read_text())
        self.mapping=yaml.safe_load((self.root/self.settings['mapping_config']).read_text())
        self.state=state if state is not None else {}
        self.st=None;self.wrist=None;self.rows=[]
        self.home_ready=False;self.reference_joints=None;self.fake_scene=None
        self.last_home_frame_time=None
        for key in ['target_match_radius_mm','max_batch_age_s']:
            if not math.isfinite(float(self.settings[key])) or self.settings[key]<=0:
                raise ValueError(f'Invalid {key}')
        if self.settings['hold_xy'] is not None:
            raise ValueError('External batch leaves HOLD ungrasped; hold_xy must be null')
        expected=dict(arm=picker.ARM,gripper=picker.GRIPPER,camera=picker.CAM,motion=picker.MOTION,teleop=picker.TELEOP)
        if self.settings['resources']!=expected:raise ValueError('Controller and pipeline resource names differ')
        if not 1<=self.settings['max_good_places']<=12:raise ValueError('Invalid GOOD area capacity')

    async def __aenter__(self):
        if self.fake:self.st=FakeStation(self.settings)
        else:
            credentials=load_credentials(self.root/'.env')
            args=picker.parse_args([])
            args.address=credentials['VIAM_ADDRESS'];args.mode='pick';args.yes=True;args.force=False
            args.index=None;args.anomaly_service=None;args.record_hz=0
            args.frames=self.settings['frames'];args.close_up=self.settings['close_up']
            args.good_xy=self.settings['good_xy'];args.reject_xy=self.settings['reject_xy'];args.hold_xy=None
            args.debug_dir=str(self.root/'logs/inspection_pick');Path(args.debug_dir).mkdir(parents=True,exist_ok=True)
            import viam.components.switch
            self.st=IntegratedStation(args,self.config)
            try:
                await self.st.connect()
                await self.st.require_idle_controller()
            except BaseException:
                if hasattr(self.st,'robot'):await self.st.close()
                raise
        self.st.placed=self.state.setdefault('placed',{'box':0,'reject':0,'hold':0})
        if self.fake:self.st.lying=self.state.setdefault('fake_table',[])
        return self

    async def __aexit__(self,kind,exc,tb):
        if not self.fake:
            try:
                if exc is not None and self.st.moved:
                    try:await asyncio.wait_for(self.st.robot.stop_all(),5)
                    except Exception:print('Stop failed: use machine Stop / physical E-stop.')
            finally:await self.st.close()

    async def home_for_check(self):
        """Continuous-mode safe boundary: never home/open if still holding."""
        if self.fake:
            self.st.actions.append({'home_for_validation':True})
        else:
            await self.st.require_idle_controller()
            holding=await self.st.gripper.is_holding_something(timeout=10)
            if holding.is_holding_something:
                raise ValueError('Gripper still holds an item; refusing home/open')
            await self.st.go_home()
        self.home_ready=True

    async def capture_home_frames(self,settings):
        """Two RGB-only samples at the original verified home pose, no inference."""
        if not self.home_ready:raise ValueError('Home scene capture requires verified home')
        if self.fake:
            if self.fake_scene is None:raise ValueError('Missing fake reference scene')
            return self.fake_scene.copy(),self.fake_scene.copy()
        from robot.viam_hw import decode_color,select_image
        async def capture_one():
            await self.st.require_idle_controller()
            before=list((await self.st.arm.get_joint_positions()).values)
            images,metadata=await self.st.cam.get_images()
            timestamp=getattr(metadata,'captured_at',None)
            if timestamp is None:raise ValueError('Home-view capture has no freshness metadata')
            stamp=float(timestamp.seconds)+float(timestamp.nanos)*1e-9
            age=time.time()-stamp
            if (not math.isfinite(stamp) or stamp<=0 or age<0 or age>settings['max_frame_age_s']
                    or self.last_home_frame_time is not None and stamp<=self.last_home_frame_time):
                raise ValueError('Home-view frame is stale, repeated, or timestamp is invalid')
            self.last_home_frame_time=stamp
            after=list((await self.st.arm.get_joint_positions()).values)
            await self.st.require_idle_controller()
            reference=self.reference_joints
            if (not reference or len(before)!=len(reference) or len(after)!=len(reference)
                    or not np.isfinite(reference+before+after).all()
                    or max(abs(a-b) for a,b in zip(before,reference))>settings['home_joint_tolerance_deg']
                    or max(abs(a-b) for a,b in zip(after,reference))>settings['home_joint_tolerance_deg']):
                raise ValueError('Wrist comparison pose differs from original home capture')
            return decode_color(select_image(images))
        await asyncio.sleep(settings['settle_s'])
        first=await asyncio.wait_for(capture_one(),settings['capture_timeout_s'])
        await asyncio.sleep(settings['sample_interval_s'])
        second=await asyncio.wait_for(capture_one(),settings['capture_timeout_s'])
        return first,second

    async def capture_and_inspect(self,inspector):
        self.captured_at=time.monotonic()
        if self.fake:
            self.wrist=await inspector.capture_and_inspect()
            image=Image.open(Path(self.wrist['output_dir'])/'frame.png').convert('RGB')
            rotation=np.eye(3);translation=np.zeros(3)
            objects=[]
            for index,crop in enumerate(self.wrist['localization']['objects']):
                l,t,r,b=crop['bbox_xyxy'];ys,xs=np.mgrid[t:b,l:r]
                point=np.array([400.+40*index,0.,0.])
                objects.append(SimpleNamespace(idx=100+index,kind='nut',source='depth',
                    u=crop['center_uv'][0],v=crop['center_uv'][1],xs=xs.ravel(),ys=ys.ravel(),
                    p_table=point,p_axis=point+[10.,0.,0.]))
            meta=dict(fake=True,coordinate_source='synthetic test coordinates')
        else:
            await self.st.require_idle_controller()
            before=list((await self.st.arm.get_joint_positions()).values)
            rotation,translation=await self.st.cam_frame()
            depth,gray,bgr,_,summary,in_frame=await self.st.capture()
            after=list((await self.st.arm.get_joint_positions()).values)
            await self.st.require_idle_controller()
            if (not before or len(before)!=len(after) or not np.isfinite(before+after).all()
                    or max(abs(a-b) for a,b in zip(before,after))>self.mapping['max_joint_change_deg']):
                raise ValueError('Arm moved during shared RGB/depth capture')
            if time.monotonic()-self.captured_at>self.mapping['max_capture_s']:
                raise ValueError('Shared capture exceeded time limit')
            if bgr is None or in_frame<.6:raise ValueError('Missing or misaligned RGB/depth capture')
            image=Image.fromarray(bgr[:,:,::-1].copy())
            if image.size!=(self.st.K.width,self.st.K.height):raise ValueError('RGB/intrinsics dimensions differ')
            _,depth_info=picker.detect_depth(depth,self.st.K,bgr,gray)
            self.wrist=inspector.process(image)  # EXACT RGB frame associated with these depth objects
            folder=Path(self.wrist['output_dir'])
            np.save(folder/'picker_depth.npy',depth)
            if gray is not None:np.save(folder/'picker_gray.npy',gray)
            self.reference_joints=before
            meta=dict(fake=False,intrinsics=asdict(self.st.K),rotation=rotation.tolist(),
                translation_mm=translation.tolist(),joints_before_deg=before,joints_after_deg=after,
                acquisition='sequential RGB/depth, stationary pose; not hardware synchronized')
        folder=Path(self.wrist['output_dir'])
        if self.fake:self.fake_scene=image.copy()
        if not self.wrist.get('localization'):
            raise ValueError('External boxes currently require automatic crop localization')
        mask=np.asarray(Image.open(folder/'foreground_mask.png'))>0
        if self.fake:
            self.rows=bind_crops(self.wrist['localization']['objects'],objects,mask,rotation,translation,
                                self.mapping,picker.PICK_KINDS)
            for row in self.rows:row['height_mm']=14.0
        else:
            self.rows=bind_rois(self.wrist['localization']['objects'],mask,depth_info,self.st.K,
                                rotation,translation,self.mapping)
        (folder/'picker_bindings.json').write_text(json.dumps(dict(capture=meta,objects=self.rows,
             actions_executed=False,physically_validated=False),indent=2,allow_nan=False))
        print(f"Pipeline boxes: {sum(r['status']=='matched' for r in self.rows)}/{len(self.rows)} localized from supplied ROIs; no reclassification.",flush=True)
        return self.wrist

    async def execute(self,decisions,checkpoint=None,after_attempt=None):
        # Optional continuous-mode hook runs only after a fully completed attempt
        # AND delivery/return. Failures propagate without invoking recovery/home.
        if self.wrist is None:raise ValueError('No shared batch capture')
        by_id={d['id']:d for d in decisions}
        if len(by_id)!=len(decisions):raise ValueError('Duplicate decision IDs')
        record_by_id={o['id']:o for o in self.wrist['objects']}
        results=[];prepared=False
        report=dict(target_source='pipeline_boxes',fake=self.fake,sorting_executed=False,objects=results)
        folder=Path(self.wrist['output_dir'])
        def save():
            (folder/'batch_pick.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        save()
        order=self.settings.get('pick_order','tallest_first')
        if order not in ('tallest_first','input'):raise ValueError('Invalid pick_order')
        pending=list(self.rows)
        report['pick_order']=order
        completed=None
        while True:
            if completed is not None and after_attempt is not None:
                check=await after_attempt(completed)
                completed['home_check']=check
                prepared=False  # Hook returned home; next target must go to scan again.
                if check['status']=='reset_requested':
                    report.update(status='reset_requested',pending_crop_ids=[r['crop_id'] for r in pending])
                    save()
                    return report
                if check['status']!='unchanged':raise ValueError('Invalid after-attempt result')
                save()
            completed=None
            # Replan remaining work only; no new inference or geometry assumptions.
            if order=='tallest_first':pending.sort(key=lambda r:-float(r.get('height_mm',0.)))
            # Before the first target AND after each attempt/delivery (including the last).
            # Nothing consumes events between grasp and delivery/return.
            if checkpoint is not None:
                signal=checkpoint()
                report.setdefault('runtime_events',[]).extend(asdict(e) for e in signal.events)
                if signal.reset_requested:
                    report.update(status='reset_requested',pending_crop_ids=[r['crop_id'] for r in pending])
                    save()
                    return report
            if not pending:break
            binding=pending.pop(0)  # HOLD/refused/returned rejects are not retried this batch.
            ident=binding['crop_id'];decision=by_id.get(ident,{})
            action=decision.get('final_action','hold')
            entry=dict(crop_id=ident,action=action,binding=binding,status='pending',sorting_executed=False)
            results.append(entry)
            completed=entry
            if action not in ('accept','reject'):
                entry['status']='hold';print(f'Batch crop {ident}: HOLD, not grasped.');save();continue
            if binding['status']!='matched':
                entry['status']='unmatched'
                print(f"Batch crop {ident}: not grasped — {binding.get('reason','no unique depth match')}",flush=True)
                save();continue
            if time.monotonic()-self.captured_at>self.settings['max_batch_age_s']:
                entry['status']='batch_expired';print(f'Batch crop {ident}: batch expired; not grasped.');save();continue
            if action=='accept' and self.st.placed['box']>=self.settings['max_good_places']:
                entry['status']='good_area_full';print(f'Batch crop {ident}: GOOD area full; not grasped.');save();continue
            xy=np.array(binding['table_point_world_mm'][:2])
            if any(picker.dist_to_rect(xy,picker.place_area(s))==0 for s in
                   (self.settings['good_xy'],self.settings['reject_xy']) if s is not None):
                entry['status']='in_place_area';print(f'Batch crop {ident}: already in a place area; not picked.');save();continue
            route={'accept':'box','reject':'reject'}[action]
            print(f"Batch crop {ident}: {action} -> existing picker at world XY {xy.tolist()}",flush=True)
            if self.fake:
                self.home_ready=False
                self.st.actions.append(dict(simulated_pick=xy.tolist(),crop_id=ident))
                result=dict(status='held',decision=route,xy=xy,grasp_z=69.)
            else:
                await self.st.require_idle_controller()
                if (await self.st.gripper.is_holding_something()).is_holding_something:
                    raise ValueError('Gripper holds an item; stopping batch')
                if not prepared:
                    if not self.home_ready:await self.st.go_home()
                    setup_good_destination(self.st,self.settings)
                    await self.st.go_scan();prepared=True
                self.home_ready=False
                # Deliberately no image extraction, model, VLM call or whole-scene comparison.
                async def verdict(st,item,target,info,objects):
                    if time.monotonic()-self.captured_at>self.settings['max_batch_age_s']:
                        return 'hold',None,0.
                    # A standing part has no blob (target None): pick_one aimed at its top within STANDING_AIM_MM.
                    standing=target is None and item.get('kind')=='bolt-standing'
                    actual=np.asarray(item['xy']) if standing else st.locate(target,info)[0]
                    limit=picker.STANDING_AIM_MM if standing else self.settings['target_match_radius_mm']
                    if np.linalg.norm(actual[:2]-xy)>limit:
                        return 'hold',None,0.
                    item['pipeline_crop_id']=ident
                    return route,record_by_id[ident]['score'],0.
                self.st.inspection_handler=verdict
                debug=folder/f'pick_{ident}';debug.mkdir(exist_ok=True)
                self.st.args.debug_dir=str(debug)
                def supplied_roi(info):
                    return target_in_view(binding,info,self.st.K,self.mapping)
                # Scan from straight above the target: from the fixed scan pose a part near the edge of the view
                # (x~260, by the base) projected its ROI outside the frame and was never picked.
                above=self.st.gripper_xyz_for_camera([xy[0],xy[1],picker.TABLE_Z_EXPECTED+picker.SCAN_HEIGHT_MM])
                if picker.reachable(above):
                    await self.st.move_gripper(above,picker.NATURAL_THETA,'camera above the target (scan height)',linear=True)
                result=await picker.pick_one(self.st,'pick',[],target_world_xy=xy,
                    target_tolerance_mm=self.settings['target_match_radius_mm'],roi_target=supplied_roi,
                    target_height_mm=binding.get('height_mm'))
            entry.update(status=result['status'],why=result.get('why'),picker_decision=result.get('decision'),
                         picker_target_xy_mm=np.asarray(result['xy']).tolist() if result.get('xy') is not None else None)
            save()  # retain incomplete attempt if delivery subsequently fails
            placed_before = self.st.placed.get(route, 0)
            delivery=await execute_result(self.st,result,[])
            # Placement counters advance only after successful destination release.
            # Full-area put-backs and reject-return do NOT remove the source object.
            entry['removed_from_source'] = (result['status']=='held'
                and self.st.placed.get(route, 0)==placed_before+1)
            print(f"Batch crop {ident}: {delivery}"+(f" — {result['why']}" if result.get('why') else ''),flush=True)
            entry.update(delivery=delivery,sorting_executed=not self.fake and result['status']=='held')
            report['sorting_executed']|=entry['sorting_executed']
            if self.fake:entry['simulated_actions']=list(self.st.actions);self.st.actions.clear()
            save()
        report['status']='batch_complete';save()
        return report
