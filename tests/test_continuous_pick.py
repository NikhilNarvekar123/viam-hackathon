import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import yaml

from anomaly.wrist import WristInspector
from core.runtime_events import EventRuntime
from robot.continuous_pick import run_continuous, HomeMonitor
from robot.external_box_pick import ExternalBoxBatch

ROOT=Path(__file__).resolve().parents[1]


def setup(tmp_path):
    shutil.copytree(ROOT/'config',tmp_path/'config')
    config=yaml.safe_load((tmp_path/'config/config.yaml').read_text())
    settings=yaml.safe_load((tmp_path/'config/continuous_pick.yaml').read_text())
    settings.update(idle_poll_s=.001,sample_interval_s=.001,settle_s=0)
    return config,settings


def decisions():
    return [dict(id=1,final_action='accept'),dict(id=2,final_action='reject'),dict(id=3,final_action='hold')]


def test_unchanged_scene_reuses_models_and_idles_without_retry_or_overview(tmp_path):
    config,settings=setup(tmp_path)
    stop=asyncio.Event();samples=[];homes=[]
    sample=HomeMonitor.sample;home=ExternalBoxBatch.home_for_check;process=WristInspector.process
    async def monitored(self,*args,**kwargs):
        result=await sample(self,*args,**kwargs)
        samples.append(result)
        if len(samples)==6:stop.set()
        return result
    async def home_count(self):
        homes.append(True)
        return await home(self)
    with patch.object(HomeMonitor,'sample',monitored), \
         patch.object(ExternalBoxBatch,'home_for_check',home_count), \
         patch.object(WristInspector,'process',autospec=True,side_effect=process) as inference, \
         patch('robot.continuous_pick.decide_wrist_objects',new_callable=AsyncMock,return_value=decisions()) as vlm, \
         patch('robot.overview.overview_stage',side_effect=AssertionError('Side camera must not be used')), \
         patch('main.wait_for_next_cycle',side_effect=AssertionError('No ENTER pauses')):
        result=asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,stop=stop))
    assert result['status']=='stopped' and result['batches']==1
    assert result['state']['placed']['box']==1
    inference.assert_called_once();vlm.assert_awaited_once()
    assert len(homes)==4  # initial + ACCEPT + REJECT + HOLD
    assert all(row['status']=='unchanged' for row in samples)
    report=json.loads(next(tmp_path.rglob('batch_pick.json')).read_text())
    assert [row['action'] for row in report['objects']]==['accept','reject','hold']
    assert all(row['home_check']['status']=='unchanged' for row in report['objects'])


@pytest.mark.parametrize('kind',['reset_batch','scene_changed'])
def test_request_during_delivery_waits_then_fresh_inspection_preserves_placement_count(tmp_path,kind):
    config,settings=setup(tmp_path)
    runtime=EventRuntime(config['runtime_events']);stop=asyncio.Event();phase=[];inspections=[]
    from robot.inspection_pick import execute_result
    async def delivery(st,result,skip):
        phase.append('delivery_started')
        runtime.publish(kind,'audio' if kind=='reset_batch' else 'overview')
        out=await execute_result(st,result,skip)
        phase.append('delivery_finished')
        return out
    home=ExternalBoxBatch.home_for_check
    async def homing(self):
        phase.append('home')
        await home(self)
    async def inspect(wrist,*args,**kwargs):
        inspections.append(wrist['output_dir'])
        if len(inspections)==2:stop.set()
        return decisions()
    original_checkpoint=runtime.checkpoint
    def checkpoint():
        assert not phase or phase[-1]!='delivery_started'
        return original_checkpoint()
    with patch.object(ExternalBoxBatch,'home_for_check',homing), \
         patch.object(runtime,'checkpoint',checkpoint), \
         patch('robot.external_box_pick.execute_result',side_effect=delivery), \
         patch('robot.continuous_pick.decide_wrist_objects',side_effect=inspect):
        result=asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,event_runtime=runtime,stop=stop))
    assert len(inspections)==2
    first=json.loads((Path(inspections[0])/'batch_pick.json').read_text())
    assert len(first['objects'])==1 and first['status']=='reset_requested'
    assert first['pending_crop_ids']==[2,3]
    assert phase[:4]==['home','delivery_started','delivery_finished','home']
    assert result['state']['placed']['box']==1


def test_scene_change_discards_old_targets_and_reinspects_without_channel_event(tmp_path):
    config,settings=setup(tmp_path);stop=asyncio.Event();inspections=[]
    after=HomeMonitor.after_attempt
    async def changed(self,entry):
        self.batch.fake_scene.paste('black',(550,80,580,110))
        return await after(self,entry)
    async def inspect(wrist,*args,**kwargs):
        inspections.append(wrist['output_dir'])
        if len(inspections)==2:stop.set()
        return decisions()
    with patch.object(HomeMonitor,'after_attempt',changed), \
         patch('robot.continuous_pick.decide_wrist_objects',side_effect=inspect):
        asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,stop=stop))
    first=json.loads((Path(inspections[0])/'batch_pick.json').read_text())
    assert len(inspections)==2 and len(first['objects'])==1
    assert first['objects'][0]['home_check']['scene']['status']=='changed'


def test_event_during_inference_prevents_first_pick(tmp_path):
    config,settings=setup(tmp_path);stop=asyncio.Event();runtime=EventRuntime(config['runtime_events'])
    calls=[]
    async def inspect(wrist,*args,**kwargs):
        calls.append(wrist['output_dir'])
        runtime.publish('reset_batch','audio')
        if len(calls)==2:stop.set()
        return decisions()
    with patch('robot.continuous_pick.decide_wrist_objects',side_effect=inspect), \
         patch('robot.external_box_pick.execute_result',side_effect=AssertionError('No delivery expected')):
        asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,event_runtime=runtime,stop=stop))
    assert len(calls)==2
    assert not json.loads((Path(calls[0])/'batch_pick.json').read_text())['objects']


def test_idle_scene_update_starts_fresh_inspection_not_repeated_rejects(tmp_path):
    config,settings=setup(tmp_path);stop=asyncio.Event();inspections=[]
    sample=HomeMonitor.sample
    async def sample_with_update(self,*args,**kwargs):
        if self.index==4:  # Initial preflight + three attempts completed; now idle.
            self.batch.fake_scene.paste('black',(550,80,580,110))
        return await sample(self,*args,**kwargs)
    async def inspect(wrist,*args,**kwargs):
        inspections.append(wrist['output_dir'])
        if len(inspections)==2:stop.set()
        return decisions()
    with patch.object(HomeMonitor,'sample',sample_with_update), \
         patch('robot.continuous_pick.decide_wrist_objects',side_effect=inspect):
        result=asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,stop=stop))
    assert len(inspections)==2 and result['state']['placed']['box']==1
    first=json.loads((Path(inspections[0])/'batch_pick.json').read_text())
    assert first['status']=='batch_complete' and len(first['objects'])==3


def test_event_after_last_hold_is_checked_even_with_no_pending_targets(tmp_path):
    config,settings=setup(tmp_path);stop=asyncio.Event();runtime=EventRuntime(config['runtime_events'])
    sample=HomeMonitor.sample;inspections=[]
    async def sampled(self,*args,**kwargs):
        if self.index==3:runtime.publish('reset_batch','audio')
        return await sample(self,*args,**kwargs)
    async def inspect(wrist,*args,**kwargs):
        inspections.append(wrist['output_dir'])
        if len(inspections)==2:stop.set()
        return decisions()
    with patch.object(HomeMonitor,'sample',sampled), \
         patch('robot.continuous_pick.decide_wrist_objects',side_effect=inspect):
        asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,event_runtime=runtime,stop=stop))
    first=json.loads((Path(inspections[0])/'batch_pick.json').read_text())
    assert first['status']=='reset_requested' and first['pending_crop_ids']==[]
    assert [row['action'] for row in first['objects']]==['accept','reject','hold']


def test_delivery_failure_never_calls_post_delivery_home_or_reset(tmp_path):
    config,settings=setup(tmp_path);homes=[]
    home=ExternalBoxBatch.home_for_check
    async def homing(self):
        homes.append(True);await home(self)
    with patch.object(ExternalBoxBatch,'home_for_check',homing), \
         patch('robot.continuous_pick.decide_wrist_objects',new_callable=AsyncMock,return_value=decisions()), \
         patch('robot.external_box_pick.execute_result',new_callable=AsyncMock,side_effect=ValueError('delivery failed')):
        with pytest.raises(ValueError,match='delivery failed'):
            asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings))
    assert len(homes)==1


def test_gripper_still_holding_refuses_home_open(tmp_path):
    config,_=setup(tmp_path)
    batch=ExternalBoxBatch(tmp_path,config,fake=True)
    batch.fake=False
    batch.st=SimpleNamespace(require_idle_controller=AsyncMock(),
                             gripper=SimpleNamespace(is_holding_something=AsyncMock(
                                 return_value=SimpleNamespace(is_holding_something=True))),
                             go_home=AsyncMock())
    with pytest.raises(ValueError,match='still holds'):
        asyncio.run(batch.home_for_check())
    batch.st.go_home.assert_not_awaited()
    assert not batch.home_ready


def test_fault_latch_refuses_initial_home(tmp_path):
    config,settings=setup(tmp_path);runtime=EventRuntime(config['runtime_events'])
    for _ in range(config['runtime_events']['queue_capacity']+1):runtime.publish('notice','audio')
    with patch.object(ExternalBoxBatch,'home_for_check',new_callable=AsyncMock) as home:
        with pytest.raises(RuntimeError,match='overflow'):
            asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings,event_runtime=runtime))
    home.assert_not_awaited()


def test_missing_motion_authorization_refused_before_model_or_hardware(tmp_path):
    config,settings=setup(tmp_path)
    with patch('robot.continuous_pick.WristInspector',side_effect=AssertionError('No model construction')):
        with pytest.raises(ValueError,match='confirm-motion'):
            asyncio.run(run_continuous(config,tmp_path,settings=settings))


def test_invalid_settings_refused_before_model_or_hardware(tmp_path):
    config,settings=setup(tmp_path);settings['validation']['min_change_area_px']=0
    with patch('robot.continuous_pick.WristInspector',side_effect=AssertionError('No model construction')):
        with pytest.raises(ValueError,match='min_change_area_px'):
            asyncio.run(run_continuous(config,tmp_path,fake=True,settings=settings))


@pytest.mark.parametrize('problem',['pose','stale','missing_timestamp'])
def test_live_home_capture_refuses_invalid_pose_or_freshness(tmp_path,problem):
    config,settings=setup(tmp_path)
    batch=ExternalBoxBatch(tmp_path,config,fake=True)
    batch.fake=False;batch.home_ready=True;batch.reference_joints=[0.,0.]
    import time
    stamp=time.time()-1
    timestamp=SimpleNamespace(seconds=int(stamp),nanos=int((stamp-int(stamp))*1e9))
    if problem=='stale':timestamp.seconds=1
    metadata=SimpleNamespace(captured_at=timestamp) if problem!='missing_timestamp' else None
    joints=[10.,0.] if problem=='pose' else [0.,0.]
    batch.st=SimpleNamespace(require_idle_controller=AsyncMock(),
        arm=SimpleNamespace(get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=joints))),
        cam=SimpleNamespace(get_images=AsyncMock(return_value=([],metadata))))
    with pytest.raises(ValueError):asyncio.run(batch.capture_home_frames(settings))


def test_continuous_cli_rejects_limits_and_pause_overrides(monkeypatch):
    import main
    for flags in (['--n','1'],['--pause-between-cycles'],['--inspect-only'],['--stage','pick']):
        monkeypatch.setattr('sys.argv',['main.py','--continuous','--fake',*flags])
        with pytest.raises(SystemExit) as error:main.main()
        assert error.value.code==2
