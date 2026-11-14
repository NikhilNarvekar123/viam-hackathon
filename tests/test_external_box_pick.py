import asyncio
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch

import numpy as np
import pytest
import yaml
import pick_bolt as picker
from anomaly.wrist import WristInspector
from robot.external_box_pick import ExternalBoxBatch,run_external_box_batch

ROOT=Path(__file__).resolve().parents[1]


def setup(tmp_path):
    shutil.copytree(ROOT/'config',tmp_path/'config')
    config=yaml.safe_load((tmp_path/'config/config.yaml').read_text())
    settings=yaml.safe_load((tmp_path/config['wrist_inspection_config']).read_text())
    inspector=WristInspector(settings,tmp_path,fake=True)
    return config,inspector


def test_original_verdicts_each_processed_once_without_second_inference(tmp_path):
    config,inspector=setup(tmp_path)
    async def exercise():
        async with ExternalBoxBatch(tmp_path,config,fake=True) as batch:
            wrist=await batch.capture_and_inspect(inspector)
            with patch.object(inspector,'process',side_effect=AssertionError('Second inference')), \
                 patch('vlm.service.decide_wrist_objects',side_effect=AssertionError('Second VLM call')):
                report=await batch.execute([{'id':1,'final_action':'accept'},
                                           {'id':2,'final_action':'reject'},
                                           {'id':3,'final_action':'hold'}])
            assert [r['status'] for r in report['objects']]==['held','held','hold']
            first,second,_=report['objects']
            releases=lambda r:[a['open_at'] for a in r['simulated_actions'] if 'open_at' in a]
            assert releases(first)[0][:2]==[300,250]
            assert releases(second)[0][:2]==second['binding']['table_point_world_mm'][:2]
            assert sum('simulated_pick' in a for r in [first,second] for a in r['simulated_actions'])==2
    asyncio.run(exercise())


def test_unmatched_target_is_not_replaced_with_any_other_object(tmp_path):
    config,inspector=setup(tmp_path)
    async def exercise():
        async with ExternalBoxBatch(tmp_path,config,fake=True) as batch:
            await batch.capture_and_inspect(inspector)
            batch.rows[0]['status']='unmatched'
            result=await batch.execute([{'id':1,'final_action':'accept'}])
            assert result['objects'][0]['status']=='unmatched'
            assert batch.st.actions==[]
    asyncio.run(exercise())


def test_real_mode_requires_confirmation(tmp_path):
    config,_=setup(tmp_path)
    with pytest.raises(ValueError,match='confirm-motion'):ExternalBoxBatch(tmp_path,config)


def test_external_scan_match_does_not_fall_back_to_biggest_target():
    obj=SimpleNamespace(kind='nut',idx=55)
    st=SimpleNamespace(args=picker.parse_args([]),box=None,
        look=AsyncMock(return_value=([obj],{})),locate=lambda *_:(np.array([500.,0.,0.]),0.))
    result=asyncio.run(picker.pick_one(st,'pick',[],target_world_xy=np.array([300.,0.]),target_tolerance_mm=15.))
    assert result['status']=='target_unmatched'


def test_pipeline_calls_vlm_once_for_entire_batch(tmp_path):
    config,inspector=setup(tmp_path)
    decisions=[{'id':i,'final_action':'accept'} for i in range(1,4)]
    with patch('vlm.service.decide_wrist_objects',new_callable=AsyncMock,return_value=decisions) as vlm:
        _,_,report=asyncio.run(run_external_box_batch(config,tmp_path,inspector,True,False,{}))
    vlm.assert_awaited_once()
    assert len(report['objects'])==3
    assert all(r['status']=='held' for r in report['objects'])


def test_live_path_passes_bound_coordinates_and_original_reject_to_picker(tmp_path):
    config,inspector=setup(tmp_path)
    async def exercise():
        async with ExternalBoxBatch(tmp_path,config,fake=True) as batch:
            await batch.capture_and_inspect(inspector)
            batch.st.require_idle_controller=AsyncMock()
            batch.st.go_home=AsyncMock()
            batch.st.gripper.is_holding_something=AsyncMock(return_value=SimpleNamespace(is_holding_something=False))
            async def pick(st,stop,skip,**kwargs):
                xy=kwargs['target_world_xy']
                st.locate=lambda *_:(np.array([*xy,0.]),0.)
                route,score,_=await st.inspection_handler(st,{},None,{},[])
                assert route=='reject'
                return dict(status='held',decision=route,xy=xy,grasp_z=69.)
            batch.fake=False  # all hardware interfaces still mocked
            try:
                with patch('pick_bolt.pick_one',side_effect=pick) as pick_call, \
                     patch('vlm.service.decide_wrist_objects',side_effect=AssertionError('Repeated inference')):
                    report=await batch.execute([{'id':1,'final_action':'reject'}])
                pick_call.assert_awaited_once()
                assert report['objects'][0]['picker_decision']=='reject'
                assert np.allclose(pick_call.call_args.kwargs['target_world_xy'],batch.rows[0]['table_point_world_mm'][:2])
            finally:batch.fake=True
    asyncio.run(exercise())


def test_batch_order_is_tallest_first_including_rejects(tmp_path):
    config,inspector=setup(tmp_path)
    async def exercise():
        async with ExternalBoxBatch(tmp_path,config,fake=True) as batch:
            await batch.capture_and_inspect(inspector)
            for row,height in zip(batch.rows,[10.,80.,20.]):row['height_mm']=height
            result=await batch.execute([{'id':1,'final_action':'accept'},
                                       {'id':2,'final_action':'reject'},
                                       {'id':3,'final_action':'accept'}])
            assert [r['crop_id'] for r in result['objects']]==[2,3,1]
            assert result['objects'][0]['picker_decision']=='reject'
    asyncio.run(exercise())


def test_expired_batch_does_not_pick(tmp_path):
    config,inspector=setup(tmp_path)
    async def exercise():
        async with ExternalBoxBatch(tmp_path,config,fake=True) as batch:
            await batch.capture_and_inspect(inspector)
            batch.captured_at-=batch.settings['max_batch_age_s']+1
            result=await batch.execute([{'id':1,'final_action':'reject'}])
            assert result['objects'][0]['status']=='batch_expired'
            assert batch.st.actions==[]
    asyncio.run(exercise())
