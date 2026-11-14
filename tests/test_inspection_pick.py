import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch

import numpy as np
import pytest
import yaml
import pick_bolt as picker
from robot.inspection_pick import (FakeStation,InspectionBridge,IntegratedStation,
                                  execute_result,match_target,run_pick_loop)

ROOT=Path(__file__).resolve().parents[1]


def settings():return yaml.safe_load((ROOT/'config/pick_bolt.yaml').read_text())


def target():
    return SimpleNamespace(idx=9,u=100,v=100,xs=np.array([90,110]),ys=np.array([90,110]))


def test_matches_target_geometry_not_frame_id():
    obj={'id':77,'bbox_xyxy':[90,90,111,111],'center_uv':[100,100]}
    assert match_target(target(),[obj],640,settings()) is obj


def test_ambiguous_or_missing_target_refuses():
    obj={'id':1,'bbox_xyxy':[90,90,111,111],'center_uv':[100,100]}
    for objects in [[],[obj,dict(obj,id=2)]]:
        with pytest.raises(ValueError,match='Expected one'):
            match_target(target(),objects,640,settings())


@pytest.mark.parametrize('decision,reject_xy,expected_xy',[('box',settings()['reject_xy'],settings()['good_xy']),
                                                          ('reject',settings()['reject_xy'],settings()['reject_xy']),
                                                          ('reject',None,[400,0])])
def test_accept_places_good_reject_goes_to_reject_area_or_back(decision,reject_xy,expected_xy):
    st=FakeStation(dict(settings(),reject_xy=reject_xy));skip=[]
    result=dict(status='held',decision=decision,xy=np.array([400.,0.]),grasp_z=69.)
    asyncio.run(execute_result(st,result,skip))
    releases=[a['open_at'] for a in st.actions if 'open_at' in a]
    assert len(releases)==1 and releases[0][:2]==expected_xy
    assert len(skip)==1


def test_next_run_does_not_place_on_top_of_parts_already_there():
    table=[];spots=[]
    for _ in range(3):  # three runs: each station is new, the parts stay on the table
        st=FakeStation(settings());st.lying=table
        asyncio.run(execute_result(st,dict(status='held',decision='box',xy=np.array([400.,0.]),grasp_z=69.,
                                           size=(20.,20.,20.)),[]))
        spots+=[a['open_at'][:2] for a in st.actions if 'open_at' in a]
    assert len(spots)==3 and min(np.hypot(*np.subtract(a,b)) for i,a in enumerate(spots) for b in spots[:i])>=28


def test_hold_does_not_release_or_place():
    st=FakeStation(settings())
    asyncio.run(execute_result(st,dict(status='inspection_hold',xy=np.array([400.,0.])),[]))
    assert st.actions==[]


def test_decision_hook_overrides_legacy_accept_stub():
    st=SimpleNamespace(inspection_handler=AsyncMock(return_value=('reject',.9,0.0)),vision=None)
    assert asyncio.run(picker.decide(st,{'score':1.},target(),{},[]))[0]=='reject'
    st.inspection_handler.assert_awaited_once()


def test_invalid_hook_answer_cannot_default_to_accept():
    st=SimpleNamespace(inspection_handler=AsyncMock(return_value=('bad',0.,0.)),vision=None)
    with pytest.raises(ValueError):asyncio.run(picker.decide(st,{},target(),{},[]))


def test_missing_confirmation_refuses_before_loading_or_connecting(tmp_path):
    with patch('robot.inspection_pick.InspectionBridge') as bridge:
        with pytest.raises(ValueError,match='confirm-motion'):
            asyncio.run(run_pick_loop({},tmp_path,confirm_motion=False))
    bridge.assert_not_called()


def test_stale_frame_holds_without_inference(tmp_path):
    bridge=object.__new__(InspectionBridge)
    bridge.root=tmp_path;bridge.settings=settings();bridge.fake=False
    info={'bgr':np.zeros((480,640,3),dtype=np.uint8),'capture_monotonic':time.monotonic()-60}
    assert asyncio.run(bridge(None,{},target(),info,[]))==('hold',None,0.0)


def test_valid_reject_does_not_recapture_or_compare_scene(tmp_path):
    from PIL import Image,ImageDraw
    config=yaml.safe_load((ROOT/'config/config.yaml').read_text())
    bridge=InspectionBridge(ROOT,config,settings(),fake=True)
    bridge.root=tmp_path;bridge.fake=False  # exercise live branch with mocked inference and hardware
    st=FakeStation(settings())
    st.require_idle_controller=AsyncMock()
    st.capture=AsyncMock(side_effect=AssertionError('Unexpected scene recapture'))
    image=Image.new('RGB',(640,480),'gray')
    ImageDraw.Draw(image).rectangle((250,210,330,240),fill='black')
    t=SimpleNamespace(idx=1,u=290,v=225,xs=np.array([250,330]),ys=np.array([210,240]))
    info=dict(bgr=np.asarray(image)[:,:,::-1].copy(),capture_monotonic=time.monotonic(),in_frame=1)
    with patch('robot.inspection_pick.decide_wrist_objects',new_callable=AsyncMock,
               return_value=[{'final_action':'reject'}]), \
         patch('pick_bolt.settled',side_effect=AssertionError('Unexpected scene comparison')) as compare:
        result=asyncio.run(bridge(st,{'xy':np.array([400.,0.])},t,info,[t]))
    assert result[0]=='reject'
    st.require_idle_controller.assert_awaited_once()
    st.capture.assert_not_awaited();compare.assert_not_called()


def test_hold_hook_prevents_actual_grasp_path():
    args=picker.parse_args([])
    t=SimpleNamespace(idx=1,kind='bolt-lying',area_mm2=1000.,length_mm=83.,width_mm=31.,height_mm=22.,score=.9)
    st=SimpleNamespace(args=args,box=None,vision=None,K=None,t_gc=np.zeros(3),
        look=AsyncMock(return_value=([t],{'seen_at':time.time()})),
        locate=lambda *_:(np.array([400.,0.,0.]),0.),
        inspection_handler=AsyncMock(return_value=('hold',None,0.)),
        open_jaws=AsyncMock(),move_gripper=AsyncMock())
    with patch('pick_bolt.blocked_by_tall',return_value=None):
        result=asyncio.run(picker.pick_one(st,'pick',[]))
    assert result['status']=='inspection_hold'
    st.open_jaws.assert_not_awaited();st.move_gripper.assert_not_awaited()


def test_standalone_motion_is_disabled_before_connection():
    args=picker.parse_args(['--mode','sort'])
    with patch('pick_bolt.Station.connect',new_callable=AsyncMock) as connect:
        with pytest.raises(ValueError,match='Standalone motion'):
            asyncio.run(picker.run(args))
    connect.assert_not_awaited()


def test_fake_pipeline_never_connects_or_loads_real_model(tmp_path):
    import shutil
    shutil.copytree(ROOT/'config',tmp_path/'config')
    config=yaml.safe_load((tmp_path/'config/config.yaml').read_text())
    with patch('robot.inspection_pick.IntegratedStation.connect',new_callable=AsyncMock) as connect, \
         patch('vlm.service.connect',new_callable=AsyncMock) as vlm_connect, \
         patch('anomaly.wrist.OpenVINODetector') as detector:
        asyncio.run(run_pick_loop(config,tmp_path,fake=True,cycles=1))
    connect.assert_not_awaited();vlm_connect.assert_not_awaited();detector.assert_not_called()
    assert list((tmp_path/'logs/inspection_pick').glob('run_*/cycle_001.json'))
