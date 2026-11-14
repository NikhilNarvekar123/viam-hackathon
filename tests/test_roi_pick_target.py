import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
import numpy as np
import pytest
import pick_bolt as picker
from robot.roi_pick_target import bind_rois,measure_roi,validate_grasp_target


def scene(height=3.):
    K=picker.Intrinsics(600,600,320,240,640,480)
    mask=np.zeros((480,640),bool);mask[230:250,310:330]=True
    depth=np.full(mask.shape,440.);depth[mask]-=height
    heights=np.zeros_like(depth);heights[mask]=height
    info=dict(depth=depth,height=heights,foot=np.array([0.,0.,440.]),cam_height_mm=440.)
    crop=dict(id=7,bbox_xyxy=[310,230,330,250],center_uv=[319.5,239.5])
    settings=dict(roi_min_valid_pixels=12,roi_min_valid_fraction=.3,roi_height_percentile=95.)
    return K,mask,info,crop,settings


def test_low_roi_localizes_without_any_detector_proposals_but_needs_closeup():
    K,mask,info,crop,settings=scene(3.)
    rows=bind_rois([crop],mask,info,K,np.eye(3),np.zeros(3),settings)
    assert rows[0]['status']=='matched' and rows[0]['height_mm']==3.
    obj,_,_=measure_roi(crop,mask,info,K,settings)
    assert 'too low' in validate_grasp_target(obj)


def test_missing_depth_is_not_replaced_by_table_estimate():
    K,mask,info,crop,settings=scene();info['depth'][mask]=0
    assert bind_rois([crop],mask,info,K,np.eye(3),np.zeros(3),settings)[0]['status']=='unmatched'


def test_split_height_regions_do_not_require_two_proposal_matches():
    K,mask,info,crop,settings=scene(15.)
    info['height'][240:250,310:330]=80.
    rows=bind_rois([crop],mask,info,K,np.eye(3),np.zeros(3),settings)
    assert rows[0]['status']=='matched' and rows[0]['height_mm']==80.


def test_supplied_roi_reaches_closeup_when_native_detector_finds_nothing():
    obj=SimpleNamespace(idx=1,kind='object',u=320.,v=240.,length_mm=25.,width_mm=25.,
                        height_mm=10.,area_mm2=300.,score=1.)
    info={'seen_at':time.time()}
    st=SimpleNamespace(args=picker.parse_args([]),box=None,vision=None,t_gc=np.zeros(3),
        K=SimpleNamespace(cx=320,cy=240),look=AsyncMock(return_value=([],info)),
        locate=lambda *_:(np.array([400.,0.,0.]),0.),gripper_xyz_for_camera=lambda xyz:xyz,
        move_gripper=AsyncMock(),open_jaws=AsyncMock(),inspection_handler=AsyncMock(return_value=('hold',None,0.)))
    with patch('pick_bolt.blocked_by_tall',return_value=None):
        result=asyncio.run(picker.pick_one(st,'pick',[],target_world_xy=np.array([400.,0.]),
                          target_tolerance_mm=15.,roi_target=lambda _:obj))
    assert result['status']=='inspection_hold'
    assert st.look.await_count==2
    st.move_gripper.assert_awaited_once()  # approach for close-up, not descent/grasp
    st.open_jaws.assert_not_awaited()
