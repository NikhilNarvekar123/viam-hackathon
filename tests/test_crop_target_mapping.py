import numpy as np
import pytest
from types import SimpleNamespace
from robot.crop_target_mapping import bind_crops


def inputs():
    mask=np.zeros((100,100),bool);mask[40:51,40:51]=True
    ys,xs=np.where(mask)
    obj=SimpleNamespace(idx=99,kind='nut',source='depth',xs=xs,ys=ys,u=45,v=45,
                        p_table=np.array([10.,20.,30.]),p_axis=np.array([11.,20.,30.]))
    crop=dict(id=7,bbox_xyxy=[40,40,51,51],center_uv=[45,45])
    return crop,obj,mask


def bind(crops,objects,mask,R=None):
    return bind_crops(crops,objects,mask,np.eye(3) if R is None else R,[100,200,300],
                      dict(min_foreground_overlap=.4,max_center_distance_px=12),('nut',))


def test_pixel_match_not_id_number_and_world_transform():
    crop,obj,mask=inputs();row=bind([crop],[obj],mask)[0]
    assert row['crop_id']==7 and row['depth_id']==99 and row['status']=='matched'
    assert row['table_point_world_mm']==[110,220,330]


def test_two_crops_cannot_claim_same_target():
    crop,obj,mask=inputs()
    rows=bind([crop,dict(crop,id=8)],[obj],mask)
    assert all(r['status']=='unmatched' for r in rows)


def test_one_crop_cannot_choose_between_depth_objects():
    crop,obj,mask=inputs()
    assert bind([crop],[obj,obj],mask)[0]['status']=='unmatched'


def test_color_only_target_does_not_supply_depth_binding():
    crop,obj,mask=inputs();obj.source='dark'
    assert bind([crop],[obj],mask)[0]['status']=='unmatched'


def test_empty_depth_is_unmatched():
    crop,_,mask=inputs()
    assert bind([crop],[],mask)[0]['status']=='unmatched'


def test_nonrigid_transform_refused():
    crop,obj,mask=inputs()
    with pytest.raises(ValueError,match='rigid'):
        bind([crop],[obj],mask,R=np.eye(3)*2)
