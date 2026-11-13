import pytest
from tools.overview_hover import map_point, motion_targets


def mapping():
    return {'markers':[{'overview_uv':p} for p in [(0,0),(10,0),(10,10),(0,10)]],
            'homography_overview_to_world_xy':[[2,0,100],[0,3,200],[0,0,1]]}


def test_mapping():
    assert map_point([5,5],mapping()) == [110,215]


def test_no_extrapolation():
    with pytest.raises(ValueError,match='Outside'):
        map_point([11,5],mapping())


def test_safety_block():
    with pytest.raises(ValueError,match='blocked'):
        motion_targets([],{'safety_verified':False})


def motion():
    return dict(safety_verified=True,driver_speed_and_paths_verified=True,
        hover_endpoint_z_mm=400,inspection_point_minus_endpoint_xy_mm=[10,20],
        endpoint_xy_bounds_mm=[0,0,200,200],orientation=dict(o_x=0,o_y=0,o_z=-1,theta=0),max_segment_mm=50)


def test_endpoint_offset():
    assert motion_targets([{'id':1,'world_xy_mm':[110,120]}],motion()) == [dict(id=1,x=100.0,y=100.0,z=400.0)]


def test_bounds():
    with pytest.raises(ValueError,match='outside'):
        motion_targets([{'id':1,'world_xy_mm':[999,120]}],motion())


def test_no_missing_height():
    cfg=motion();cfg['hover_endpoint_z_mm']=None
    with pytest.raises(ValueError,match='missing'):
        motion_targets([],cfg)
