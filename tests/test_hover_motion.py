from types import SimpleNamespace
import numpy as np
import pytest
from robot.hover_motion import calculate_hover, box_corners


def test_hover_uses_gripper_bottom_and_camera_offset():
    # Endpoint at z=458, lowest gripper at 303 (155 below).
    # Camera is 80 mm to the right of endpoint and points straight down.
    basis=[[380,0,440],[480,0,440],[380,-100,440],[380,0,340]]
    target=calculate_hover([300,0,458],303,basis,[320,-90],-23,101.6)
    np.testing.assert_allclose(target,[240,-90,233.6])
    assert target[2]-155-(-23)==pytest.approx(101.6)


def test_camera_tilt_is_accounted_for():
    direction=np.array([0.1,0,-np.sqrt(0.99)])
    basis=np.array([[380,0,440],[480,0,440],[380,-100,440],[380,0,440]+100*direction])
    target=calculate_hover([300,0,458],303,basis,[320,-90],-23,101.6)
    cam=np.array(target)+[80,0,-18]
    hit=cam+((-23-cam[2])/direction[2])*direction
    np.testing.assert_allclose(hit,[320,-90,-23],atol=1e-8)


def test_reject_camera_not_downward():
    with pytest.raises(ValueError,match='downward'):
        calculate_hover([0,0,400],245,[[0,0,400],[100,0,400],[0,100,400],[0,0,500]],[100,100],-23,101.6)


def test_gripper_box_vertices():
    g=SimpleNamespace(HasField=lambda key:key=='box',
        center=SimpleNamespace(x=0,y=0,z=-6,theta=0,o_x=0,o_y=0,o_z=1),
        box=SimpleNamespace(dims_mm=SimpleNamespace(x=45,y=120,z=112)))
    corners=box_corners(g)
    assert len(corners)==8
    assert max(c[2] for c in corners)==50
    assert min(c[2] for c in corners)==-62
