import copy
import numpy as np
import pytest
from PIL import Image,ImageDraw
from anomaly.object_crops import automatic_crops


def settings():
    return dict(reference_size_px=[640,480],background_blur_radius_px=20,
                min_area_px=80,max_area_px=12000,frame_edge_margin_px=15,
                min_crop_side_px=128,min_laplacian_variance=50)


def candidate(box,area=300,angle=None):
    l,t,r,b=box
    return dict(bbox_xyxy=box,center_uv=[(l+r)/2,(t+b)/2],area_px=area,
                image_angle_deg=angle,warnings=[])


def mock_candidates(monkeypatch,objects):
    monkeypatch.setattr('anomaly.object_crops.locate_objects',lambda *args:
        ({'objects':copy.deepcopy(objects),'workspace_roi':[0,0,640,480]},np.ones((480,640),bool)))


@pytest.mark.parametrize('scale',[1,2])
def test_border_margin_and_minimum_scale(monkeypatch,scale):
    mock_candidates(monkeypatch,[candidate([10,100,30,140]),candidate([150,100,170,140])])
    cfg=settings();cfg['min_laplacian_variance']=0
    _,result,_=automatic_crops(Image.new('RGB',(640*scale,480*scale)),cfg)
    assert len(result['objects'])==1
    assert result['objects'][0]['crop_side_px']==128*scale
    assert result['rejected_candidates'][0]['warnings']==['frame_edge_margin']


def test_smooth_shadow_rejected_but_structured_patch_retained(monkeypatch):
    mock_candidates(monkeypatch,[candidate([100,100,140,140]),candidate([200,200,240,240])])
    image=Image.new('RGB',(640,480),(90,90,90));draw=ImageDraw.Draw(image)
    for x in range(200,240,4): draw.line((x,200,x,239),fill='white',width=1)
    _,result,_=automatic_crops(image,settings())
    assert len(result['objects'])==1
    assert result['objects'][0]['bbox_xyxy']==[200,200,240,240]
    assert 'insufficient_foreground_structure' in result['rejected_candidates'][0]['warnings']


def test_fragment_suppression_not_padded_crop_overlap(monkeypatch):
    objects=[candidate([283,168,362,233],2164,145),candidate([307,209,347,239],327,144),
             candidate([344,260,368,283],377,None)]
    mock_candidates(monkeypatch,objects)
    cfg=settings();cfg['min_laplacian_variance']=0
    _,result,_=automatic_crops(Image.new('RGB',(640,480)),cfg)
    assert len(result['objects'])==2
    assert result['rejected_candidates'][0]['warnings']==['likely_fragment_of_larger_candidate']
