from pathlib import Path
import copy

import numpy as np
from PIL import Image
import pytest
import yaml

from robot.home_scene import HomeScene

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    settings = yaml.safe_load((ROOT/'config/continuous_pick.yaml').read_text())['validation']
    settings['reference_size_px'] = [160,120]
    crop = yaml.safe_load((ROOT/'config/wrist_inspection.yaml').read_text())['auto_crop']
    crop['reference_size_px'] = [160,120]
    rgb = np.full((120,160,3),160,dtype=np.uint8)
    mask = np.zeros((120,160),dtype=bool)
    boxes = [[20,20,32,36],[70,40,82,56]]
    objects = [dict(id=i+1,bbox_xyxy=box) for i,box in enumerate(boxes)]
    for x0,y0,x1,y1 in boxes:
        rgb[y0:y1,x0:x1] = 20
        mask[y0:y1,x0:x1] = True
    image = Image.fromarray(rgb)
    return HomeScene(image,mask,objects,crop,settings),rgb,settings


def check(scene, rgb, removed=None):
    image=Image.fromarray(rgb)
    return scene.check(image,image,removed)


def test_unchanged_scene_retains_original_reference():
    scene,rgb,_=fixture()
    before=scene.reference.copy()
    assert check(scene,rgb)['status']=='unchanged'
    assert np.array_equal(before,scene.reference)


def test_expected_accept_removal_is_allowed_without_refreshing_remaining_scene():
    scene,rgb,_=fixture()
    rgb[20:36,20:32]=160
    assert check(scene,rgb,1)['status']=='unchanged'
    assert 1 not in scene.boxes and 2 in scene.boxes
    assert check(scene,rgb)['status']=='unchanged'


def test_two_expected_removals_can_be_accepted_sequentially():
    scene,rgb,_=fixture()
    rgb[20:36,20:32]=160
    assert check(scene,rgb,1)['status']=='unchanged'
    rgb[40:56,70:82]=160
    assert check(scene,rgb,2)['status']=='unchanged'
    assert not scene.boxes


def test_unexpected_change_elsewhere_is_not_swallowed_by_expected_removal():
    scene,rgb,_=fixture()
    rgb[20:36,20:32]=160
    rgb[80:90,110:120]=20
    assert check(scene,rgb,1)['status']=='changed'
    assert 1 in scene.boxes  # No patch accepted on failed scene validation.


def test_accepted_object_still_present_is_uncertain():
    scene,rgb,_=fixture()
    assert check(scene,rgb,1)['reason']=='expected_removal_not_visible'


def test_substituted_foreground_in_removed_patch_is_not_accepted():
    scene,rgb,_=fixture()
    rgb[20:36,20:32]=240
    assert check(scene,rgb,1)['status']=='uncertain'


def test_overlapping_expected_patch_never_hides_neighbor():
    scene,rgb,_=fixture()
    scene.boxes[2]=[31,20,40,36]
    rgb[20:36,20:32]=160
    assert check(scene,rgb,1)['reason']=='expected_patch_overlaps_other_original_object'


def test_reject_return_unchanged_is_retained_but_displaced_return_resets():
    scene,rgb,_=fixture()
    assert check(scene,rgb)['status']=='unchanged'
    rgb[20:36,20:32]=160
    rgb[20:36,36:48]=20
    assert check(scene,rgb)['status']=='changed'


def test_unstable_pair_is_uncertain():
    scene,rgb,_=fixture()
    moved=rgb.copy();moved[80:100,110:130]=20  # 400 px: more than the calibrated camera noise
    assert scene.check(Image.fromarray(rgb),Image.fromarray(moved))['reason']=='home_view_not_stable'


def test_large_light_change_uncertain_small_uniform_change_tolerated():
    scene,rgb,_=fixture()
    assert check(scene,rgb+10)['status']=='unchanged'
    assert check(scene,rgb+20)['status']=='uncertain'


def test_unrelated_reference_is_not_updated_allowing_gradual_changes_to_disappear():
    scene,rgb,_=fixture()
    rgb[80:90,110:120]=140
    assert check(scene,rgb)['status']=='unchanged'
    rgb[80:90,110:120]=100  # 60 levels: past the calibrated difference threshold
    assert check(scene,rgb)['status']=='changed'


def test_invalid_dimensions_and_unknown_expected_id_are_uncertain():
    scene,rgb,_=fixture()
    assert scene.check(Image.new('RGB',(10,10)),Image.new('RGB',(10,10)))['status']=='uncertain'
    assert check(scene,rgb,99)['status']=='uncertain'


def test_segmentation_failure_is_uncertainty_not_silent_accept(monkeypatch):
    scene,rgb,_=fixture();rgb[20:36,20:32]=160
    def fail(*args):raise ValueError('ambiguous foreground')
    monkeypatch.setattr('robot.home_scene.automatic_crops',fail)
    assert check(scene,rgb,1)['reason']=='post_pick_segmentation_uncertain'


@pytest.mark.parametrize('key,value',[('difference_threshold',float('nan')),('table_ring_px',0),
                                      ('max_changed_pixels',0),('min_removal_fraction',0),
                                      ('expected_margin_px',-1)])
def test_bad_settings(key,value):
    scene,rgb,settings=fixture()
    settings=copy.deepcopy(settings);settings[key]=value
    with pytest.raises(ValueError):
        HomeScene(Image.fromarray(rgb),scene.foreground,[],{},settings)
