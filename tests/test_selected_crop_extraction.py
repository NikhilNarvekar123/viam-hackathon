import numpy as np
import pytest
from PIL import Image
from anomaly.object_crops import extract_candidate_crops


def scene():
    pixels=np.full((100,100,3),180,dtype=np.uint8)
    mask=np.zeros((100,100),dtype=bool)
    mask[40:50,40:50]=True
    mask[55:90,55:65]=True
    pixels[mask]=20
    objects=[dict(id=1,center_uv=[45,45],bbox_xyxy=[40,40,50,50],crop_bbox_xyxy=[25,25,65,65]),
             dict(id=2,center_uv=[60,72],bbox_xyxy=[55,55,65,90],crop_bbox_xyxy=[25,37,95,107])]
    return Image.fromarray(pixels),dict(objects=objects,neighbor_mask_margin_px=0,neighbor_feather_px=0),mask


def test_valid_selected_target_ignores_unrelated_boundary_crop_but_masks_neighbor():
    image,result,mask=scene()
    crops=extract_candidate_crops(image,result,mask,object_indices=[0])
    assert len(crops)==1 and crops[0].size==(40,40)
    pixels=np.asarray(crops[0])
    assert (pixels[15:25,15:25]==20).all()  # target preserved
    assert (pixels[30:40,30:40]==180).all()  # other candidate still masked


def test_selected_boundary_target_still_refuses_padding():
    image,result,mask=scene()
    with pytest.raises(ValueError,match='crosses frame boundary'):
        extract_candidate_crops(image,result,mask,object_indices=[1])


def test_default_batch_extraction_still_refuses_boundary_crop():
    image,result,mask=scene()
    with pytest.raises(ValueError,match='crosses frame boundary'):
        extract_candidate_crops(image,result,mask)


@pytest.mark.parametrize('indices',[[-1],[2],[True]])
def test_invalid_selection_is_rejected(indices):
    image,result,mask=scene()
    with pytest.raises(ValueError,match='Invalid candidate'):
        extract_candidate_crops(image,result,mask,object_indices=indices)
