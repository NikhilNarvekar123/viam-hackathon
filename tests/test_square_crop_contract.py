import numpy as np
from PIL import Image,ImageDraw
from anomaly.object_crops import automatic_crops,extract_candidate_crops,square_cutout
from anomaly.preprocess import prepare_image


def cfg():
    return dict(reference_size_px=[640,480],background_blur_radius_px=20,difference_threshold=20,
        min_area_px=80,max_area_px=12000,opening_radius_px=0,closing_radius_px=1,crop_padding_px=8,
        orientation_min_ratio=2,max_foreground_fraction=.4,workspace_roi=None,crop_padding_factor=1.35,
        min_crop_side_px=32,neighbor_mask_margin_px=1,neighbor_feather_px=1)


def test_resolution_invariance():
    image=Image.new('RGB',(640,480),(150,150,150));draw=ImageDraw.Draw(image)
    draw.rectangle((170,150,190,210),fill=(30,30,30));draw.ellipse((400,300,430,330),fill=(20,20,20))
    rois,a,mask=automatic_crops(image,cfg())
    large=image.resize((1280,960),Image.Resampling.NEAREST)
    rois2,b,mask2=automatic_crops(large,cfg())
    assert len(rois)==len(rois2)==2
    for x,y in zip(a['objects'],b['objects']):
        np.testing.assert_allclose(np.array(x['center_uv'])*2,y['center_uv'],atol=2)
        np.testing.assert_allclose(np.array(x['crop_bbox_xyxy'])*2,y['crop_bbox_xyxy'],atol=4)
    crops=extract_candidate_crops(image,a,mask);crops2=extract_candidate_crops(large,b,mask2)
    for x,y in zip(crops,crops2):
        assert x.width==x.height and y.width==y.height
        # Small resampling differences allowed, no resolution-dependent scale jump.
        assert abs(x.width*2-y.width)<=4


def test_square_oob_padding_uses_table_not_black():
    image=Image.new('RGB',(20,20),(120,130,140))
    crop=square_cutout(image,[-10,-5,30,35],[120,130,140])
    assert crop.size==(40,40)
    np.testing.assert_array_equal(np.asarray(crop)[0,0],[120,130,140])


def test_neighbor_contour_mask_does_not_erase_target_or_rectangle():
    image=Image.new('RGB',(100,100),(150,150,150));d=ImageDraw.Draw(image)
    d.ellipse((20,40,30,50),fill='black');d.ellipse((45,40,55,50),fill='red')
    mask_image=Image.new('L',image.size);md=ImageDraw.Draw(mask_image)
    md.ellipse((20,40,30,50),fill=255);md.ellipse((45,40,55,50),fill=255)
    result={'objects':[{'center_uv':[25,45],'crop_bbox_xyxy':[0,15,60,75]}],
            'neighbor_mask_margin_px':1,'neighbor_feather_px':1}
    crop=extract_candidate_crops(image,result,np.asarray(mask_image)>0)[0]
    assert crop.getpixel((25,30))==(0,0,0)
    assert max(abs(v-150) for v in crop.getpixel((50,30)))<3
    assert crop.getpixel((44,24))==(150,150,150)
    assert result['objects'][0]['neighbor_pixels_masked']>0


def test_disconnected_mark_inside_target_is_preserved():
    image=Image.new('RGB',(100,100),(150,150,150));draw=ImageDraw.Draw(image)
    draw.ellipse((30,30,70,70),outline='black',width=4)
    draw.rectangle((47,47,53,53),fill='red')
    mask=np.any(np.asarray(image)!=150,axis=2)
    result={'objects':[{'center_uv':[50,50],'bbox_xyxy':[30,30,71,71],
                       'crop_bbox_xyxy':[10,10,90,90]}],
            'neighbor_mask_margin_px':1,'neighbor_feather_px':1}
    crop=extract_candidate_crops(image,result,mask)[0]
    assert crop.getpixel((40,40))==(255,0,0)


def test_neighbor_holes_and_four_pixel_halo_filled_without_erasing_target():
    image=Image.new('RGB',(100,100),(150,150,150));draw=ImageDraw.Draw(image)
    draw.rectangle((20,40,30,50),fill='red')
    draw.ellipse((45,40,65,60),outline='black',width=3)
    draw.point((55,50),fill='blue')  # Inner pixels must not survive ring masking.
    mask=np.any(np.asarray(image)!=150,axis=2)
    result={'objects':[
        {'center_uv':[25,45],'bbox_xyxy':[20,40,31,51],'crop_bbox_xyxy':[0,10,80,90]},
        {'center_uv':[55,50],'bbox_xyxy':[45,40,66,61],'crop_bbox_xyxy':[15,10,95,90]}],
        'neighbor_mask_margin_px':4,'neighbor_feather_px':0}
    crop=extract_candidate_crops(image,result,mask)[0]
    assert crop.getpixel((25,35))==(255,0,0)
    assert crop.getpixel((55,40))==(150,150,150)
    assert crop.getpixel((45,40))==(150,150,150)
    assert crop.getpixel((41,40))==(150,150,150)


def test_proportional_window_matches_requested_formula():
    image=Image.new('RGB',(640,480),(150,150,150));draw=ImageDraw.Draw(image)
    draw.rectangle((150,150,169,179),fill='black')
    settings=dict(cfg(),crop_padding_factor=1.45,min_crop_side_px=36)
    for scale in [1,2]:
        scaled=image.resize((640*scale,480*scale),Image.Resampling.NEAREST)
        _,result,mask=automatic_crops(scaled,settings)
        for obj in result['objects']:
            l,t,r,b=obj['bbox_xyxy']
            expected=max(int(36*scale),int(max(r-l,b-t)*1.45))
            assert obj['crop_side_px']==expected
            crop=extract_candidate_crops(scaled,result,mask)[0]
            assert crop.size==(expected,expected)


def test_extraction_refuses_artificial_border_padding():
    import pytest
    image=Image.new('RGB',(30,30),'gray');mask=np.zeros((30,30),bool);mask[1:5,1:5]=True
    result={'objects':[{'center_uv':[3,3],'crop_bbox_xyxy':[-15,-15,21,21]}],
            'neighbor_mask_margin_px':4,'neighbor_feather_px':0}
    with pytest.raises(ValueError,match='crosses frame boundary'):
        extract_candidate_crops(image,result,mask)


def test_manual_nonsquare_input_is_not_stretched():
    image=Image.new('RGB',(80,20),(150,150,150));d=ImageDraw.Draw(image)
    d.rectangle((30,2,45,17),fill='black')
    out=prepare_image(image,80)
    assert out.size==(80,80)
    pixels=np.asarray(out);yy,xx=np.where(pixels[:,:,0]<50)
    assert xx.max()-xx.min()==yy.max()-yy.min()==15
    assert out.getpixel((0,0))==(150,150,150)
