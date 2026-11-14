import numpy as np
import pytest
from PIL import Image
from anomaly.enhance import sharpen_crop
from vlm.evidence import evidence

CFG={'enabled':True,'sigma':1.2,'amount':.35,'noise_floor':2.0}


def test_uniform_rgb_and_low_amplitude_noise_not_amplified():
    image=Image.new('RGB',(36,36),(80,130,170))
    assert sharpen_crop(image,CFG).tobytes()==image.tobytes()
    noisy=np.full((36,36,3),128,dtype=np.uint8);noisy[::2,::2]=129
    assert sharpen_crop(Image.fromarray(noisy),CFG).tobytes()==Image.fromarray(noisy).tobytes()


def test_sharpen_matches_requested_noise_gated_gaussian_formula():
    import cv2
    pixels=np.full((36,36,3),80,dtype=np.uint8);pixels[:,18:]=160
    image=Image.fromarray(pixels);before=image.tobytes()
    gray=cv2.cvtColor(pixels.astype(np.float32),cv2.COLOR_RGB2GRAY)
    detail=gray-cv2.GaussianBlur(gray,(0,0),sigmaX=1.2,sigmaY=1.2,borderType=cv2.BORDER_REFLECT_101)
    detail=np.where(np.abs(detail)>=2,detail,0)
    expected=np.clip(np.rint(pixels.astype(np.float32)+.35*detail[:,:,None]),0,255).astype(np.uint8)
    actual=np.asarray(sharpen_crop(image,CFG))
    np.testing.assert_array_equal(actual,expected)
    assert image.tobytes()==before
    assert actual[18,17,0]<80 and actual[18,18,0]>160


@pytest.mark.parametrize('key,value',[('sigma',0),('amount',2),('noise_floor',-1),('sigma',float('nan'))])
def test_invalid_sharpening_rejected(key,value):
    with pytest.raises(ValueError):sharpen_crop(Image.new('RGB',(36,36)),dict(CFG,**{key:value}))


def test_vlm_resize_is_exact_opencv_lanczos4_rgb():
    import cv2
    pixels=np.random.default_rng(7).integers(0,256,(36,36,3),dtype=np.uint8)
    actual,_=evidence(Image.fromarray(pixels),None,{},'Inspect.',224)
    expected=cv2.resize(pixels,(224,224),interpolation=cv2.INTER_LANCZOS4)
    np.testing.assert_array_equal(np.asarray(actual),expected)
