"""Color-agnostic, noise-gated unsharp masking for extracted object crops."""
import math
import numpy as np
from PIL import Image


def sharpen_crop(image, settings):
    image=image.convert('RGB')
    if not settings.get('enabled',False): return image.copy()
    sigma=float(settings['sigma']);amount=float(settings['amount'])
    noise_floor=float(settings.get('noise_floor',2.0))
    if (not all(math.isfinite(v) for v in (sigma,amount,noise_floor)) or
            sigma<=0 or not 0<=amount<=1 or not 0<=noise_floor<=255):
        raise ValueError('Invalid crop sharpening parameters')
    import cv2
    pixels=np.asarray(image,dtype=np.float32)
    luminance=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    blurred=cv2.GaussianBlur(luminance,(0,0),sigmaX=sigma,sigmaY=sigma,
                             borderType=cv2.BORDER_REFLECT_101)
    detail=luminance-blurred
    # Low-contrast differences are not boosted. Apply luminance detail equally
    # to all RGB channels instead of independently exaggerating chromatic noise.
    detail=np.where(np.abs(detail)>=noise_floor,detail,0.0)
    enhanced=np.clip(np.rint(pixels+amount*detail[:,:,None]),0,255).astype(np.uint8)
    return Image.fromarray(enhanced)
