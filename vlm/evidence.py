"""Native-resolution, single-target VLM evidence; never model overlays."""
import math
from PIL import Image
import numpy as np


def stage1_pass(anomaly, settings):
    if not settings.get('stage1_fast_pass',False): return False
    score,threshold=anomaly.get('score'),anomaly.get('threshold')
    return (anomaly.get('classification')=='normal' and
            isinstance(score,(int,float)) and isinstance(threshold,(int,float)) and
            math.isfinite(score) and math.isfinite(threshold) and score<threshold)

def raw_crop(frame, anomaly):
    """Slice the unmodified capture; clip bounds without resize or synthetic padding."""
    l,t,r,b=anomaly.get('crop_bbox_xyxy',anomaly['bbox_xyxy'])
    box=(max(0,int(l)),max(0,int(t)),min(frame.width,int(r)),min(frame.height,int(b)))
    if box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError('Empty VLM crop')
    return frame.convert('RGB').crop(box)


def evidence(crop, overlay, anomaly, prompt, image_size=None):
    # Compatibility arguments explicitly ignored: no scores/maps/colors reach VLM.
    question=(prompt+'\nThe supplied image is the isolated square crop of the single centered target. '
              'Neighbor contours were replaced with tabletop background. It contains no annotation or heatmap. '
              'Inspect only the physical target, not the filled background.')
    image=crop.copy().convert('RGB')
    if image_size is not None:
        if type(image_size) is not int or image_size<1 or image.width!=image.height:
            raise ValueError('VLM resize requires a square crop and positive image size')
        import cv2
        # Resizing is channel-agnostic: keep RGB order, no BGR conversion.
        pixels=cv2.resize(np.asarray(image),(image_size,image_size),interpolation=cv2.INTER_LANCZOS4)
        image=Image.fromarray(pixels)
    return image,question


def colored_pixels(crop, min_saturation, hue_range=None):
    """Strongly colored pixels in a crop (tape, paint, colored marker). Bare metal stays near gray even under the
    wrist camera's teal tint: on 14 Nov taped parts had 370-1100 px above 0.4 saturation, clean parts 3 at most."""
    hsv=np.asarray(crop.convert('RGB').convert('HSV'),dtype=np.float32)/255.0
    hit=(hsv[...,1]>min_saturation)&(hsv[...,2]>0.15)
    if hue_range is not None:  # hue as a fraction of the color wheel: blue tape reads about 0.58
        hit&=(hsv[...,0]>=hue_range[0])&(hsv[...,0]<=hue_range[1])
    return int(hit.sum())
