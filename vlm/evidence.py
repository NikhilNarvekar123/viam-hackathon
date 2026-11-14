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
