"""Locate a supplied ROI directly from raw depth/table geometry, not detector proposals.

ROI geometry seeds an approach only. Close-up geometry must satisfy the original
picker's size/height/grasp limits before descent. No classifier is called here.
"""
import math
import numpy as np
import pick_bolt as picker


def measure_roi(crop, foreground, info, K, settings):
    l,t,r,b=map(int,crop['bbox_xyxy'])
    if not (0<=l<r<=K.width and 0<=t<b<=K.height):raise ValueError('ROI outside camera frame')
    mask=np.zeros((K.height,K.width),bool);mask[t:b,l:r]=foreground[t:b,l:r]
    ys,xs=np.where(mask)
    if len(xs)<settings['roi_min_valid_pixels']:raise ValueError('Insufficient ROI foreground')
    valid=mask & np.isfinite(info['depth']) & (info['depth']>0) & np.isfinite(info['height'])
    count=int(valid.sum())
    if count<settings['roi_min_valid_pixels'] or count/len(xs)<settings['roi_min_valid_fraction']:
        raise ValueError('Insufficient valid depth inside supplied ROI')
    heights=info['height'][valid]
    height=max(0.,float(np.percentile(heights,settings['roi_height_percentile'])))
    ray_height=max(0.,float(np.median(heights)))
    d=float(info['cam_height_mm']);n=-np.asarray(info['foot'],float)/d
    if not math.isfinite(d) or d<=0 or not np.isfinite(n).all():raise ValueError('Invalid table plane')
    def on_plane(u,v):
        ray=np.array([(u-K.cx)/K.fx,(v-K.cy)/K.fy,1.])
        denominator=float(n@ray)
        if abs(denominator)<1e-6:raise ValueError('ROI ray parallel to table')
        surface=ray*((ray_height-d)/denominator)
        if not np.isfinite(surface).all() or surface[2]<=0:raise ValueError('Invalid ROI depth projection')
        return surface-n*ray_height,surface
    u,v=float(xs.mean()),float(ys.mean())
    centered=np.stack([xs-u,ys-v])
    axis=np.linalg.eigh(np.cov(centered))[1][:,1]
    along=centered.T@axis;across=centered.T@np.array([-axis[1],axis[0]])
    table,surface=on_plane(u,v);p_axis,_=on_plane(u+25*axis[0],v+25*axis[1])
    scale=surface[2]/K.fx
    obj=picker.Obj(int(crop['id']),'object',1.,u,v,float((np.ptp(along)+1)*scale),
                   float((np.ptp(across)+1)*scale),height,float(len(xs)*scale*scale),
                   table,p_axis,ys,xs,'external_roi')
    corners=[on_plane(x,y)[1] for x,y in [(l,t),(r-1,t),(r-1,b-1),(l,b-1)]]
    return obj,np.asarray(corners),dict(valid_depth_pixels=count,foreground_pixels=len(xs),
                                      median_height_mm=ray_height,height_mm=height)


def bind_rois(crops,foreground,info,K,rotation,translation,settings):
    R=np.asarray(rotation,float);translation=np.asarray(translation,float)
    rows=[]
    for crop in crops:
        row=dict(crop_id=crop['id'],bbox_xyxy=crop['bbox_xyxy'],status='unmatched',source='supplied_roi')
        try:
            obj,corners,stats=measure_roi(crop,foreground,info,K,settings)
            world=R@obj.p_table+translation;axis=R@obj.p_axis+translation
            row.update(status='matched',table_point_world_mm=world.tolist(),
                       surface_corners_world_mm=(corners@R.T+translation).tolist(),
                       axis_world_deg=math.degrees(math.atan2(axis[1]-world[1],axis[0]-world[0])),
                       length_mm=obj.length_mm,width_mm=obj.width_mm,**stats)
        except ValueError as exc:row['reason']=str(exc)
        rows.append(row)
    return rows


def target_in_view(binding,info,K,settings):
    """Reproject the supplied object region into the current stationary camera view."""
    corners=np.asarray(binding['surface_corners_world_mm'],float)
    R=np.asarray(info['R']);translation=np.asarray(info['t'])
    camera=(corners-translation)@R
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=0):raise ValueError('Target behind camera')
    uv=np.column_stack([K.fx*camera[:,0]/camera[:,2]+K.cx,K.fy*camera[:,1]/camera[:,2]+K.cy])
    margin=settings['roi_projection_margin_px']*K.width/640
    l,t=np.floor(uv.min(axis=0)-margin).astype(int);r,b=np.ceil(uv.max(axis=0)+margin+1).astype(int)
    if not (0<=l<r<=K.width and 0<=t<b<=K.height):raise ValueError('Projected target ROI outside frame')
    bgr=info.get('bgr')
    if bgr is None or picker.cv2 is None:raise ValueError('Current RGB image is required for ROI refinement')
    cv=picker.cv2
    gray=cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    sat=cv.cvtColor(bgr,cv.COLOR_BGR2HSV)[...,1]
    foreground=(cv.absdiff(gray,cv.medianBlur(gray,51))>settings['roi_color_difference']) | \
               (sat.astype(int)-cv.medianBlur(sat,51).astype(int)>settings['roi_saturation_difference'])
    # Only RGB evidence INSIDE the projected supplied region. No global proposal matching.
    obj,_,_=measure_roi(dict(id=binding['crop_id'],bbox_xyxy=[l,t,r,b]),foreground,info,K,settings)
    return obj


def validate_approach_target(obj):
    if obj.height_mm is None or not np.isfinite([obj.height_mm,obj.length_mm,obj.width_mm,obj.area_mm2]).all():
        return 'ROI has invalid measured geometry'
    if obj.height_mm>picker.BOLT_MAX_H or obj.length_mm>picker.MAX_OBJECT_LEN or obj.area_mm2>picker.MAX_AREA_MM2:
        return 'ROI exceeds configured object size/height limits'
    if obj.area_mm2<picker.MIN_AREA_MM2:return 'ROI too small'
    return None


def validate_grasp_target(obj):
    """Low initial ROIs may approach; close-up must satisfy physical grasp limits."""
    refusal=validate_approach_target(obj)
    if refusal:return refusal
    if obj.height_mm<picker.GRIP_MIN_H_MM:
        return 'Close-up ROI is too low for the configured gripper height limit'
    return None
