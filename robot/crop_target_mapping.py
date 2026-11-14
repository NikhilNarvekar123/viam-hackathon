"""Read-only, one-to-one association of RGB crops with same-view picker depth objects.

No image ID is interpreted as a depth ID. Coordinates use the teammate's table
projection and camera/world transform, not a new pixel-to-robot calibration.
"""
import math
import numpy as np


def bind_crops(candidates, depth_objects, foreground, rotation, translation, settings, pick_kinds):
    rotation=np.asarray(rotation,float);translation=np.asarray(translation,float)
    if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Invalid camera-to-world transform')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=.01) or not np.isclose(np.linalg.det(rotation),1,atol=.01):
        raise ValueError('Camera transform is not a rigid rotation')
    height,width=foreground.shape
    rows=[];claims={}
    for crop in candidates:
        l,t,r,b=crop['bbox_xyxy']
        row=dict(crop_id=crop['id'],bbox_xyxy=list(crop['bbox_xyxy']),status='unmatched',matches=[])
        if not 0<=l<r<=width or not 0<=t<b<=height:
            row['reason']='invalid RGB bounds';rows.append(row);continue
        rgb=np.zeros_like(foreground,dtype=bool);rgb[t:b,l:r]=foreground[t:b,l:r]
        for index,obj in enumerate(depth_objects):
            if obj.kind not in pick_kinds or obj.source!='depth' or obj.xs is None or obj.ys is None:continue
            xs=np.asarray(obj.xs);ys=np.asarray(obj.ys)
            if xs.size==0 or xs.shape!=ys.shape or not np.isfinite(obj.p_table).all():continue
            if np.any(xs<0) or np.any(xs>=width) or np.any(ys<0) or np.any(ys>=height):continue
            pixels=np.zeros_like(rgb);pixels[ys.astype(int),xs.astype(int)]=True
            overlap=float((rgb&pixels).sum()/max(1,min(rgb.sum(),pixels.sum())))
            distance=math.dist(crop['center_uv'],[obj.u,obj.v])
            if overlap>=settings['min_foreground_overlap'] and distance<=settings['max_center_distance_px']*width/640:
                row['matches'].append(dict(depth_index=index,depth_id=int(obj.idx),overlap=overlap,center_distance_px=distance))
        for match in row['matches']:
            claims.setdefault(match['depth_index'],[]).append(len(rows))
        if len(row['matches'])!=1:
            row['reason']='no compatible depth target' if not row['matches'] else 'multiple compatible depth targets'
        rows.append(row)
    for index,owners in claims.items():
        for owner in owners:
            row=rows[owner]
            if len(owners)!=1:
                row['reason']='depth target claimed by multiple RGB crops';continue
            if len(row['matches'])!=1:continue
            obj=depth_objects[index]
            world=rotation@obj.p_table+translation
            axis=rotation@obj.p_axis+translation
            if not np.isfinite(world).all() or not np.isfinite(axis).all():
                row['reason']='nonfinite world geometry';continue
            row.update(status='matched',depth_id=int(obj.idx),depth_index=index,kind=obj.kind,
                       table_point_world_mm=world.tolist(),
                       axis_world_deg=math.degrees(math.atan2(axis[1]-world[1],axis[0]-world[0])),
                       depth_bbox_xyxy=[int(min(obj.xs)),int(min(obj.ys)),int(max(obj.xs))+1,int(max(obj.ys))+1])
    return rows
