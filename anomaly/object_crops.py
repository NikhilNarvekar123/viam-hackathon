"""Resolution-normalized tabletop segmentation and square, neighbor-masked crops.

This is heuristic segmentation, not a trained instance detector. Touching parts
can merge. Framing settings must be validated against the actual training dataset.
"""
import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from robot.locate import locate_objects, components, morphology
from anomaly.enhance import sharpen_crop


def automatic_crops(image, settings):
    radius = settings['background_blur_radius_px']
    if not isinstance(radius, (int,float)) or not math.isfinite(radius) or not 1<=radius<=200:
        raise ValueError('background_blur_radius_px must be between 1 and 200')
    maximum = settings['max_area_px']
    if not isinstance(maximum,int) or maximum<settings['min_area_px']:
        raise ValueError('max_area_px must be >= min_area_px')
    reference_size=settings.get('reference_size_px',[640,480])
    if len(reference_size)!=2 or any(type(x) is not int or x<1 for x in reference_size):
        raise ValueError('reference_size_px must contain two positive integers')
    sx,sy=image.width/reference_size[0],image.height/reference_size[1]
    if not math.isclose(sx,sy,rel_tol=0.01):
        raise ValueError('Camera aspect ratio differs from cropper reference; configure matching reference_size_px')
    # All segmentation thresholds/morphology operate on one reference resolution.
    canonical=image.convert('RGB').resize(tuple(reference_size),Image.Resampling.BILINEAR)
    reference=canonical.filter(ImageFilter.GaussianBlur(radius))
    result,mask=locate_objects(canonical,reference,settings)
    padding_factor=float(settings.get('crop_padding_factor',1.35))
    minimum=float(settings.get('min_crop_side_px',32))
    if not math.isfinite(padding_factor) or not 1<=padding_factor<=5 or not math.isfinite(minimum) or minimum<1:
        raise ValueError('Invalid square crop framing settings')
    edge_margin=float(settings.get('frame_edge_margin_px',0))
    if not math.isfinite(edge_margin) or not 0<=edge_margin<min(reference_size)/2:
        raise ValueError('Invalid frame edge margin')
    min_texture=float(settings.get('min_laplacian_variance',0))
    if not math.isfinite(min_texture) or min_texture<0:
        raise ValueError('Invalid minimum Laplacian variance')
    gray=np.asarray(canonical.convert('L'),dtype=np.float32)
    padded=np.pad(gray,1,mode='edge')
    laplacian=4*gray-padded[:-2,1:-1]-padded[2:,1:-1]-padded[1:-1,:-2]-padded[1:-1,2:]
    accepted,rejected=[],[]
    for obj in result['objects']:
        l,t,r,b=obj['bbox_xyxy']
        lap=laplacian[t:b,l:r]
        foreground=mask[t:b,l:r]
        texture=float(np.var(lap[foreground])) if lap.size and foreground.any() else 0.0
        obj['laplacian_variance']=texture
        if texture<min_texture:
            obj['warnings'].append('insufficient_foreground_structure')
        if min(l,t,reference_size[0]-r,reference_size[1]-b)<=edge_margin:
            obj['warnings'].append('frame_edge_margin')
        if obj['area_px']>maximum or obj['warnings']:
            rejected.append(obj)
        else:
            accepted.append(obj)
    # Conservative duplicate suppression in canonical pixels, never by padded crop overlap.
    distance=float(settings.get('dedup_center_distance_px',3))
    iou_limit=float(settings.get('dedup_iou_threshold',0.8))
    if not math.isfinite(distance) or distance<0 or not 0<iou_limit<=1:
        raise ValueError('Invalid deduplication settings')
    fragment_coverage=float(settings.get('fragment_min_bbox_coverage',0.75))
    fragment_ratio=float(settings.get('fragment_max_area_ratio',0.25))
    fragment_angle=float(settings.get('fragment_max_angle_difference_deg',15))
    if not 0<fragment_coverage<=1 or not 0<fragment_ratio<1 or not 0<=fragment_angle<=90:
        raise ValueError('Invalid fragment suppression settings')
    unique=[]
    for obj in sorted(accepted,key=lambda o:-o['area_px']):
        duplicate=False
        for previous in unique:
            a,b=obj['bbox_xyxy'],previous['bbox_xyxy']
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
            if math.dist(obj['center_uv'],previous['center_uv'])<=distance and overlap/union>=iou_limit:
                obj['warnings'].append('near_duplicate');duplicate=True;break
            angle,other=obj['image_angle_deg'],previous['image_angle_deg']
            aligned=(angle is not None and other is not None and
                     min(abs(angle-other),180-abs(angle-other))<=fragment_angle)
            coverage=overlap/((a[2]-a[0])*(a[3]-a[1]))
            if aligned and coverage>=fragment_coverage and obj['area_px']/previous['area_px']<=fragment_ratio:
                obj['warnings'].append('likely_fragment_of_larger_candidate');duplicate=True;break
        if duplicate:
            rejected.append(obj)
        else: unique.append(obj)
    accepted=sorted(unique,key=lambda o:(o['center_uv'][1],o['center_uv'][0]))
    for obj in accepted+rejected:
        obj['area_reference_px']=obj['area_px']
        obj['area_px']=int(round(obj['area_px']*sx*sy))
        obj['center_uv']=[obj['center_uv'][0]*sx,obj['center_uv'][1]*sy]
        obj['bbox_xyxy']=[int(round(v*(sx if i%2==0 else sy))) for i,v in enumerate(obj['bbox_xyxy'])]
        cx,cy=obj['center_uv'];l,t,r,b=obj['bbox_xyxy']
        side=max(int(minimum*sx),int(max(r-l,b-t)*padding_factor))
        left=int(math.floor(cx-side/2));top=int(math.floor(cy-side/2))
        # Keep exact centered geometry. Extraction refuses out-of-frame windows.
        obj['crop_bbox_xyxy']=[left,top,left+side,top+side]
        obj['crop_side_px']=side
        obj['crop_padding_outside_frame']=bool(left<0 or top<0 or left+side>image.width or top+side>image.height)
    # A crop that would leave the frame is never padded (extraction refuses it): skip that part this batch
    # instead of aborting the whole batch. It is picked once it has been moved, or from another view.
    for obj in [o for o in accepted if o['crop_padding_outside_frame']]:
        obj['warnings'].append('crop_crosses_frame');accepted.remove(obj);rejected.append(obj)
    for index,obj in enumerate(accepted,1): obj['id']=index
    result['objects']=accepted
    result['rejected_candidates']=rejected
    result['workspace_roi']=[int(round(v*(sx if i%2==0 else sy))) for i,v in enumerate(result['workspace_roi'])]
    result['reference_size_px']=reference_size
    result['crop_sharpening']=settings.get('crop_sharpening',{'enabled':False})
    result['neighbor_mask_margin_px']=max(0,int(round(settings.get('neighbor_mask_margin_px',1)*sx)))
    result['neighbor_feather_px']=float(settings.get('neighbor_feather_px',1))*sx
    result['method']='Gaussian local contrast at reference resolution; centroid-centered square crops; contour-only neighbor removal'
    native_mask=np.asarray(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST),dtype=bool)
    return [o['crop_bbox_xyxy'] for o in accepted],result,native_mask


def square_cutout(image, box, background_color):
    l,t,r,b=box
    if r-l!=b-t or r<=l: raise ValueError('Expected nonempty square crop')
    out=Image.new('RGB',(r-l,b-t),tuple(int(x) for x in background_color))
    bounds=(max(0,l),max(0,t),min(image.width,r),min(image.height,b))
    if bounds[2]>bounds[0] and bounds[3]>bounds[1]:
        out.paste(image.crop(bounds),(bounds[0]-l,bounds[1]-t))
    return out


def filled_contour(mask):
    """Fill enclosed holes without convex-hull/rectangle erasure of concavities."""
    padded=Image.fromarray(np.pad(mask.astype(np.uint8)*255,1)).copy()
    ImageDraw.floodfill(padded,(0,0),128)
    return (np.asarray(padded)[1:-1,1:-1]!=128)


def extract_candidate_crops(image, result, mask, *, object_indices=None):
    """Extract requested targets, retaining ALL candidates for neighbor masking.

    By default extract every candidate. Selected extraction still refuses any
    requested out-of-frame window; unrelated boundary crops need not be extracted.
    """
    requested=result['objects']
    if object_indices is not None:
        indices=list(object_indices)
        if any(type(i) is not int or not 0<=i<len(requested) for i in indices):
            raise ValueError('Invalid candidate crop index')
        requested=[requested[i] for i in indices]
    image=image.convert('RGB');pixels=np.asarray(image)
    labels=np.zeros(mask.shape,dtype=np.int32)
    centers=[]
    for index,points in enumerate(components(mask),1):
        labels[points[:,0],points[:,1]]=index
        centers.append((index,float(points[:,1].mean()),float(points[:,0].mean()),
                        [int(points[:,1].min()),int(points[:,0].min()),
                         int(points[:,1].max()+1),int(points[:,0].max()+1)]))
    def target_label(obj):
        bounds=obj.get('bbox_xyxy');cx,cy=obj['center_uv']
        if not centers: raise ValueError('Missing target foreground component')
        return min(centers,key=lambda q: sum(abs(a-b) for a,b in zip(q[3],bounds))
                   if bounds else (q[1]-cx)**2+(q[2]-cy)**2)[0]
    detected_labels={target_label(obj) for obj in result['objects']}
    filled={label:filled_contour(labels==label) for label in detected_labels}
    crops=[]
    for obj in requested:
        l,t,r,b=obj['crop_bbox_xyxy'];cx,cy=obj['center_uv']
        if l<0 or t<0 or r>image.width or b>image.height:
            raise ValueError('Centered inspection crop crosses frame boundary; reposition target. No padded inference permitted.')
        if not centers: raise ValueError('Missing target foreground component')
        bounds=obj.get('bbox_xyxy')
        label=min(centers,key=lambda q: sum(abs(a-b) for a,b in zip(q[3],bounds))
                  if bounds else (q[1]-cx)**2+(q[2]-cy)**2)[0]
        target=filled[label]
        # Table color comes from local nonforeground pixels, with a small exclusion halo.
        halo=morphology(mask, min(5,max(1,result['neighbor_mask_margin_px']+1)),True)
        cl,ct,cr,cb=max(0,l),max(0,t),min(image.width,r),min(image.height,b)
        local=pixels[ct:cb,cl:cr];table=local[~halo[ct:cb,cl:cr]]
        if not len(table):
            table=pixels[~halo]
        if not len(table): raise ValueError('No tabletop pixels available to estimate crop background')
        color=np.median(table,axis=0).astype(np.uint8)
        # A disconnected mark/reflection INSIDE the target's bounding box may be
        # part of the target, not a neighbor. Preserve whole ambiguous components.
        bounds=obj.get('bbox_xyxy')
        protected_labels={label}
        if bounds:
            x0,y0,x1,y1=bounds
            protected_labels.update(int(v) for v in np.unique(labels[y0:y1,x0:x1])
                                    if v and (int(v) not in detected_labels or int(v)==label))
        protected=target|np.isin(labels,list(protected_labels))
        neighbor=(labels!=0)&~protected
        for other in detected_labels-{label}:
            neighbor|=filled[other]
        margin=result['neighbor_mask_margin_px']
        neighbor=morphology(neighbor,margin,True)&~protected
        alpha=Image.fromarray(neighbor.astype(np.uint8)*255)
        feather=result['neighbor_feather_px']
        if feather>0: alpha=alpha.filter(ImageFilter.GaussianBlur(feather))
        # Never alter target foreground pixels when feathering neighboring contours.
        a=np.asarray(alpha).copy();a[protected]=0
        cleaned=Image.composite(Image.new('RGB',image.size,tuple(map(int,color))),image,Image.fromarray(a))
        crop=square_cutout(cleaned,obj['crop_bbox_xyxy'],color)
        obj['background_color_rgb']=color.tolist()
        obj['neighbor_pixels_masked']=int(neighbor[ct:cb,cl:cr].sum())
        crop=sharpen_crop(crop,result.get('crop_sharpening',{'enabled':False}))
        obj['crop_sharpening']=result.get('crop_sharpening',{'enabled':False})
        crops.append(crop)
    return crops
