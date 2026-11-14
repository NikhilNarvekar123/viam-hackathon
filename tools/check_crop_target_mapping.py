"""Legacy proposal-matching audit (main now localizes supplied ROIs directly).
No homing, grasping, inference, or Viam config changes.

RGB and point clouds are acquired sequentially with the arm stationary, not claimed
hardware-synchronized. --replay reprocesses an audit folder without any connection.
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
import yaml
import pick_bolt as picker
from anomaly.object_crops import automatic_crops
from robot.crop_target_mapping import bind_crops
from robot.inspection_pick import IntegratedStation
from robot.viam_hw import load_credentials


def analyze(folder,settings,crop_settings):
    meta=json.loads((folder/'capture.json').read_text())
    image=Image.open(folder/'frame.png').convert('RGB')
    depth=np.load(folder/'depth.npy',allow_pickle=False)
    gray=np.load(folder/'gray.npy',allow_pickle=False) if (folder/'gray.npy').exists() else None
    K=picker.Intrinsics(**meta['intrinsics'])
    if image.size!=(K.width,K.height) or depth.shape!=(K.height,K.width):
        raise ValueError('RGB/depth/intrinsics dimensions differ')
    if meta['in_frame']<.6:raise ValueError('Point cloud alignment check failed')
    objects,_=picker.detect_depth(depth,K,np.asarray(image)[:,:,::-1].copy(),gray)
    _,segmentation,mask=automatic_crops(image,crop_settings)
    rows=bind_crops(segmentation['objects'],objects,mask,meta['rotation'],meta['translation_mm'],settings,picker.PICK_KINDS)
    report=dict(capture=meta,association_settings=settings,objects=rows,actions_executed=False,
                physically_validated=False,matched=sum(r['status']=='matched' for r in rows),total=len(rows))
    (folder/'mapping.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    overlay=image.copy();draw=ImageDraw.Draw(overlay)
    for row in rows:
        good=row['status']=='matched';color='lime' if good else 'red'
        draw.rectangle(row['bbox_xyxy'],outline=color,width=2)
        l,t,_,_=row['bbox_xyxy'];text=f"RGB {row['crop_id']}"
        if good:
            obj=objects[row['depth_index']]
            for x,y in zip(obj.xs[::3],obj.ys[::3]):draw.point((int(x),int(y)),fill='cyan')
            draw.rectangle(row['depth_bbox_xyxy'],outline='orange',width=1)
            x,y,z=row['table_point_world_mm'];text+=f" -> depth {row['depth_id']}: {x:.0f},{y:.0f},{z:.0f} mm"
        else:text+=' UNMATCHED'
        draw.text((l,max(0,t-12)),text,fill=color,stroke_width=1,stroke_fill='black')
    overlay.save(folder/'mapping_overlay.png')
    print(json.dumps(dict(output=str(folder),matched=report['matched'],total=report['total'],objects=rows),indent=2))
    return report


async def capture(folder,config,settings):
    credentials=load_credentials(ROOT/'.env')
    args=picker.parse_args([]);args.address=credentials['VIAM_ADDRESS'];args.anomaly_service=None
    args.record_hz=0;args.debug_dir=str(folder)
    st=IntegratedStation(args,config)
    try:
        await st.connect()
        await st.require_idle_controller()
        before=list((await st.arm.get_joint_positions()).values)
        start=time.monotonic()
        rotation,translation=await st.cam_frame()
        depth,gray,bgr,_,summary,in_frame=await st.capture()
        elapsed=time.monotonic()-start
        await st.require_idle_controller()
        after=list((await st.arm.get_joint_positions()).values)
        if not before or len(before)!=len(after) or not np.isfinite(before+after).all():
            raise ValueError('Invalid joint position report')
        if elapsed>settings['max_capture_s'] or max(abs(a-b) for a,b in zip(before,after))>settings['max_joint_change_deg']:
            raise ValueError('Capture too slow or arm pose changed; no mapping produced')
        if bgr is None:raise ValueError('Missing RGB image')
        Image.fromarray(bgr[:,:,::-1].copy()).save(folder/'frame.png')
        np.save(folder/'depth.npy',depth)
        if gray is not None:np.save(folder/'gray.npy',gray)
        meta=dict(intrinsics=asdict(st.K),rotation=rotation.tolist(),translation_mm=translation.tolist(),
                  joints_before_deg=before,joints_after_deg=after,capture_seconds=elapsed,
                  in_frame=in_frame,point_cloud_summary=summary,pose='current; no homing or motion',
                  acquisition='sequential RGB and fused point clouds while stationary')
        (folder/'capture.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    finally:
        if hasattr(st,'robot'):await st.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args()
    settings=yaml.safe_load((ROOT/'config/crop_target_mapping.yaml').read_text())
    config=yaml.safe_load((ROOT/'config/config.yaml').read_text())
    crop_settings=yaml.safe_load((ROOT/config['wrist_inspection_config']).read_text())['auto_crop']
    folder=args.replay or ROOT/'logs/crop_target_mapping'/datetime.now(timezone.utc).strftime('capture_%Y%m%dT%H%M%S_%fZ')
    if args.replay is None:
        folder.mkdir(parents=True)
        asyncio.run(capture(folder,config,settings))
    analyze(folder,settings,crop_settings)


if __name__=='__main__':main()
