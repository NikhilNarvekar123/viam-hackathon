"""Inspect exported output tensors directly, before wrapper/map visualization resizing."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import yaml
from anomaly.object_crops import automatic_crops,extract_candidate_crops
from anomaly.enhance import sharpen_crop
from anomaly.detector import OpenVINODetector
from anomaly.preprocess import image_tensor


def tensor_stats(values):
    array=np.asarray(values)
    if not array.size or not np.isfinite(array).all():
        raise ValueError('Expected nonempty finite tensor')
    flat=array.astype(np.float64).ravel()
    median=float(np.median(flat));maximum=float(flat.max())
    return dict(shape=list(array.shape),dtype=str(array.dtype),min=float(flat.min()),
                mean=float(flat.mean()),median=median,max=maximum,
                max_over_median=maximum/median if median!=0 else None,
                ratio_note='undefined: median is zero' if median==0 else None,
                fraction_at_one=float(np.mean(flat>=1.0-1e-6)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True,type=Path,help='Original lossless full frame, not screenshot')
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/'config/wrist_inspection.yaml').read_text())
    cfg=copy.deepcopy(config['auto_crop']);sharp=dict(cfg['crop_sharpening'],enabled=True)
    cfg['crop_sharpening']={'enabled':False}
    with Image.open(args.image) as saved:image=saved.convert('RGB')
    _,objects,mask=automatic_crops(image,cfg)
    crops=extract_candidate_crops(image,objects,mask)
    detector=OpenVINODetector(config['model'],ROOT)
    out=ROOT/'logs/anomaly_diagnostics'/datetime.now(timezone.utc).strftime('tensors_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    report=dict(source_frame=str(args.image.resolve()),config=config,sharp_settings=sharp,objects=[],
                note='Direct exported pred_score/anomaly_map tensors; IR includes normalization/clipping. No heatmap resize, VLM, camera capture or motion.')
    print('ID variant native pred_score map_min map_mean map_median map_max max/median saturated_fraction',flush=True)
    for obj,crop in zip(objects['objects'],crops):
        for name,processed in [('unsharpened',crop),('sharpened',sharpen_crop(crop,sharp))]:
            # Copy immediately: infer-request buffers may be reused by subsequent calls.
            outputs=detector.compiled({detector.input:image_tensor(processed,config['model'])})
            score=np.array(outputs[detector.score_output],copy=True)
            amap=np.array(outputs[detector.heatmap_output],copy=True)
            row=dict(id=obj['id'],variant=name,native_size=list(crop.size),bbox_xyxy=obj['bbox_xyxy'],
                     crop_bbox_xyxy=obj['crop_bbox_xyxy'],pred_score=tensor_stats(score),anomaly_map=tensor_stats(amap))
            spatial=np.squeeze(amap)
            if spatial.ndim!=2: raise ValueError('Expected a 2D spatial anomaly map after singleton axes removed')
            peak=np.unravel_index(np.argmax(spatial),spatial.shape)
            row['anomaly_map']['first_peak_yx']=list(map(int,peak))
            np.savez_compressed(out/f'object_{obj["id"]:03d}_{name}.npz',pred_score=score,anomaly_map=amap)
            processed.save(out/f'object_{obj["id"]:03d}_{name}.png')
            report['objects'].append(row)
            m=row['anomaly_map'];ratio=m['max_over_median']
            print(obj['id'],name,crop.size,round(row['pred_score']['mean'],6),
                  *(round(m[k],6) for k in ['min','mean','median','max']),
                  round(ratio,6) if ratio is not None else 'undefined',round(m['fraction_at_one'],6),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('REPORT',out/'report.json')


if __name__=='__main__':main()
