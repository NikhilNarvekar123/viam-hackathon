"""Offline-only Stage 1 diagnostics. Never connects to Viam or calls a VLM."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
import yaml
from anomaly.detector import OpenVINODetector
from anomaly.preprocess import image_tensor,prepare_image
from anomaly.object_crops import automatic_crops,extract_candidate_crops
from anomaly.threshold import score_stats,suggest_threshold,classification_counts
from core.types import Frame
from robot.locate import annotate


def graph_contract(path):
    import openvino as ov
    model=ov.Core().read_model(str(path))
    pre=[]
    for node in model.get_ordered_ops():
        if node.get_type_name()=='Convolution': break
        rec={'name':node.friendly_name,'type':node.get_type_name()}
        if node.get_type_name()=='Constant':
            values=node.get_data()
            if values.size<=16: rec['values']=values.reshape(-1).tolist()
        pre.append(rec)
    post=[]
    for node in model.get_ordered_ops():
        if node.get_type_name()=='Clamp':
            post.append({'name':node.friendly_name,'attributes':node.get_attributes()})
    return dict(input_shape=list(model.input().shape),dtype=model.input().element_type.get_type_name(),
        declared_layout=str(ov.layout_helpers.get_layout(model.input())),
        inferred_layout='NCHW from input shape and first convolution channel axis',
        preprocessing_nodes=pre,clamps=post,outputs=[{'names':sorted(p.names),'shape':list(p.shape)} for p in model.outputs],
        metadata_files=[str(p) for p in path.parent.glob('*.json')])


def logged_frames(root,model_path):
    for path in sorted((root/'logs/wrist_inspection').glob('inspect_*/results.json')):
        r=json.loads(path.read_text())
        if r.get('replay_source') or r.get('fake') or r.get('source')=='fake' or r.get('camera_name')=='fake': continue
        prior_model=r.get('model') or r.get('settings',{}).get('model',{}).get('path')
        frame=path.parent/'frame.png'
        if prior_model!=model_path or not frame.exists(): continue
        with Image.open(frame) as image:
            if image.size not in [(640,480),(1280,960)]: continue
        yield frame,r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/wrist_inspection.yaml')
    parser.add_argument('--labels',type=Path,help='JSON mapping relative frame path#ID or sample path to good/anomaly/unknown')
    parser.add_argument('--limit',type=int,help='Only latest N logged frames')
    parser.add_argument('--sample',action='append',default=[],help='Optional single-object reference image; repeat for more. Labels come only from --labels.')
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text());settings=cfg['model']
    labels=json.loads(args.labels.read_text()) if args.labels else {}
    if any(v not in ('good','anomaly','unknown') for v in labels.values()):
        raise ValueError('Labels must be good, anomaly, or unknown')
    out=ROOT/'logs/anomaly_diagnostics'/datetime.now(timezone.utc).strftime('diagnostic_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    contract=graph_contract(ROOT/settings['path'])
    (out/'graph_contract.json').write_text(json.dumps(contract,indent=2))
    print('INPUT',contract['input_shape'],contract['dtype'],contract['inferred_layout'])
    detector=OpenVINODetector(settings,ROOT)
    report={'config':cfg,'graph_contract':contract,'samples':[],'baselines':[],'frames':[],'threshold_changed':False}
    # Reference images and ground-truth labels are private inputs, never embedded in source.
    sample_names=args.sample
    template={}
    for name in sample_names:
        path=ROOT/name
        if not path.exists(): continue
        image=Image.open(path).convert('RGB');t=image_tensor(image,settings)
        label=labels.get(name,'unknown');template[name]=label
        score=detector.detect(Frame(name,image)).score
        variants={'exact_wrapper_RGB_0_1':score,'BGR_0_1':detector._infer(np.ascontiguousarray(t[:,::-1]))[0],
                  'RGB_0_255':detector._infer(t*255)[0],
                  'double_imagenet':detector._infer((t-np.array([.485,.456,.406],dtype=np.float32)[None,:,None,None])/
                         np.array([.229,.224,.225],dtype=np.float32)[None,:,None,None])[0]}
        report['samples'].append(dict(key=name,label=label,score=score,variants=variants))
        print('SAMPLE',name,label,variants)
    frames=list(logged_frames(ROOT,settings['path']))
    if args.limit is not None: frames=frames[-args.limit:]
    for frame,old in frames:
        image=Image.open(frame).convert('RGB')
        rois,result,mask=automatic_crops(image,cfg['auto_crop'])
        crops=extract_candidate_crops(image,result,mask)
        pixels=np.asarray(image)
        color=np.median(pixels[~mask],axis=0).astype('uint8').tolist() if (~mask).any() else np.median(pixels.reshape(-1,3),axis=0).astype('uint8').tolist()
        blank=Image.new('RGB',(settings['image_size'],)*2,tuple(color))
        baseline=detector.detect(Frame('gray',blank)).score
        report['baselines'].append(dict(frame=str(frame.relative_to(ROOT)),median_rgb=color,score=baseline))
        destination=out/frame.parent.name;destination.mkdir()
        annotate(image,result,mask).save(destination/'detected_objects.png')
        sheet=Image.new('RGB',(len(crops)*160 if crops else 160,330),'white');draw=ImageDraw.Draw(sheet)
        rows=[]
        old_objects={o['id']:o for o in old['objects']}
        for obj,crop in zip(result['objects'],crops):
            ident=obj['id']
            # Filtering renumbers IDs. Never transfer a user's label by the new ID.
            matches=[o for o in old_objects.values() if o['bbox_xyxy']==obj['bbox_xyxy']]
            prior=matches[0] if len(matches)==1 else None
            stable_id=str(prior['id']) if prior else 'bbox:'+','.join(map(str,obj['bbox_xyxy']))
            key=f'{frame.relative_to(ROOT)}#{stable_id}';label=labels.get(key,'unknown');template[key]=label
            score=detector.detect(Frame(key,crop)).score
            crop.save(destination/f'object_{ident:03d}.png')
            # Keep before/after association explicit; IDs may change if segmentation changes.
            row=dict(key=key,id=ident,label=label,score=score,bbox_xyxy=obj['bbox_xyxy'],crop_bbox_xyxy=obj['crop_bbox_xyxy'],
                     background_color_rgb=obj['background_color_rgb'],neighbor_pixels_masked=obj['neighbor_pixels_masked'],
                     prior_score_same_id=prior['score'] if prior else None,prior_id_match_verified=prior is not None,
                     prior_object_id=prior['id'] if prior else None)
            rows.append(row)
            oldcrop=frame.parent/f"object_{prior['id']:03d}.png" if prior else frame.parent/'unmatched.png'
            if oldcrop.exists():
                with Image.open(oldcrop) as im: sheet.paste(im.convert('RGB').resize((128,128)),((ident-1)*160,25))
            sheet.paste(prepare_image(crop,128),((ident-1)*160,190))
            draw.text(((ident-1)*160,5),f'#{ident} OLD saved',fill='black')
            draw.text(((ident-1)*160,165),f'NEW {score:.4f}',fill='black')
        sheet.save(destination/'before_after.png')
        (destination/'objects.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        report['frames'].append(dict(frame=str(frame.relative_to(ROOT)),objects=rows))
        print(frame.parent.name,'median gray',color,'baseline',round(baseline,6),
              'ALL NEW SCORES',[(r['id'],round(r['score'],6)) for r in rows],flush=True)
    all_rows=report['samples']+[r for f in report['frames'] for r in f['objects']]
    distributions={label:[r['score'] for r in all_rows if r['label']==label] for label in ['good','anomaly','unknown']}
    report['distributions']={label:dict(scores=values,stats=score_stats(values)) for label,values in distributions.items()}
    if distributions['good'] and distributions['anomaly']:
        report['threshold_proposal']=suggest_threshold(distributions['good'],distributions['anomaly'])
        report['threshold_proposal']['validation']='In-sample proposal only; evaluate independent labeled holdout before deployment'
        report['current_threshold_metrics']=classification_counts(distributions['good'],distributions['anomaly'],settings['threshold'])
    else:
        report['threshold_proposal']=None
        report['calibration_blocker']='No confirmed labeled examples in both classes. Do not infer labels from scores or VLM outputs.'
    with (out/'scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['key','label','score','prior_score_same_id'])
        writer.writeheader()
        writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in all_rows)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'labels_template.json').write_text(json.dumps(template,indent=2)+'\n')
    print('DISTRIBUTIONS',json.dumps(report['distributions'],indent=2))
    print('THRESHOLD',report.get('threshold_proposal'),report.get('calibration_blocker',''))
    print('REPORT',out/'report.json')


if __name__=='__main__':main()
