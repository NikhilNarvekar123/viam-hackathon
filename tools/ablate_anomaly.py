"""Offline 2x2 masking/sharpening diagnostic; no Viam or VLM calls."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw
import yaml
from anomaly.object_crops import automatic_crops,extract_candidate_crops
from anomaly.enhance import sharpen_crop
from anomaly.detector import OpenVINODetector
from core.types import Frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/'config/wrist_inspection.yaml').read_text())
    settings=copy.deepcopy(config['auto_crop'])
    sharp=dict(settings.get('crop_sharpening',{}),enabled=True)
    settings['crop_sharpening']={'enabled':False}
    with Image.open(args.image) as saved:image=saved.convert('RGB')
    boxes,result,mask=automatic_crops(image,settings)
    masked=extract_candidate_crops(image,result,mask)  # Also rejects out-of-frame squares.
    raw=[image.crop(box) for box in boxes]
    variants={'raw':raw,'raw_sharp':[sharpen_crop(crop,sharp) for crop in raw],
              'masked':masked,'masked_sharp':[sharpen_crop(crop,sharp) for crop in masked]}
    model=OpenVINODetector(config['model'],ROOT)
    out=ROOT/'logs/anomaly_diagnostics'/datetime.now(timezone.utc).strftime('ablation_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    rows=[]
    grid=Image.new('RGB',(4*180,max(1,len(boxes))*205),'white');draw=ImageDraw.Draw(grid)
    for i,obj in enumerate(result['objects']):
        row={'id':obj['id'],'bbox_xyxy':obj['bbox_xyxy'],'crop_bbox_xyxy':obj['crop_bbox_xyxy'],'scores':{}}
        for column,(name,crops) in enumerate(variants.items()):
            crop=crops[i];detection=model.detect(Frame(str(obj['id']),crop))
            row['scores'][name]=detection.score
            crop.save(out/f'object_{obj["id"]:03d}_{name}.png')
            x=column*180;y=i*205
            grid.paste(crop.resize((160,160)),(x+10,y+30))
            draw.text((x+5,y+5),f'#{obj["id"]} {name}: {detection.score:.4f}',fill='black')
        rows.append(row)
        print(obj['id'],row['scores'],flush=True)
    report={'source_frame':str(args.image.resolve()),'config':config,'objects':rows,
            'note':'Diagnostic only. No threshold/settings changes, motion or VLM inference.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    grid.save(out/'comparison.png')
    print('REPORT',out/'report.json')


if __name__=='__main__':main()
