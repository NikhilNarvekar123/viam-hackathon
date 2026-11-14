"""Wrist camera -> crops -> Anomalib -> Viam mainVLM. No arm/gripper commands."""
import argparse
import asyncio
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw
import yaml
from anomaly.object_crops import automatic_crops, extract_candidate_crops
from anomaly.detector import OpenVINODetector
from anomaly.stub import StubDetector
from robot.locate import annotate
from robot.viam_hw import connect,disconnect,load_credentials
from tools.inspect_wrist import capture,validate_rois,inspect,annotate_scores


def parse_answer(raw, combinations=None, response_schema='legacy'):
    fallback={'defect_type':'uncertain','action':'hold','reason':'Invalid VLM response; manual review required.','valid':False}
    try:
        if isinstance(raw,str):
            raw=raw.strip()
            lines=raw.splitlines()
            if len(lines)>=3 and lines[0] in ('```json','```') and lines[-1]=='```':
                raw='\n'.join(lines[1:-1])
        result=json.loads(raw)
        if response_schema=='grounded_inspection':
            if not isinstance(result,dict) or list(result)!=['observation','defect']:
                return fallback
            observation=result['observation'];defect=result['defect']
            if not isinstance(observation,str) or not observation.strip() or len(observation)>1000 or type(defect) is not bool:
                return fallback
            return dict(observation=observation,defect=defect,
                        action='reject' if defect else 'accept',
                        defect_type='physical_defect' if defect else 'none',reason=observation,valid=True)
        if response_schema!='legacy': return fallback
        if not isinstance(result,dict) or set(result)!={'defect_type','action','reason'}:
            return fallback
        combinations=combinations or {'accept':{'none'},
            'reject':{'tape','marker','tape_and_marker','foreign_material','structural_damage','mixed'},
            'hold':{'uncertain'}}
        action=result['action'];defect=result['defect_type'];reason=result['reason']
        if not isinstance(action,str) or not isinstance(defect,str) or defect not in combinations.get(action,set()):
            return fallback
        if not isinstance(reason,str) or not reason.strip() or len(reason)>1000:
            return fallback
        return dict(result,valid=True)
    except (ValueError,TypeError):
        return fallback


async def ask_crop(service,image,prompt,settings):
    buffer=BytesIO();image.convert('RGB').save(buffer,format='PNG')
    request={'command':'ask','question':prompt,
             'image_b64':base64.b64encode(buffer.getvalue()).decode('ascii'),
             'max_new_tokens':settings['max_new_tokens'],'do_sample':False}
    start=time.perf_counter()
    try:
        response=await service.do_command(request,timeout=settings['request_timeout_s'])
        raw=response.get('answer','')
        parsed=parse_answer(raw,settings.get('response_defect_types'),settings.get('response_schema','legacy'))
        return dict(parsed,raw_answer=raw,device=response.get('device'),
                    service_seconds=response.get('seconds'),latency_ms=(time.perf_counter()-start)*1000)
    except Exception as exc:
        return dict(defect_type='uncertain',action='hold',reason='VLM request failed; manual review required.',
                    valid=False,raw_answer=None,error=f'{type(exc).__name__}: {exc}',
                    latency_ms=(time.perf_counter()-start)*1000)


async def run(args):
    settings=yaml.safe_load(args.config.read_text())
    crop_settings=yaml.safe_load((ROOT/settings['crop_config']).read_text())
    crop_settings['camera_name']=settings['camera_name']
    if args.fake:
        image=Image.new('RGB',(640,480),'gray')
        draw=ImageDraw.Draw(image)
        draw.rectangle((200,180,230,230),fill='black')
        draw.rectangle((400,250,440,280),fill='black')
    elif args.image:
        with Image.open(args.image) as saved: image=saved.convert('RGB')
    else:
        print(f"Capturing {settings['camera_name']} (NO arm/gripper commands)",flush=True)
        image=await capture(crop_settings,False)
    folder=ROOT/settings['output_dir']/datetime.now(timezone.utc).strftime('vlm_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    image.save(folder/'frame.png')
    prepared_crops=None;localization=None
    if args.whole_frame:
        rois=[[0,0,image.width,image.height]]
        print('Treating entire input as ONE crop; use only for a single isolated object.')
    elif args.roi:
        rois=args.roi
    else:
        rois,localization,mask=automatic_crops(image,crop_settings['auto_crop'])
        prepared_crops=extract_candidate_crops(image,localization,mask)
        annotate(image,localization,mask).save(folder/'detected_objects.png')
        (folder/'candidates.json').write_text(json.dumps(localization,indent=2))
    if rois and prepared_crops is None: validate_rois(image,rois)
    print(f'{len(rois)} candidate crops. Results: {folder}',flush=True)
    threshold=crop_settings['model']['threshold']
    anomaly_records=[]
    if rois:
        print('Running Anomalib on all crops before VLM...',flush=True)
        detector=(StubDetector(threshold=threshold,error_rate=0,seed=7) if args.fake else
                  OpenVINODetector(crop_settings['model'],ROOT))
        anomaly_records=inspect(image,rois,detector,threshold,folder,crops=prepared_crops,
                                objects=localization['objects'] if localization else None)
        annotate_scores(image,anomaly_records).save(folder/'anomaly_scores.png')
        (folder/'anomalib_results.json').write_text(json.dumps(dict(model=crop_settings['model'],
            device=getattr(detector,'device','stub'),fake=args.fake,objects=anomaly_records),indent=2,allow_nan=False))
    # Use the same gating, evidence, response validation and routing as main.
    from vlm.service import decide_wrist_objects
    records=await decide_wrist_objects(dict(output_dir=str(folder),objects=anomaly_records),
                                      settings,ROOT,fake=args.fake)
    (folder/'results.json').write_text(json.dumps(dict(fake=args.fake,
        source=str(args.image) if args.image else settings['camera_name'],
        objects=records,actions_executed=False),indent=2,allow_nan=False)+'\n')
    print('No actions executed. Check detections and decisions against known labels.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--image',type=Path,help='Saved image instead of live wrist capture')
    source.add_argument('--fake',action='store_true',help='Synthetic scene, stub anomaly scores and mock VLM decisions; no connection')
    regions=parser.add_mutually_exclusive_group()
    regions.add_argument('--whole-frame',action='store_true',help='Input is already a single-object crop')
    regions.add_argument('--roi',nargs=4,type=int,action='append',metavar=('L','T','R','B'))
    parser.add_argument('--config',type=Path,default=ROOT/'config/vlm_service.yaml')
    args=parser.parse_args()
    try: asyncio.run(run(args))
    except KeyboardInterrupt: parser.exit(130,'Cancelled; connections closed.\n')
    except (ValueError,OSError,ImportError,KeyError,RuntimeError) as exc:
        parser.exit(1,f'VLM test error: {exc}\n')


if __name__=='__main__': main()
