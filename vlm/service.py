"""Viam-hosted VLM stage over saved Anomalib crops; decisions only, no movement."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from robot.viam_hw import connect, disconnect, load_credentials
from tools.test_vlm import ask_crop
from vlm.evidence import colored_pixels, evidence, stage1_pass


def route_decision(anomaly, vlm, destinations, policy='disagreement_hold'):
    if policy not in ('vlm_authoritative','disagreement_hold'):
        raise ValueError('Unknown routing policy')
    action = vlm.get('action')
    reason = 'VLM decision'
    if not vlm.get('valid') or action not in destinations or action == 'hold':
        action,reason = 'hold','Uncertain, invalid, or unavailable VLM decision'
    elif policy == 'disagreement_hold' and action == 'accept' and anomaly.get('classification') != 'normal':
        action,reason = 'hold','Anomalib/VLM disagreement or missing anomaly classification'
    return dict(final_action=action,destination=destinations[action],routing_reason=reason)


async def decide_wrist_objects(wrist, settings, root, fake=False):
    folder=Path(wrist['output_dir'])
    records=[];machine=None;service=None;status=None
    # Blue tape decides alone, except while a voice-confirmed inspection policy is active: that goes to the VLM.
    color_only=bool(settings.get('color_only')) and not settings.get('inspection_policy')
    error=None if settings.get('enabled',True) else 'VLM disabled by configuration; manual review required'
    destinations=settings['action_to_destination']
    if set(destinations)!={'accept','reject','hold'}:
        raise ValueError('VLM destinations must define accept, reject, hold')
    with Image.open(folder/'frame.png') as original: frame=original.convert('RGB')
    annotated=frame.copy()
    draw=ImageDraw.Draw(annotated)
    try:
        if (any(not stage1_pass(o,settings) for o in wrist['objects']) and not fake and error is None
                and not color_only):
            try:
                from viam.services.vision import VisionClient
                machine=await connect(load_credentials(root/'.env'))
                service=VisionClient.from_robot(machine,settings['service_name'])
                status=await service.do_command({'command':'status'},timeout=30)
                prompt=settings.get('prompt',status.get('prompt'))
                if status.get('state')!='ready' or not isinstance(prompt,str) or not prompt.strip():
                    raise ValueError('VLM not ready or inspection prompt missing')
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                print(f'VLM unavailable: {error}. Escalated objects route to HOLD; Stage 1 passes are unchanged.')
        for anomaly in wrist['objects']:
            ident=anomaly['id']
            passed=stage1_pass(anomaly,settings)
            color=settings.get('color_check') or {}
            colored=None
            if color.get('enabled') and (folder/f'object_{ident:03d}.png').exists():
                with Image.open(folder/f'object_{ident:03d}.png') as saved: colored=colored_pixels(saved,float(color['min_saturation']),color.get('hue_range'))
            color_reject=colored is not None and colored>=int(color['min_pixels'])
            if color_reject:
                # Tape, paint or colored marker: a defect whatever Stage 1 or the VLM would say.
                passed=False
                vlm=dict(action='reject',defect_type='colored_tape_paint_or_marker',valid=True,skipped=True,
                         reason=f'Color check: {colored} strongly colored pixels; VLM not called',
                         raw_answer=None,device='cpu',latency_ms=0)
            elif color_only:
                passed=True  # the color check is the whole decision: no blue tape -> accept
                vlm=dict(action=None,defect_type=None,reason='No blue tape: accept (color-only rule); VLM not called',
                         valid=False,skipped=True,raw_answer=None,device=None,latency_ms=0)
            elif passed:
                vlm=dict(action=None,defect_type=None,reason='Stage 1 below threshold; VLM not called',
                         valid=False,skipped=True,raw_answer=None,device=None,latency_ms=0)
            elif fake or error:
                vlm=dict(action='hold',defect_type='uncertain',reason='FAKE decision' if fake else error,
                         valid=bool(fake),raw_answer=None,device='fake' if fake else None,latency_ms=0)
                if error: vlm['error']=error
                if not settings.get('enabled',True): vlm['skipped']=True
            else:
                # Saved by the shared extractor at native resolution, before model resize.
                with Image.open(folder/f'object_{ident:03d}.png') as saved: crop=saved.convert('RGB')
                combined,question=evidence(crop,None,anomaly,settings.get('prompt',status.get('prompt')),
                                           settings.get('image_size_px'))
                combined.save(folder/f'object_{ident:03d}_vlm_input.png')
                vlm=await ask_crop(service,combined,question,settings)
            routing=(dict(final_action='reject',destination=destinations['reject'],routing_reason='Color check')
                     if color_reject else
                     dict(final_action='accept',destination=destinations['accept'],
                          routing_reason='No blue tape (color-only rule)' if color_only
                          else 'Stage 1 score below operating threshold') if passed else
                     route_decision(anomaly,vlm,destinations,settings.get('routing_policy','disagreement_hold')))
            routing['colored_pixels']=colored
            record=dict(id=ident,bbox_xyxy=anomaly['bbox_xyxy'],anomaly=anomaly,vlm=vlm,
                        vlm_image_type=None if passed or vlm.get('skipped') else 'neighbor_masked_square_resized',
                        routing_policy=settings.get('routing_policy','disagreement_hold'),
                        inspection_policy=settings.get('inspection_policy'),
                        **routing)
            records.append(record)
            print(f"Object {ident}: Anomalib={anomaly['score']:.3f}, VLM={vlm['action']}, "
                  f"final={record['final_action']} -> {record['destination']}: {vlm['reason']}",flush=True)
            l,t,r,b=anomaly['bbox_xyxy']
            color={'accept':'lime','reject':'red','hold':'yellow'}[record['final_action']]
            draw.rectangle((l,t,r-1,b-1),outline=color,width=2)
            draw.text((max(0,l),max(0,t-14)),f"#{ident} {record['final_action']}",fill=color,stroke_width=1,stroke_fill='black')
            (folder/'vlm_results.json').write_text(json.dumps(dict(fake=fake,service=settings['service_name'],
                status=status,prompt=settings.get('prompt',status.get('prompt') if status else None),
                inspection_policy=settings.get('inspection_policy'),
                objects=records,actions_executed=False),indent=2,allow_nan=False)+'\n')
            annotated.save(folder/'decisions.png')
        if not records:
            (folder/'vlm_results.json').write_text(json.dumps({'objects':[],'actions_executed':False}))
            annotated.save(folder/'decisions.png')
        print(f"VLM decisions: {folder/'vlm_results.json'} (no actions executed)")
        return records
    finally:
        if machine is not None: await disconnect(machine)
