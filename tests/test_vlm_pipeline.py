import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock,patch
from PIL import Image
import pytest
from vlm.service import route_decision,decide_wrist_objects,evidence

DEST={'accept':'box','reject':'reject_bin','hold':'hold_area'}


@pytest.mark.parametrize('anomaly,action,valid,expected',[
 ('anomalous','accept',True,'hold'),('normal','accept',True,'accept'),
 ('normal','reject',True,'reject'),('anomalous','reject',True,'reject'),
 ('normal','hold',True,'hold'),('normal','accept',False,'hold'),
 ('unclassified','accept',True,'hold'),('normal','move',True,'hold')])
def test_routing(anomaly,action,valid,expected):
    r=route_decision({'classification':anomaly},{'action':action,'valid':valid},DEST)
    assert r['final_action']==expected and r['destination']==DEST[expected]


def test_evidence_preserves_original():
    original=Image.new('RGB',(10,10),'green');overlay=Image.new('RGB',(10,10),'red')
    image,prompt=evidence(original,overlay,{'score':0.8,'threshold':0.4,'classification':'anomalous'},'Inspect.')
    assert image.size==original.size
    assert image.tobytes()==original.tobytes()
    assert image.getpixel((0,0))==original.getpixel((0,0))
    assert '0.800000' not in prompt and 'single centered target' in prompt


@pytest.mark.parametrize('classification', ['normal','anomalous','unclassified'])
def test_authoritative_vlm_accept(classification):
    r=route_decision({'classification':classification},{'action':'accept','valid':True},DEST,'vlm_authoritative')
    assert r['final_action']=='accept' and r['destination']=='box'


@pytest.mark.parametrize('valid,action', [(False,'accept'),(True,'hold'),(True,'move')])
def test_authoritative_policy_still_holds_invalid_or_uncertain(valid,action):
    assert route_decision({},dict(valid=valid,action=action),DEST,'vlm_authoritative')['final_action']=='hold'


def test_vlm_evidence_resizes_isolated_square_without_padding():
    original=Image.new('RGB',(36,36),(40,70,110))
    image,_=evidence(original,None,{},'Inspect.',image_size=224)
    assert image.size==(224,224)
    assert image.getpixel((0,0))==image.getpixel((223,223))==(40,70,110)


def test_raw_evidence_is_native_unmodified_frame_slice():
    from vlm.evidence import raw_crop
    import numpy as np
    frame=Image.fromarray(np.arange(60*80*3,dtype=np.uint8).reshape(60,80,3))
    crop=raw_crop(frame,{'bbox_xyxy':[20,20,30,30],'crop_bbox_xyxy':[10,5,50,45]})
    assert crop.size==(40,40)
    assert crop.tobytes()==frame.crop((10,5,50,45)).tobytes()


def test_real_service_path_sends_masked_native_crop_and_routes_accept(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    frame=Image.fromarray(np.arange(40*40*3,dtype=np.uint8).reshape(40,40,3))
    frame.save(tmp_path/'frame.png')
    # The prepared crop differs from the full frame; no overlay file is provided.
    masked=frame.crop((5,5,30,30));masked.putpixel((0,0),(150,150,150))
    masked.save(tmp_path/'object_001.png')
    wrist={'output_dir':str(tmp_path),'objects':[{'id':1,'score':1.0,'classification':'anomalous',
        'bbox_xyxy':[12,12,22,22],'crop_bbox_xyxy':[5,5,30,30]}]}
    settings={'service_name':'mainVLM','action_to_destination':DEST,'routing_policy':'vlm_authoritative',
              'prompt':'Configured inspection prompt.'}
    service=SimpleNamespace(do_command=AsyncMock(return_value={'state':'ready','prompt':'Old default.'}))
    answer=dict(action='accept',valid=True,reason='No visible defect.')
    with patch('vlm.service.connect',new_callable=AsyncMock), \
         patch('vlm.service.disconnect',new_callable=AsyncMock), \
         patch('vlm.service.load_credentials'), \
         patch('viam.services.vision.VisionClient.from_robot',return_value=service), \
         patch('vlm.service.ask_crop',new_callable=AsyncMock,return_value=answer) as ask:
        records=asyncio.run(decide_wrist_objects(wrist,settings,tmp_path))
    sent=ask.call_args.args[1]
    assert sent.size==(25,25) and sent.tobytes()==masked.tobytes()
    assert ask.call_args.args[2].startswith(settings['prompt'])
    assert records[0]['final_action']=='accept'


def test_stage1_pass_skips_vlm_connection(tmp_path):
    Image.new('RGB',(40,40)).save(tmp_path/'frame.png')
    wrist={'output_dir':str(tmp_path),'objects':[{'id':1,'score':.74,'threshold':.75,
           'classification':'normal','bbox_xyxy':[10,10,30,30]}]}
    settings={'service_name':'mainVLM','action_to_destination':DEST,'stage1_fast_pass':True}
    with patch('vlm.service.connect',new_callable=AsyncMock) as connect:
        records=asyncio.run(decide_wrist_objects(wrist,settings,tmp_path))
    connect.assert_not_awaited()
    assert records[0]['final_action']=='accept' and records[0]['vlm']['skipped']
    assert records[0]['vlm']['action'] is None


@pytest.mark.parametrize('score,expected',[(.74,True),(.75,False),(.76,False),(float('nan'),False)])
def test_stage1_threshold_boundary(score,expected):
    from vlm.evidence import stage1_pass
    assert stage1_pass(dict(score=score,threshold=.75,classification='normal'),
                       {'stage1_fast_pass':True})==expected


def test_disabled_vlm_never_connects_and_holds_escalated_objects(tmp_path):
    Image.new('RGB',(40,40)).save(tmp_path/'frame.png')
    wrist={'output_dir':str(tmp_path),'objects':[{'id':1,'score':1.0,'threshold':.75,
           'classification':'anomalous','bbox_xyxy':[10,10,30,30]}]}
    settings={'service_name':'mainVLM','action_to_destination':DEST,'stage1_fast_pass':True,'enabled':False}
    with patch('vlm.service.connect',new_callable=AsyncMock) as connect, \
         patch('vlm.service.ask_crop',new_callable=AsyncMock) as ask:
        records=asyncio.run(decide_wrist_objects(wrist,settings,tmp_path))
    connect.assert_not_awaited();ask.assert_not_awaited()
    assert records[0]['final_action']=='hold' and records[0]['vlm']['skipped']


def test_fake_stage_no_connection(tmp_path):
    Image.new('RGB',(40,40),'white').save(tmp_path/'frame.png')
    wrist={'output_dir':str(tmp_path),'objects':[{'id':1,'score':0.8,'classification':'anomalous','bbox_xyxy':[10,10,30,30]}]}
    settings={'service_name':'mainVLM','action_to_destination':DEST}
    with patch('vlm.service.connect',new_callable=AsyncMock) as connect:
        records=asyncio.run(decide_wrist_objects(wrist,settings,tmp_path,fake=True))
    connect.assert_not_awaited()
    assert records[0]['destination']=='hold_area'
    assert json.loads((tmp_path/'vlm_results.json').read_text())['actions_executed'] is False
    assert (tmp_path/'decisions.png').exists()
