import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
import pytest
from tools.test_vlm import parse_answer,ask_crop


@pytest.mark.parametrize('raw',['not json','{}','[]','{"action":"accept","defect_type":"tape","reason":"x"}',
    '{"action":"move","defect_type":"none","reason":"x"}',
    '{"action":[],"defect_type":"none","reason":"x"}'])
def test_invalid_response_holds(raw):
    result=parse_answer(raw)
    assert result['action']=='hold' and not result['valid']


def test_fenced_json():
    raw='```json\n'+json.dumps(dict(action='accept',defect_type='none',reason='No defect visible.'))+'\n```'
    assert parse_answer(raw)['valid']


@pytest.mark.parametrize('defect',['foreign_material','structural_damage','mixed'])
def test_generalized_defects_parse(defect):
    result=parse_answer(json.dumps(dict(action='reject',defect_type=defect,reason='Visible defect.')))
    assert result['valid'] and result['action']=='reject'


@pytest.mark.parametrize('defect,action',[(True,'reject'),(False,'accept')])
def test_grounded_schema_maps_boolean(defect,action):
    result=parse_answer(json.dumps(dict(observation='Visible surface description.',defect=defect)),
                        response_schema='grounded_inspection')
    assert result['valid'] and result['action']==action
    assert result['observation']==result['reason'] and result['defect'] is defect


@pytest.mark.parametrize('raw',[
    '{"defect":true,"observation":"Coating."}',
    '{"observation":"","defect":false}',
    '{"observation":"Unclear.","defect":null}',
    '{"observation":"Metal.","defect":"false"}',
    '{"observation":"Metal.","defect":0}',
    '{"action":"accept","defect_type":"none","reason":"Metal."}'])
def test_grounded_schema_uncertain_invalid_or_wrong_order_holds(raw):
    result=parse_answer(raw,response_schema='grounded_inspection')
    assert not result['valid'] and result['action']=='hold'


def test_valid_response():
    result=parse_answer(json.dumps(dict(action='reject',defect_type='tape',reason='Visible tape.')))
    assert result['action']=='reject' and result['valid']


def test_sends_supplied_image_and_evidence():
    service=SimpleNamespace(do_command=AsyncMock(return_value={'answer':json.dumps(dict(action='hold',defect_type='uncertain',reason='Poor view.')),'device':'GPU'}))
    settings={'request_timeout_s':120,'max_new_tokens':128}
    result=asyncio.run(ask_crop(service,Image.new('RGB',(40,20)),'Evidence prompt',settings))
    request=service.do_command.call_args.args[0]
    assert request['question']=='Evidence prompt' and 'camera_name' not in request
    payload=base64.b64decode(request['image_b64'])
    assert payload.startswith(b'\x89PNG\r\n\x1a\n')
    decoded=Image.open(BytesIO(payload))
    assert decoded.format=='PNG' and decoded.size==(40,20)
    assert decoded.convert('RGB').tobytes()==Image.new('RGB',(40,20)).tobytes()
    assert result['action']=='hold' and result['device']=='GPU'


def test_inference_error_holds():
    service=SimpleNamespace(do_command=AsyncMock(side_effect=RuntimeError('unavailable')))
    result=asyncio.run(ask_crop(service,Image.new('RGB',(16,16)),'test',{'request_timeout_s':1,'max_new_tokens':128}))
    assert result['action']=='hold' and not result['valid']
