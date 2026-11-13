import asyncio
from pathlib import Path
import json
from unittest.mock import patch

from PIL import Image, ImageDraw
import yaml

from robot.overview import count_candidates, overview_stage

ROOT=Path(__file__).resolve().parents[1]


def locator():
    return yaml.safe_load((ROOT/'config/localization.yaml').read_text())


def test_counts_and_filters_edges():
    background=Image.new('RGB',(640,480),'white')
    image=background.copy();draw=ImageDraw.Draw(image)
    draw.rectangle((120,100,150,140),fill='black')
    draw.rectangle((300,200,350,250),fill='black')
    draw.rectangle((0,300,20,350),fill='black')
    result,_=count_candidates(image,background,locator(),12000)
    assert result['candidate_count']==2
    assert len(result['rejected_candidates'])==1
    assert [o['id'] for o in result['objects']]==[1,2]


def test_empty_frame():
    image=Image.new('RGB',(640,480),'white')
    result,_=count_candidates(image,image,locator(),12000)
    assert result['candidate_count']==0


def test_fake_stage_no_connection_or_models(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config/localization.yaml').write_text((ROOT/'config/localization.yaml').read_text())
    cfg=yaml.safe_load((ROOT/'config/config.yaml').read_text())
    with patch('robot.overview.capture_camera',side_effect=AssertionError('Must not connect')):
        result=asyncio.run(overview_stage(cfg,tmp_path,fake=True))
    assert result['candidate_count']==3 and result['fake']
    reports=list((tmp_path/'logs/overview').glob('*/objects.json'))
    assert len(reports)==1
    assert json.loads(reports[0].read_text())['candidate_count']==3
    assert (reports[0].parent/'detected_objects.png').exists()
