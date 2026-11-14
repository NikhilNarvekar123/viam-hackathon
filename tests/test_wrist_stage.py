import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from PIL import Image
import yaml
from anomaly.wrist import WristInspector
from main import run_overview_loop

ROOT=Path(__file__).resolve().parents[1]


def settings():
    return yaml.safe_load((ROOT/'config/wrist_inspection.yaml').read_text())


def test_process_manual_crop_and_threshold(tmp_path):
    cfg=settings();cfg['rois']=[[0,0,16,16]];cfg['model']['threshold']=0.4
    detector=SimpleNamespace(device='mock',detect=Mock(return_value=SimpleNamespace(score=0.4,heatmap=np.zeros((16,16)))))
    stage=WristInspector(cfg,tmp_path,detector=detector)
    result=stage.process(Image.new('RGB',(32,32),'gray'))
    assert result['candidate_count']==1 and result['anomalous_count']==1
    assert (Path(result['output_dir'])/'scored_objects.png').exists()
    detector.detect.assert_called_once()


def test_empty_frame_no_inference(tmp_path):
    detector=SimpleNamespace(device='mock',detect=Mock())
    stage=WristInspector(settings(),tmp_path,detector=detector)
    result=stage.process(Image.new('RGB',(640,480),'gray'))
    assert result['candidate_count']==0
    detector.detect.assert_not_called()


def test_compile_once_and_capture_after_each_overview(tmp_path):
    events=[]
    async def overview(*args):
        events.append('home_overview');return {'candidate_count':3}
    async def wrist():
        events.append('wrist');return {'output_dir':str(tmp_path),'candidate_count':2,'anomalous_count':1}
    inspector=SimpleNamespace(capture_and_inspect=AsyncMock(side_effect=wrist))
    cfg={'overview':{'cycle_delay_s':0.1},'wrist_inspection_config':str(ROOT/'config/wrist_inspection.yaml')}
    with patch('anomaly.wrist.WristInspector',return_value=inspector) as build, patch('robot.overview.overview_stage',side_effect=overview):
        asyncio.run(run_overview_loop(cfg,fake=True,cycles=2))
    build.assert_called_once()
    assert events==['home_overview','wrist','home_overview','wrist']
