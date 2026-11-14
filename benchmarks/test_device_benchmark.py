"""Benchmark harness tests use a FAKE OpenVINO backend, not device validation."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

spec=importlib.util.spec_from_file_location('device_benchmark',Path(__file__).resolve().parent/'benchmark_anomalib_devices.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def setup(tmp_path):
    root=tmp_path/'production';work=tmp_path/'benchmark'
    (root/'config').mkdir(parents=True);(root/'models').mkdir()
    (root/'main.py').write_text('raise RuntimeError("Production code must never be imported")')
    model=root/'models/model.xml';model.write_text('fake-model');model.with_suffix('.bin').write_bytes(b'fake-weights')
    (root/'config/wrist_inspection.yaml').write_text(yaml.safe_dump({'model':dict(path='models/model.xml',
        image_size=128,normalization_in_model=True,score_output='pred_score',threshold=.75)}))
    folder=root/'logs/wrist_inspection/inspect_1';folder.mkdir(parents=True)
    (folder/'results.json').write_text('{"fake":false}')
    Image.new('RGB',(36,36),'gray').save(folder/'object_001.png')
    args=argparse.Namespace(root=root,work=work,image=None)
    bench.init(args)
    return root,work,json.loads((work/'state.json').read_text())


def test_snapshot_detects_production_changes(tmp_path):
    root,_,state=setup(tmp_path);bench.verify(state)
    (root/'main.py').write_text('changed')
    with pytest.raises(RuntimeError,match='changed'):bench.verify(state)


def test_workspace_inside_production_refused(tmp_path):
    with pytest.raises(ValueError,match='OUTSIDE'):
        bench.init(argparse.Namespace(root=tmp_path,work=tmp_path/'bench',image=None))


def test_input_preprocessing_contract(tmp_path):
    path=tmp_path/'input.png';Image.new('RGB',(36,36),(255,0,0)).save(path)
    values=bench.input_array(path)
    assert values.shape==(1,3,128,128) and values.dtype==np.float32
    assert np.all(values[:,0]==1) and np.all(values[:,1:]==0)


def fake_ov(execution_override=None):
    class Port:
        shape=(1,3,128,128)
        element_type=SimpleNamespace(get_type_name=lambda:'f32')
        def __init__(self,name):self.name=name
        def get_names(self):return {self.name}
    class Compiled:
        outputs=[Port('pred_score'),Port('anomaly_map')]
        def __init__(self,device):self.device=device
        def get_property(self,name):return [execution_override or self.device]
        def input(self,index):return Port('image')
        def create_infer_request(self):
            delta=0 if self.device=='CPU' else .02
            outputs=[np.array([.74+delta],np.float32),np.full((1,1,4,4),.3+delta,np.float32)]
            return SimpleNamespace(infer=lambda _:None,get_output_tensor=lambda i:SimpleNamespace(data=outputs[i]))
    class Core:
        available_devices=['CPU','GPU','NPU']
        def set_property(self,value):assert value=={'CACHE_DIR':''}
        def get_property(self,*args):return 'MOCK DEVICE'
        def read_model(self,path):return SimpleNamespace(inputs=[Port('image')],input=lambda _:Port('image'))
        def compile_model(self,model,device,options):
            expected={'PERFORMANCE_HINT':'LATENCY','CACHE_DIR':''}
            if device=='GPU':expected['INFERENCE_PRECISION_HINT']='f32'
            assert options==expected
            return Compiled(device)
    return SimpleNamespace(Core=Core,__version__='MOCK')


def test_cpu_reference_and_gpu_gate_change(tmp_path,monkeypatch):
    _,work,state=setup(tmp_path)
    monkeypatch.setitem(sys.modules,'openvino',fake_ov())
    for device in ['CPU','GPU']:
        args=argparse.Namespace(work=work,device=device,runs=3,warmup=1)
        assert bench.worker(args,state,work/(device+'.json'))==0
    report=json.loads((work/'GPU.json').read_text())
    assert report['execution_devices']==['GPU']
    assert report['anomaly_gate_changed'] is True
    assert not report['versus_cpu']['output_0']['allclose']
    assert (work/'cpu_reference.npz').stat().st_size<1024*1024
    bench.verify(state)


def test_device_fallback_cannot_be_reported_as_npu(tmp_path,monkeypatch):
    _,work,state=setup(tmp_path)
    monkeypatch.setitem(sys.modules,'openvino',fake_ov(execution_override='CPU'))
    args=argparse.Namespace(work=work,device='NPU',runs=3,warmup=1)
    assert bench.worker(args,state,work/'NPU.json')==1
    report=json.loads((work/'NPU.json').read_text())
    assert report['status']=='error' and 'not confirmed' in report['error']
