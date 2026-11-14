#!/usr/bin/env python3
"""Isolated saved-crop OpenVINO benchmark; NEVER imports robot/project code.

No downloads, package installs, service changes, camera captures or model writes.
Run init, then run CPU, GPU, NPU individually. Production source/config/model/input
hashes are checked before and after each run. Reported device is checked; AUTO and
HETERO are not used. CPU reference arrays are capped at 16 MiB uncompressed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(state):
    bad=[]
    for path,expected in state['hashes'].items():
        if not Path(path).is_file() or sha(path)!=expected:bad.append(path)
    if bad:raise RuntimeError('Production/input files changed or disappeared: '+', '.join(bad))


def init(args):
    import yaml
    root=args.root.resolve();work=args.work.resolve()
    if root==work or root in work.parents:raise ValueError('Work directory must be OUTSIDE production repo')
    if (work/'state.json').exists():raise ValueError('state.json already exists; use verify or choose a new --work directory')
    settings=yaml.safe_load((root/'config/wrist_inspection.yaml').read_text())['model']
    model=Path(settings['path'])
    if not model.is_absolute():model=root/model
    if model.is_dir():model=model/'model.xml'
    model=model.resolve();weights=model.with_suffix('.bin')
    for path in [root/'main.py',model,weights]:
        if not path.is_file():raise ValueError('Missing '+str(path))
    if not settings.get('normalization_in_model') or settings.get('image_size')!=128:
        raise ValueError('This benchmark is for the configured RGB /255, embedded-normalization, 128x128 export')
    image=args.image.resolve() if args.image else None
    if image is None:
        for folder in sorted((root/'logs/wrist_inspection').glob('inspect_*'),reverse=True):
            try:report=json.loads((folder/'results.json').read_text())
            except (OSError,ValueError):continue
            if report.get('fake') is not False:continue
            candidates=sorted(folder.glob('object_[0-9][0-9][0-9].png'))
            if candidates:image=candidates[0].resolve();break
    if image is None or not image.is_file():raise ValueError('No saved REAL crop found; supply --image /path/to/crop.png')
    files={root/'main.py',model,weights,image}
    files.update(p for p in (root/'config').rglob('*') if p.is_file())
    for name in ['anomaly','core','robot','vlm','tools','bench']:
        files.update((root/name).rglob('*.py'))
    print('Hashing production source/config, model and selected crop; reading only.',flush=True)
    hashes={str(p.resolve()):sha(p) for p in sorted(files)}
    try:
        git=subprocess.check_output(['git','--no-optional-locks','-C',str(root),'status','--porcelain'],text=True,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.CalledProcessError):git=None
    work.mkdir(parents=True,exist_ok=True)
    dump(work/'state.json',dict(root=str(root),model=str(model),image=str(image),model_settings=settings,
                               hashes=hashes,initial_git_status=git))
    print('Baseline saved:',work/'state.json')
    print('Model:',model)
    print('Saved real crop:',image)
    print('No inference performed. Next: python benchmark_anomalib_devices.py run CPU')


def input_array(image):
    import numpy as np
    from PIL import Image
    with Image.open(image) as opened:im=opened.convert('RGB')
    if im.width!=im.height:
        pixels=np.asarray(im)
        edge=np.concatenate([pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]])
        color=tuple(np.median(edge,axis=0).astype(np.uint8).tolist())
        side=max(im.size);square=Image.new('RGB',(side,side),color)
        square.paste(im,((side-im.width)//2,(side-im.height)//2));im=square
    im=im.resize((128,128),Image.Resampling.BILINEAR)
    return np.ascontiguousarray((np.asarray(im,dtype=np.float32)/255.).transpose(2,0,1)[None])


def worker(args,state,report_path):
    report=dict(device_requested=args.device,status='starting',benchmark='single saved crop; model inference only',
                model_sha256=state['hashes'][state['model']],image_sha256=state['hashes'][state['image']],
                persistent_openvino_cache=False)
    stage='import'
    try:
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))  # No large core dump on native compiler abort.
        import numpy as np
        import openvino as ov
        core=ov.Core();core.set_property({'CACHE_DIR':''})
        report['openvino_version']=ov.__version__
        report['available_devices']=core.available_devices
        if not any(d.split('.')[0]==args.device for d in core.available_devices):raise RuntimeError('Requested device is not available')
        try:report['device_name']=str(core.get_property(args.device,'FULL_DEVICE_NAME'))
        except Exception:report['device_name']='unavailable'
        values=input_array(state['image'])
        report['input_tensor_sha256']=hashlib.sha256(values.tobytes()).hexdigest()
        stage='read_model';start=time.perf_counter()
        model=core.read_model(state['model'])
        report['read_model_ms']=(time.perf_counter()-start)*1000
        if len(model.inputs)!=1 or tuple(model.input(0).shape)!=(1,3,128,128) or model.input(0).element_type.get_type_name()!='f32':
            raise ValueError('Unexpected input contract; refusing reshape or preprocessing guesses')
        stage='compile';print(f'{args.device}: compiling (no persistent OpenVINO cache)...',flush=True)
        start=time.perf_counter()
        options={'PERFORMANCE_HINT':'LATENCY','CACHE_DIR':''}
        if args.device=='GPU':options['INFERENCE_PRECISION_HINT']='f32'
        report['compile_options']=options
        compiled=core.compile_model(model,args.device,options)
        report['compile_ms']=(time.perf_counter()-start)*1000
        execution=list(compiled.get_property('EXECUTION_DEVICES'))
        report['execution_devices']=execution
        if not execution or any(str(d).split('.')[0]!=args.device for d in execution):
            raise RuntimeError('Requested device not confirmed by EXECUTION_DEVICES: '+repr(execution))
        request=compiled.create_infer_request()
        def infer():return request.infer({compiled.input(0):values})
        stage='first_inference';start=time.perf_counter();infer()
        report['first_inference_ms']=(time.perf_counter()-start)*1000
        stage='warmup'
        for _ in range(args.warmup):infer()
        stage='timed_inference';print(f'{args.device}: {args.runs} warmed inferences...',flush=True)
        timings=[]
        for _ in range(args.runs):
            start=time.perf_counter();infer();timings.append((time.perf_counter()-start)*1000)
        outputs={f'output_{i}':np.array(request.get_output_tensor(i).data,copy=True) for i in range(len(compiled.outputs))}
        if sum(v.nbytes for v in outputs.values())>16*1024*1024:raise ValueError('Output reference exceeds 16 MiB cap; not saving')
        if any(not np.isfinite(v).all() for v in outputs.values()):raise ValueError('Nonfinite model output')
        report['output_names']={f'output_{i}':sorted(port.get_names()) for i,port in enumerate(compiled.outputs)}
        settings=state.get('model_settings',state.get('settings'))
        score_name=settings.get('score_output','pred_score')
        score_index=next(i for i,p in enumerate(compiled.outputs) if score_name in p.get_names())
        score=outputs[f'output_{score_index}']
        if score.size!=1:raise ValueError('Expected scalar anomaly score')
        report['score']=float(score.reshape(-1)[0])
        threshold=settings.get('threshold')
        report['threshold']=threshold
        report['anomalous']=report['score']>=float(threshold) if threshold is not None else None
        report['runs']=args.runs;report['extra_warmups']=args.warmup
        report['latency_ms']=dict(p50=float(np.percentile(timings,50)),p95=float(np.percentile(timings,95)),
                                  minimum=float(min(timings)),maximum=float(max(timings)),mean=float(np.mean(timings)))
        report['timings_ms']=timings
        stage='reference_comparison'
        reference=args.work/'cpu_reference.npz';metadata=args.work/'cpu_reference.json'
        if args.device=='CPU':
            np.savez_compressed(reference,**outputs)
            dump(metadata,{k:report[k] for k in ['model_sha256','image_sha256','input_tensor_sha256','output_names','anomalous','score']})
            report['reference_bytes']=reference.stat().st_size
        else:
            cpu=json.loads(metadata.read_text())
            if 'input_tensor_sha256' not in cpu and 'input_sha256' in cpu:
                cpu['input_tensor_sha256']=cpu['input_sha256']
            for key in ['model_sha256','image_sha256','input_tensor_sha256','output_names']:
                if cpu[key]!=report[key]:raise ValueError('CPU reference mismatch: '+key)
            comparisons={}
            with np.load(reference,allow_pickle=False) as saved:
                for key,value in outputs.items():
                    baseline=saved[key]
                    if baseline.shape!=value.shape:raise ValueError('CPU output shape mismatch')
                    difference=np.abs(value.astype(np.float64)-baseline.astype(np.float64))
                    comparisons[key]=dict(max_abs_error=float(difference.max()),mean_abs_error=float(difference.mean()),
                         allclose=bool(np.allclose(value,baseline,rtol=.001,atol=.001)))
            report['versus_cpu']=comparisons
            report['score_delta']=report['score']-cpu['score']
            report['anomaly_gate_changed']=report['anomalous']!=cpu['anomalous']
            report['comparison_tolerances']=dict(atol=.001,rtol=.001,note='Numerical diagnostic, not accuracy validation')
        report['status']='ok'
    except Exception as exc:
        report.update(status='error',failed_stage=stage,error=f'{type(exc).__name__}: {exc}')
        print(f'{args.device}: {stage} failed: {exc}',flush=True)
    finally:
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_process_rss_mib']=rss/(1024 if sys.platform.startswith('linux') else 1024*1024)
        report['memory_note']='Process RSS is not total device/driver memory.'
        dump(report_path,report)
    return 0 if report['status']=='ok' else 1


def stop(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=5)


def run(args):
    state=json.loads((args.work/'state.json').read_text());verify(state)
    if args.device!='CPU' and not (args.work/'cpu_reference.json').exists():raise ValueError('Run CPU first')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    folder=args.work/f'{args.device}_{stamp}_{os.getpid()}';folder.mkdir()
    temp=folder/'tmp';temp.mkdir()
    env=os.environ.copy();env.update(TMPDIR=str(temp),XDG_CACHE_HOME=str(folder/'cache'),PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(Path(__file__).resolve()),'_worker',args.device,'--work',str(args.work),
             '--runs',str(args.runs),'--warmup',str(args.warmup),'--report',str(folder/'report.json')]
    process=None;reason=None;exit_code=1
    try:
        process=subprocess.Popen(command,env=env,cwd=args.work,start_new_session=True)
        start=time.monotonic()
        while process.poll() is None:
            if time.monotonic()-start>args.timeout:reason='Timeout';break
            if shutil.disk_usage(args.work).free<3*1024**3:reason='Disk space below 3 GiB';break
            if Path('/proc/meminfo').exists():
                for line in Path('/proc/meminfo').read_text().splitlines():
                    if line.startswith('MemAvailable:') and int(line.split()[1])*1024<args.min_available_gib*1024**3:
                        reason=f'Available RAM below {args.min_available_gib} GiB';break
            if reason:break
            time.sleep(.25)
        if reason:stop(process);dump(folder/'stopped.json',dict(reason=reason))
        else:exit_code=process.returncode
    finally:
        try:
            if process is not None:stop(process)
        finally:
            if process is None or process.poll() is not None:
                shutil.rmtree(temp,ignore_errors=True)
                shutil.rmtree(folder/'cache',ignore_errors=True)
            verify(state);print('Production/input file hashes: UNCHANGED',flush=True)
    report=folder/'report.json'
    if report.exists():
        value=json.loads(report.read_text())
        print(json.dumps({k:v for k,v in value.items() if k not in ['timings_ms','output_names']},indent=2))
    print('Report folder:',folder)
    if reason:raise RuntimeError('Benchmark stopped: '+reason)
    return exit_code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    init_parser=sub.add_parser('init');init_parser.add_argument('--root',type=Path,required=True)
    init_parser.add_argument('--image',type=Path)
    for name in ['run','_worker','verify']:
        p=sub.add_parser(name)
        if name!='verify':
            p.add_argument('device',choices=['CPU','GPU','NPU'])
            p.add_argument('--runs',type=int,default=30);p.add_argument('--warmup',type=int,default=3)
            p.add_argument('--timeout',type=float,default=180);p.add_argument('--min-available-gib',type=float,default=8)
            p.add_argument('--report',type=Path)
    for p in [init_parser,*[sub.choices[n] for n in ['run','_worker','verify']]]:
        p.add_argument('--work',type=Path,default=Path.home()/'viam-device-bench')
    args=parser.parse_args();args.work=args.work.resolve()
    if args.command=='init':init(args);return 0
    state=json.loads((args.work/'state.json').read_text())
    if args.command=='verify':verify(state);print('Production/input file hashes: UNCHANGED');return 0
    if args.runs<1 or args.warmup<0 or args.timeout<=0 or args.min_available_gib<=0:raise ValueError('Invalid benchmark limits')
    if args.command=='_worker':return worker(args,state,args.report)
    return run(args)


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('Benchmark cancelled; no production settings changed.');sys.exit(130)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
