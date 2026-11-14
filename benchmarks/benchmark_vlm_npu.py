#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from benchmark_anomalib_devices import verify

WORK = Path.home() / 'viam-device-bench'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check(state, model_hashes):
    verify(state)
    for path, expected in model_hashes.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('VLM asset changed: ' + path)


def worker(folder):
    path = folder / 'report.json'
    result = json.loads(path.read_text())
    source = json.loads(Path(result['source_report']).read_text())
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        import numpy as np
        from PIL import Image
        import openvino as ov
        import openvino_genai as genai
        result['openvino_version'] = ov.__version__
        result['genai_version'] = genai.__version__
        if not any(d.split('.')[0] == 'NPU' for d in ov.Core().available_devices):
            raise RuntimeError('NPU is not available')
        result['stage'] = 'load_compile'
        save(path, result)
        print('NPU: loading/compiling existing VLM; GPU service is untouched...', flush=True)
        start = time.perf_counter()
        pipeline = genai.VLMPipeline(source['service_status']['model_path'], 'NPU', CACHE_DIR='')
        result['load_compile_ms'] = (time.perf_counter() - start) * 1000
        image_path = Path(result['source_report']).parent / 'input.png'
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert('RGB'), dtype=np.uint8)
        tensor = ov.Tensor(np.ascontiguousarray(pixels[None]))
        result['stage'] = 'first_generation'
        save(path, result)
        print('NPU: generating ONE answer...', flush=True)
        start = time.perf_counter()
        output = pipeline.generate(source['prompt'], images=[tensor],
                                   max_new_tokens=source['max_new_tokens'],
                                   do_sample=source['do_sample'])
        result['first_generation_ms'] = (time.perf_counter() - start) * 1000
        result['answer'] = output.texts[0] if hasattr(output, 'texts') and output.texts else str(output)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = 'error'
        result['error'] = f'{type(exc).__name__}: {exc}'
        print('NPU failed:', result['error'], flush=True)
    finally:
        save(path, result)


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main():
    state = json.loads((WORK / 'state.json').read_text())
    source_path = None
    for path in sorted(WORK.glob('VLM_GPU_*/report.json'), reverse=True):
        if json.loads(path.read_text()).get('status') == 'complete':
            source_path = path
            break
    if source_path is None:
        raise RuntimeError('No completed GPU VLM benchmark found')
    source = json.loads(source_path.read_text())
    hashes = json.loads((source_path.parent / 'vlm_model_hashes.json').read_text())
    if not hashes:
        raise RuntimeError('Missing VLM asset baseline')
    print('Checking production and VLM asset hashes; reading only...', flush=True)
    check(state, hashes)
    if sha(source_path.parent / 'input.png') != source['payload_sha256']:
        raise RuntimeError('Saved GPU input image changed')
    folder = WORK / ('VLM_NPU_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + str(os.getpid()))
    folder.mkdir()
    temp = folder / 'tmp'
    temp.mkdir()
    report_path = folder / 'report.json'
    save(report_path, dict(status='starting', stage='imports', requested_device='NPU',
         source_report=str(source_path), persistent_openvino_cache=False,
         execution_placement='Requested NPU; per-component placement not independently verified',
         test='One direct GenAI generation, not warmed latency or Viam service latency'))
    env = os.environ.copy()
    env.update(TMPDIR=str(temp), XDG_CACHE_HOME=str(folder / 'cache'), PYTHONDONTWRITEBYTECODE='1')
    process = None
    reason = None
    try:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(folder)],
                                   cwd=WORK, env=env, start_new_session=True)
        start = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - start > 300:
                reason = '300-second timeout; not proof of incompatibility'
            if shutil.disk_usage(WORK).free < 3 * 1024**3:
                reason = 'Free disk below 3 GiB'
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:') and int(line.split()[1]) * 1024 < 8 * 1024**3:
                    reason = 'Available RAM below 8 GiB'
            if reason:
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        reason = 'Cancelled by operator'
    finally:
        try:
            if process is not None:
                stop(process)
        finally:
            if process is None or process.poll() is not None:
                shutil.rmtree(temp, ignore_errors=True)
                shutil.rmtree(folder / 'cache', ignore_errors=True)
            result = json.loads(report_path.read_text())
            if reason:
                result.update(status='stopped', reason=reason)
            elif result['status'] == 'starting':
                result.update(status='aborted', reason='Worker exited without completing')
            print('Rechecking production and VLM asset hashes...', flush=True)
            try:
                check(state, hashes)
                result['production_and_vlm_hashes'] = 'UNCHANGED'
            except Exception as exc:
                result.update(status='integrity_check_failed', integrity_error=str(exc))
            save(report_path, result)
            print(json.dumps(result, indent=2))
            print('Report:', report_path)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]))
    else:
        main()
