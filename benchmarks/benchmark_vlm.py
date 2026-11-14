#!/usr/bin/env python3
"""Sequential GPU VLM service benchmark. Saved crop only; no camera or motion calls."""
import sys
sys.dont_write_bytecode = True
import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import yaml
from dotenv import dotenv_values
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

WORK = Path.home() / 'viam-device-bench'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parse(raw):
    try:
        text = raw.strip()
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0] in ('```json', '```') and lines[-1] == '```':
            text = '\n'.join(lines[1:-1])
        value = json.loads(text)
        valid = (isinstance(value, dict) and list(value) == ['observation', 'defect']
                 and isinstance(value['observation'], str)
                 and 0 < len(value['observation'].strip()) and len(value['observation']) <= 1000
                 and type(value['defect']) is bool)
        if valid:
            return True, 'reject' if value['defect'] else 'accept'
    except (ValueError, TypeError, AttributeError):
        pass
    return False, 'hold'


def stats(values):
    if not values:
        return None
    return {'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
            'mean': float(np.mean(values)), 'minimum': float(min(values)), 'maximum': float(max(values))}


async def main():
    # Import ONLY our standalone hash verifier, not production Python modules.
    spec = importlib.util.spec_from_file_location('bench_guard', WORK / 'benchmark_anomalib_devices.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    baseline = json.loads((WORK / 'state.json').read_text())
    guard.verify(baseline)
    root = Path(baseline['root'])
    settings = yaml.safe_load((root / 'config/vlm_service.yaml').read_text())
    if not settings.get('enabled') or settings.get('response_schema') != 'grounded_inspection':
        raise ValueError('Expected enabled service and grounded_inspection schema; no settings changed')

    image_path = Path(baseline['image'])
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    if image.width != image.height:
        raise ValueError('Expected a saved square production crop')
    size = settings['image_size_px']
    image = Image.fromarray(cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_LANCZOS4))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    payload = buffer.getvalue()
    question = settings['prompt'] + (
        '\nThe supplied image is the isolated square crop of the single centered target. '
        'Neighbor contours were replaced with tabletop background. It contains no annotation or heatmap. '
        'Inspect only the physical target, not the filled background.')
    request = {'command': 'ask', 'question': question,
               'image_b64': base64.b64encode(payload).decode('ascii'),
               'max_new_tokens': settings['max_new_tokens'], 'do_sample': False}

    credentials = dotenv_values(root / '.env')
    keys = ('VIAM_ADDRESS', 'VIAM_API_KEY', 'VIAM_API_KEY_ID')
    credentials = {key: os.environ.get(key) or credentials.get(key) for key in keys}
    if not all(credentials.values()):
        raise ValueError('Missing Viam credentials; do not paste credentials into chat')

    def safe_error(exc):
        text = f'{type(exc).__name__}: {exc}'
        for value in credentials.values():
            text = text.replace(value, '[redacted]')
        return text[:2000]

    folder = WORK / datetime.now(timezone.utc).strftime(f'VLM_GPU_%Y%m%dT%H%M%S_{os.getpid()}')
    folder.mkdir()
    (folder / 'input.png').write_bytes(payload)
    report = {'benchmark': 'Existing GPU VLM service; sequential requests on ONE saved crop',
              'cold_model_start_measured': False, 'camera_calls': False, 'motion_calls': False,
              'image': str(image_path), 'image_sha256': sha(image_path),
              'payload_sha256': hashlib.sha256(payload).hexdigest(), 'prompt': question,
              'max_new_tokens': settings['max_new_tokens'], 'do_sample': False,
              'planned_timed_requests': 30, 'additional_warmups': 3, 'samples': []}
    machine = None
    model_hashes = {}
    try:
        options = RobotClient.Options.with_api_key(
            api_key=credentials['VIAM_API_KEY'], api_key_id=credentials['VIAM_API_KEY_ID'])
        machine = await RobotClient.at_address(credentials['VIAM_ADDRESS'], options)
        service = VisionClient.from_robot(machine, settings['service_name'])
        status = await service.do_command({'command': 'status'}, timeout=10)
        if status.get('state') != 'ready' or str(status.get('device')).split('.')[0] != 'GPU':
            raise ValueError('Service is not ready on GPU; benchmark stopped without changes')
        report['service_status'] = {key: status.get(key) for key in
                                    ('state', 'device', 'requested_device', 'model_path')}
        model_dir = Path(status['model_path'])
        if not model_dir.is_dir():
            raise ValueError('Service model directory is inaccessible; cannot snapshot weights')
        files = [path for path in model_dir.rglob('*') if path.is_file()
                 and '.cache' not in path.parts
                 and path.suffix in ('.xml', '.bin', '.json', '.safetensors', '.model', '.txt')]
        if not any(path.suffix == '.bin' for path in files):
            raise ValueError('No OpenVINO model weights found; stopping')
        print('Hashing existing VLM assets; reading only, no weight copies...', flush=True)
        model_hashes = {str(path): sha(path) for path in files}
        save(folder / 'vlm_model_hashes.json', model_hashes)
        phases = ['first'] + ['warmup'] * 3 + ['timed'] * 30
        for index, phase in enumerate(phases, 1):
            sample = {'index': index, 'phase': phase}
            start = time.perf_counter()
            try:
                response = await service.do_command(request, timeout=settings['request_timeout_s'])
                sample['request_ms'] = (time.perf_counter() - start) * 1000
                sample['device'] = response.get('device')
                if str(sample['device']).split('.')[0] != 'GPU':
                    raise ValueError('Response does not confirm GPU; refusing mixed-device measurements')
                raw = response.get('answer', '')
                valid, action = parse(raw)
                sample.update(valid=valid, action=action, raw_answer=raw,
                              service_seconds=response.get('seconds'))
                print(f'{index}/34 {phase}: {sample["request_ms"]:.1f} ms, {action}, valid={valid}', flush=True)
            except Exception as exc:
                sample.update(error=safe_error(exc), valid=False,
                              request_ms=(time.perf_counter() - start) * 1000)
                report['samples'].append(sample)
                raise  # Never retry a timeout: server inference may still be running.
            report['samples'].append(sample)
            save(folder / 'report.json', report)
        after = await service.do_command({'command': 'status'}, timeout=10)
        if after.get('model_path') != status.get('model_path') or after.get('device') != status.get('device'):
            raise ValueError('Service model/device changed during benchmark')
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='stopped', error=safe_error(exc))
        print('Stopped:', report['error'], flush=True)
    finally:
        try:
            if machine is not None:
                await machine.close()
        finally:
            try:
                guard.verify(baseline)
                if any(not Path(p).is_file() or sha(p) != h for p, h in model_hashes.items()):
                    raise RuntimeError('VLM asset hashes changed')
                report['production_hashes'] = 'UNCHANGED'
                report['vlm_asset_hashes'] = 'UNCHANGED' if model_hashes else 'not captured'
            except Exception as exc:
                report['integrity_error'] = safe_error(exc)
                report['status'] = 'integrity_check_failed'
            timed = [s for s in report['samples'] if s['phase'] == 'timed']
            completed = [s for s in timed if 'error' not in s]
            valid = [s for s in completed if s['valid']]
            actions = Counter(s['action'] for s in valid)
            report.update(timed_attempted=len(timed), timed_completed=len(completed),
                          valid_responses=len(valid),
                          valid_response_rate=len(valid) / len(timed) if timed else None,
                          completed_request_latency_ms=stats([s['request_ms'] for s in completed]),
                          valid_action_counts=dict(actions),
                          valid_action_consistency=max(actions.values()) / len(valid) if valid else None)
            first = next((s for s in report['samples'] if s['phase'] == 'first'), None)
            report['first_request_ms'] = first.get('request_ms') if first else None
            save(folder / 'report.json', report)
            print(json.dumps({k: v for k, v in report.items()
                              if k not in ('samples', 'prompt')}, indent=2))
            print('Report:', folder / 'report.json')


if __name__ == '__main__':
    asyncio.run(main())
