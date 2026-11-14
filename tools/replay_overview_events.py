"""Replay saved overview frames offline; no camera, model, or motion access.

Print JSONL detection records. `would_publish` is a simulation, not a runtime
queue write. See monitors/OVERVIEW_DETECTION.md for the manifest format.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from monitors.overview_detection import DetectionSettings, OverviewChangeDetector


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def load_mask(path):
    with Image.open(path) as image:
        values = np.asarray(image.convert('L'))
    if not np.isin(values, [0, 255]).all():
        raise ValueError('Masks must contain only black (0) and white (255) pixels')
    return values == 255


def replay(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    workspace = load_mask(root / manifest['workspace_mask'])
    excluded = load_mask(root / manifest['excluded_mask']) if manifest.get('excluded_mask') else None
    detector = OverviewChangeDetector(
        workspace, excluded, DetectionSettings(**manifest.get('settings', {})))
    for index, sample in enumerate(manifest['samples']):
        frame = load_rgb(root / sample['image'])
        clear = sample['clear_view']  # Required, never inferred from image or robot idleness.
        operation = sample.get('operation', 'observe')
        started = time.perf_counter()
        if operation == 'reference':
            detector.set_reference(frame, clear_view=clear)
            result = {'status': 'reference', 'would_publish': None}
        elif operation == 'expected':
            detector.accept_expected_change(frame, load_mask(root / sample['expected_mask']),
                                            clear_view=clear)
            result = {'status': 'expected_patch_accepted', 'would_publish': None}
        elif operation == 'observe':
            detection = detector.observe(frame, clear_view=clear)
            result = asdict(detection)
            result['would_publish'] = 'scene_changed' if detection.publish_scene_changed else None
        else:
            raise ValueError(f'Unknown replay operation: {operation}')
        result.update(sample=index, operation=operation, offline=True,
                      processing_ms=(time.perf_counter() - started) * 1000)
        yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    try:
        for record in replay(args.manifest):
            print(json.dumps(record, allow_nan=False), flush=True)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Replay failed: {exc}\n')


if __name__ == '__main__':
    main()
