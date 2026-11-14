import json

import numpy as np
from PIL import Image
import pytest

from tools.replay_overview_events import replay


def fixture_manifest(tmp_path):
    baseline = np.full((40, 60, 3), 180, dtype=np.uint8)
    changed = baseline.copy()
    changed[20:28, 30:38] = 0
    Image.fromarray(baseline).save(tmp_path / 'before.png')
    Image.fromarray(changed).save(tmp_path / 'after.png')
    Image.new('L', (60, 40), 255).save(tmp_path / 'workspace.png')
    manifest = {
        'workspace_mask': 'workspace.png',
        'samples': [
            {'image': 'before.png', 'operation': 'reference', 'clear_view': True},
            {'image': 'after.png', 'clear_view': False},
            *[{'image': 'after.png', 'clear_view': True} for _ in range(5)],
        ],
    }
    return manifest


def save(tmp_path, manifest):
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(manifest))
    return path


def test_offline_replay_records_one_hypothetical_reset(tmp_path):
    rows = list(replay(save(tmp_path, fixture_manifest(tmp_path))))
    assert [r['status'] for r in rows] == [
        'reference', 'obscured', 'pending', 'pending', 'changed', 'latched', 'latched']
    assert [r['would_publish'] for r in rows].count('scene_changed') == 1
    assert all(row['offline'] and row['processing_ms'] >= 0 for row in rows)
    json.dumps(rows, allow_nan=False)


def test_replay_expected_patch(tmp_path):
    manifest = fixture_manifest(tmp_path)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[20:28, 30:38] = 255
    Image.fromarray(mask).save(tmp_path / 'expected.png')
    manifest['samples'].insert(1, {'image': 'after.png', 'operation': 'expected',
                                   'expected_mask': 'expected.png', 'clear_view': True})
    rows = list(replay(save(tmp_path, manifest)))
    assert rows[1]['status'] == 'expected_patch_accepted'
    assert not any(row['would_publish'] for row in rows)


def test_replay_rejects_nonbinary_mask(tmp_path):
    manifest = fixture_manifest(tmp_path)
    Image.new('L', (60, 40), 100).save(tmp_path / 'workspace.png')
    with pytest.raises(ValueError, match='Masks'):
        list(replay(save(tmp_path, manifest)))


def test_replay_requires_explicit_visibility(tmp_path):
    manifest = fixture_manifest(tmp_path)
    del manifest['samples'][0]['clear_view']
    with pytest.raises(KeyError, match='clear_view'):
        list(replay(save(tmp_path, manifest)))


def test_replay_rejects_unknown_operation(tmp_path):
    manifest = fixture_manifest(tmp_path)
    manifest['samples'][0]['operation'] = 'move_arm'
    with pytest.raises(ValueError, match='Unknown replay'):
        list(replay(save(tmp_path, manifest)))
