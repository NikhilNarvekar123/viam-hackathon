import asyncio
import json
from pathlib import Path
import shutil
from unittest.mock import AsyncMock, patch

import pytest
import yaml
import main
from core.runtime_events import EventRuntime

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path):
    shutil.copytree(ROOT / 'config', tmp_path / 'config')
    return yaml.safe_load((tmp_path / 'config/config.yaml').read_text())


def test_reset_restarts_home_capture_and_inspection_not_just_old_pick_plan(tmp_path):
    from vlm.service import decide_wrist_objects
    from robot.overview import overview_stage
    config = setup(tmp_path)
    runtime = EventRuntime(config['runtime_events'])
    inspected = []
    async def inspect(*args, **kwargs):
        inspected.append(args[0]['output_dir'])
        result = await decide_wrist_objects(*args, **kwargs)
        if len(inspected) == 1:
            runtime.publish('scene_changed', 'overview')
        return result
    with patch.object(main, 'ROOT', tmp_path), \
         patch('vlm.service.decide_wrist_objects', side_effect=inspect), \
         patch('robot.overview.overview_stage', wraps=overview_stage) as overview:
        asyncio.run(main.run_overview_loop(config, fake=True, cycles=2, event_runtime=runtime))
    assert overview.await_count == 2
    assert len(set(inspected)) == 2
    reports = [json.loads((Path(p) / 'batch_pick.json').read_text()) for p in inspected]
    assert reports[0]['status'] == 'reset_requested' and not reports[0]['objects']
    assert reports[1]['status'] == 'batch_complete' and len(reports[1]['objects']) == 3


def test_event_runtime_does_not_bypass_operator_pause(tmp_path):
    config = setup(tmp_path)
    runtime = EventRuntime(config['runtime_events'])
    async def batch(*args, **kwargs):
        return (dict(output_dir=str(tmp_path), candidate_count=0, anomalous_count=0), [],
                dict(status='reset_requested', sorting_executed=False))
    with patch.object(main, 'ROOT', tmp_path), \
         patch('main.sys.stdin.isatty', return_value=True), \
         patch('robot.overview.overview_stage', new_callable=AsyncMock,
               return_value={'candidate_count': 0}) as overview, \
         patch('robot.external_box_pick.run_external_box_batch', side_effect=batch), \
         patch('main.wait_for_next_cycle', new_callable=AsyncMock, return_value=False) as pause:
        asyncio.run(main.run_overview_loop(config, fake=True, cycles=2,
                    pause_between_cycles=True, event_runtime=runtime))
    pause.assert_awaited_once()
    overview.assert_awaited_once()


def test_exhausted_batch_waits_for_event_instead_of_repeating_rejects(tmp_path):
    config = setup(tmp_path)
    runtime = EventRuntime(config['runtime_events'])
    with patch.object(main, 'ROOT', tmp_path), \
         patch.object(runtime, 'wait_for_reset', new_callable=AsyncMock,
                      side_effect=RuntimeError('idle boundary reached')) as idle:
        with pytest.raises(RuntimeError, match='idle boundary reached'):
            asyncio.run(main.run_overview_loop(config, fake=True, cycles=2, event_runtime=runtime))
    idle.assert_awaited_once()
    assert len(list(tmp_path.rglob('batch_pick.json'))) == 1


def test_unsupported_mode_refused_before_capture(tmp_path):
    config = setup(tmp_path)
    config['pick_target_source'] = 'picker'
    runtime = EventRuntime(config['runtime_events'])
    with patch('robot.overview.overview_stage', new_callable=AsyncMock) as overview:
        with pytest.raises(ValueError, match='pipeline-box'):
            asyncio.run(main.run_overview_loop(config, fake=True, cycles=1, event_runtime=runtime))
    overview.assert_not_awaited()
