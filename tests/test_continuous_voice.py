import asyncio
import json
from pathlib import Path
import shutil
import time
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from core.runtime_events import EventRuntime
from core.voice_control import VoiceControl, parse_voice_command
from robot.continuous_pick import run_continuous, HomeMonitor
from robot.external_box_pick import ExternalBoxBatch
from robot.inspection_pick import execute_result

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path):
    shutil.copytree(ROOT/'config', tmp_path/'config')
    config = yaml.safe_load((tmp_path/'config/config.yaml').read_text())
    config['voice_control']['enabled'] = True
    config['runtime_events']['idle_poll_s'] = .001
    settings = yaml.safe_load((tmp_path/'config/continuous_pick.yaml').read_text())
    settings.update(idle_poll_s=.001, sample_interval_s=.001, settle_s=0)
    runtime = EventRuntime(config['runtime_events'], fake=True)
    sequence = 0
    def say(text):
        nonlocal sequence
        sequence += 1
        if text in ('start', 'stop', 'hello'):text = 'robot ' + text
        if text.startswith('new policy '):text += ' over'
        return runtime.publish('voice_command', 'audio', json.dumps(dict(
            parse_voice_command(text), sequence=sequence, heard_at=time.monotonic())))
    return config, settings, runtime, say


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 5)


def decisions():
    return [dict(id=1, final_action='accept'), dict(id=2, final_action='reject'), dict(id=3, final_action='hold')]


def test_initially_stopped_and_gestures_never_start_picking(tmp_path):
    config, settings, runtime, say = setup(tmp_path)
    records = []
    log = VoiceControl.log
    def logged(self, event, **fields):
        records.append((event, fields));log(self, event, **fields)
    async def exercise():
        stop = asyncio.Event()
        task = asyncio.create_task(run_continuous(config, tmp_path, fake=True, settings=settings,
                                                 event_runtime=runtime, stop=stop))
        try:
            await until(lambda: records)
            runtime.publish('audio_ready', 'audio')
            say('hello')
            await until(lambda: any(e == 'gesture_completed' for e, _ in records))
            say('dance')
            await until(lambda: sum(e == 'gesture_completed' for e, _ in records) == 2)
            runtime.publish('scene_changed', 'overview')  # Cannot resume a voice-stopped loop.
            await asyncio.sleep(.02)
            stop.set()
            result = await asyncio.wait_for(task, 5)
            assert result['batches'] == 0
        finally:
            stop.set()
            await asyncio.gather(task, return_exceptions=True)
    with patch.object(VoiceControl, 'log', logged), \
         patch.object(ExternalBoxBatch, '__aenter__', new_callable=AsyncMock) as enter, \
         patch.object(ExternalBoxBatch, 'home_for_check', new_callable=AsyncMock) as home, \
         patch('robot.continuous_pick.decide_wrist_objects', new_callable=AsyncMock) as infer:
        asyncio.run(exercise())
    enter.assert_not_awaited();home.assert_not_awaited();infer.assert_not_awaited()
    assert [fields['gesture'] for event, fields in records if event == 'gesture_completed'] == ['hello', 'dance']


@pytest.mark.parametrize('utterance,ack', [
    ('stop', 'stopped'), ('hello', 'gesture_completed'), ('dance', 'gesture_completed'),
])
def test_command_during_delivery_finishes_then_stays_stopped_until_fresh_start(tmp_path, utterance, ack):
    config, settings, runtime, say = setup(tmp_path)
    records, phase, snapshots = [], [], []
    log, home = VoiceControl.log, ExternalBoxBatch.home_for_check
    async def homing(self):
        phase.append('home');await home(self)
    def logged(self, event, **fields):
        records.append(event);log(self, event, **fields)
    async def deliver(st, result, skip):
        phase.append('delivery_started')
        say(utterance)
        say('start')  # Buffered before acknowledgement: must NOT undo the stop.
        out = await execute_result(st, result, skip)
        phase.append('delivery_finished')
        return out
    stop = asyncio.Event()
    async def inspect(wrist, effective, *args, **kwargs):
        snapshots.append((wrist['output_dir'], effective))
        if len(snapshots) == 2:
            stop.set()
        return decisions()
    async def exercise():
        runtime.publish('audio_ready', 'audio');say('start')
        task = asyncio.create_task(run_continuous(config, tmp_path, fake=True, settings=settings,
                                                 event_runtime=runtime, stop=stop))
        try:
            await until(lambda: ack in records)
            assert phase == ['home', 'delivery_started', 'delivery_finished']
            await asyncio.sleep(.02)
            assert len(snapshots) == 1 and phase.count('home') == 1
            say('start')
            result = await asyncio.wait_for(task, 5)
            assert len(snapshots) == 2 and snapshots[0][0] != snapshots[1][0]
            assert result['state']['placed']['box'] == 1
        finally:
            stop.set();await asyncio.gather(task, return_exceptions=True)
    with patch.object(VoiceControl, 'log', logged), \
         patch.object(ExternalBoxBatch, 'home_for_check', homing), \
         patch('robot.external_box_pick.execute_result', side_effect=deliver), \
         patch('robot.continuous_pick.decide_wrist_objects', side_effect=inspect):
        asyncio.run(exercise())
    report = json.loads((Path(snapshots[0][0])/'batch_pick.json').read_text())
    assert len(report['objects']) == 1 and report['pending_crop_ids'] == [2, 3]
    if utterance.startswith('new policy'):
        effective = snapshots[1][1]
        assert effective['inspection_policy'] == {'version': 1, 'text': 'reject red parts'}
        assert not effective['stage1_fast_pass'] and 'reject red parts' in effective['prompt']
        assert snapshots[0][1]['inspection_policy']['version'] == 0


@pytest.mark.parametrize('reset_utterance', ['start', 'new policy reject red parts'])
def test_running_reset_discards_pending_targets_after_delivery(tmp_path, reset_utterance):
    config, settings, runtime, say = setup(tmp_path)
    stop = asyncio.Event();inspections = []
    async def deliver(st, result, skip):
        say(reset_utterance)
        return await execute_result(st, result, skip)
    async def inspect(wrist, *args, **kwargs):
        inspections.append(wrist['output_dir'])
        if len(inspections) == 2:stop.set()
        return decisions()
    runtime.publish('audio_ready', 'audio');say('start')
    with patch('robot.external_box_pick.execute_result', side_effect=deliver), \
         patch('robot.continuous_pick.decide_wrist_objects', side_effect=inspect):
        result = asyncio.run(asyncio.wait_for(run_continuous(config, tmp_path, fake=True,
            settings=settings, event_runtime=runtime, stop=stop), 5))
    assert result['state']['placed']['box'] == 1 and len(inspections) == 2
    first = json.loads((Path(inspections[0])/'batch_pick.json').read_text())
    assert len(first['objects']) == 1 and first['status'] == 'reset_requested'


def test_stop_during_inference_prevents_any_pick(tmp_path):
    config, settings, runtime, say = setup(tmp_path)
    stop = asyncio.Event();records = [];calls = []
    log = VoiceControl.log
    def logged(self, event, **fields):
        records.append(event);log(self, event, **fields)
    async def inspect(wrist, *args, **kwargs):
        calls.append(wrist['output_dir']);say('stop');return decisions()
    async def exercise():
        runtime.publish('audio_ready', 'audio');say('start')
        task = asyncio.create_task(run_continuous(config, tmp_path, fake=True, settings=settings,
                                                 event_runtime=runtime, stop=stop))
        try:
            await until(lambda: 'stopped' in records)
            await asyncio.sleep(.02)
            assert len(calls) == 1
            stop.set();await asyncio.wait_for(task, 5)
        finally:
            stop.set();await asyncio.gather(task, return_exceptions=True)
    with patch.object(VoiceControl, 'log', logged), \
         patch('robot.continuous_pick.decide_wrist_objects', side_effect=inspect), \
         patch('robot.external_box_pick.execute_result', side_effect=AssertionError('No delivery allowed')):
        asyncio.run(exercise())


def test_voice_start_cannot_clear_a_runtime_fault(tmp_path):
    config, settings, runtime, say = setup(tmp_path)
    for _ in range(config['runtime_events']['queue_capacity']+1):
        runtime.publish('notice', 'audio')
    say('start')
    with patch.object(ExternalBoxBatch, 'home_for_check', new_callable=AsyncMock) as home:
        with pytest.raises(RuntimeError, match='overflow'):
            asyncio.run(run_continuous(config, tmp_path, fake=True, settings=settings, event_runtime=runtime))
    home.assert_not_awaited()


def test_full_area_putback_does_not_claim_expected_removal(tmp_path):
    config, settings, _, _ = setup(tmp_path)
    config['voice_control']['enabled'] = False
    stop = asyncio.Event();samples = [];sample = HomeMonitor.sample
    async def sampled(self, *args, **kwargs):
        result = await sample(self, *args, **kwargs)
        samples.append(result)
        if len(samples) == 5:stop.set()
        return result
    with patch('pick_bolt.free_place', return_value=None), \
         patch.object(HomeMonitor, 'sample', sampled), \
         patch('robot.continuous_pick.decide_wrist_objects', new_callable=AsyncMock, return_value=decisions()):
        result = asyncio.run(asyncio.wait_for(run_continuous(config, tmp_path, fake=True,
            settings=settings, stop=stop), 5))
    assert result['batches'] == 1 and result['state']['placed']['box'] == 0
    report = json.loads(next(tmp_path.rglob('batch_pick.json')).read_text())
    assert all(not row.get('removed_from_source') for row in report['objects'])
    assert all(s['status'] == 'unchanged' for s in samples)
