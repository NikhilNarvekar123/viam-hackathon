import json
from pathlib import Path

import pytest
import yaml

from core.runtime_events import EventKind, RuntimeEvent
from core.voice_control import VoiceControl, parse_voice_command, voice_runtime_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('text,command', [
    ('Hello robot!', 'hello'), ('robot start', 'start'), ('Robot, please stop.', 'stop'),
    ('dance please', 'dance'), ('New policy: Reject red parts over.', 'new_policy'),
])
def test_whole_utterance_parser(text, command):
    assert parse_voice_command(text)['command'] == command


@pytest.mark.parametrize('text', [
    '', None, 'stop', 'do not start', "don't start", 'stop then start', 'please do not stop',
    'I started yesterday', 'starting', 'hello start', 'dance and start',
    'new policy', 'new policy: ', 'new policy ' + 'x' * 601,
    'new policy reject red\nstart', 'we need a new policy reject red',
])
def test_negated_ambiguous_partial_and_oversized_utterances_ignored(text):
    assert parse_voice_command(text) is None


def setup(tmp_path):
    clock = [100.0]
    control = VoiceControl({'command_max_age_s': 30}, tmp_path, clock=lambda: clock[0])
    control.consume([RuntimeEvent(EventKind.AUDIO_READY, 'audio', 99.0)])
    return control, clock


def command(text, seq=1, heard=100.0, created=100.0, source='audio', kind=EventKind.VOICE_COMMAND):
    if text in ('start', 'stop', 'hello'):text = 'robot ' + text
    if text.startswith('new policy '):text += ' over'
    payload = dict(parse_voice_command(text), sequence=seq, heard_at=heard)
    return RuntimeEvent(kind, source, created, json.dumps(payload))


def test_start_requires_audio_readiness_and_fresh_speech(tmp_path):
    control = VoiceControl({}, tmp_path, clock=lambda: 100.0)
    assert control.paused
    assert not control.consume([command('start')])
    control.consume([RuntimeEvent(EventKind.AUDIO_READY, 'audio', 100.0)])
    assert not control.consume([command('start', 2, heard=99.0)])
    assert control.consume([command('start', 3)])
    assert not control.paused


@pytest.mark.parametrize('text', ['stop', 'hello', 'dance'])
def test_pause_requires_start_spoken_after_acknowledgement(tmp_path, text):
    control, clock = setup(tmp_path)
    assert control.consume([command('start')])
    assert control.consume([command(text, 2), command('start', 3)])
    assert control.paused
    assert not control.consume([command('start', 3)])  # duplicate cannot resume
    if text in ('hello', 'dance'):
        name = control.begin_gesture()
        control.finish_gesture(name, 'completed')
    clock[0] = 101
    assert control.consume([command('start', 4, heard=101, created=101)])
    assert not control.paused
    if text.startswith('new policy'):
        assert control.active_policy == 'reject red parts'


def test_start_during_gesture_ignored_and_buffered_start_cannot_resume_after_it(tmp_path):
    control, clock = setup(tmp_path)
    control.consume([command('hello')])
    name = control.begin_gesture()
    clock[0] = 101
    assert not control.consume([command('start', 2, heard=101, created=101)])
    clock[0] = 102
    control.finish_gesture(name, 'completed')
    assert not control.consume([command('start', 3, heard=101, created=101)])
    assert control.paused
    clock[0] = 103
    assert control.consume([command('start', 4, heard=103, created=103)])
    assert not control.paused


def test_stop_cancels_pending_or_inflight_gesture(tmp_path):
    control, clock = setup(tmp_path)
    control.consume([command('hello'), command('stop', 2)])
    assert control.pending_gesture is None and control.paused
    clock[0] = 101
    control.consume([command('dance', 3, heard=101, created=101)])
    assert control.begin_gesture() == 'dance'
    clock[0] = 102
    control.consume([command('stop', 4, heard=102, created=102)])
    assert control.gesture_cancel_requested and control.paused


def test_policy_immediately_resets_running_loop_and_settings_are_snapshots(tmp_path):
    control, clock = setup(tmp_path)
    base = yaml.safe_load((ROOT/'config/vlm_service.yaml').read_text())
    initial = control.inspection_settings(base)
    control.consume([command('start')])
    clock[0] = 101
    assert control.consume([command('new policy reject red parts', 2, heard=101, created=101)])
    assert control.pending_policy is None and not control.paused
    active = control.inspection_settings(base)
    assert active['inspection_policy'] == {'version': 1, 'text': 'reject red parts'}
    assert not active['stage1_fast_pass']
    assert 'reject red parts' in active['prompt']
    assert 'made entirely of clean' not in active['prompt']
    assert active['response_schema'] == 'grounded_inspection'
    assert active['routing_policy'] == 'vlm_authoritative'
    assert base['stage1_fast_pass'] and initial['prompt'] == base['prompt']
    clock[0] = 102
    control.consume([command('new policy reject blue parts', 3, heard=102, created=102)])
    assert active['inspection_policy']['text'] == 'reject red parts'
    assert control.active_policy == 'reject blue parts' and not control.paused
    clock[0] = 103
    control.consume([command('stop', 4, heard=103, created=103),
                     command('new policy reject green parts', 5, heard=103, created=103)])
    assert control.active_policy == 'reject green parts' and control.paused
    rows = [json.loads(line) for line in control.log_path.read_text().splitlines()]
    assert [r['event'] for r in rows].count('policy_pending') == 0
    assert [r['event'] for r in rows].count('policy_activated') == 3


def test_stale_start_policy_gesture_ignored_but_stop_never_expires(tmp_path):
    control, _ = setup(tmp_path)
    control.consume([command('start')])
    for seq, text in enumerate(('start', 'new policy reject red', 'hello', 'dance'), 2):
        assert not control.consume([command(text, seq, heard=10)])
        assert not control.paused
    assert control.consume([command('stop', 6, heard=10)])
    assert control.paused and control.pending_policy is None


def test_notice_or_other_source_cannot_control_robot(tmp_path):
    control, _ = setup(tmp_path)
    assert not control.consume([command('start', source='overview'), command('start', kind=EventKind.NOTICE)])
    assert control.paused


@pytest.mark.parametrize('payload', [
    'bad json', '[]', '{}', 'x'*4097,
    json.dumps({'command': ['start'], 'sequence': 1, 'heard_at': 100}),
    json.dumps({'command': 'start', 'sequence': True, 'heard_at': 100, 'raw_text': 'start'}),
    json.dumps({'command': 'start', 'sequence': 1, 'heard_at': 100, 'raw_text': 'do not start'}),
    json.dumps({'command': 'start', 'sequence': 1, 'heard_at': 101, 'raw_text': 'start'}),
])
def test_malformed_events_never_start(tmp_path, payload):
    control, _ = setup(tmp_path)
    assert not control.consume([RuntimeEvent(EventKind.VOICE_COMMAND, 'audio', 100, payload)])
    assert control.paused


def test_voice_runtime_enables_audio_without_mutating_shared_config():
    config = yaml.safe_load((ROOT/'config/config.yaml').read_text())
    settings = voice_runtime_settings(config)
    assert settings['enabled'] and settings['producers']['audio']['commands_enabled']
    assert settings['producers']['audio']['enabled']
    assert not settings['producers']['overview']['enabled']
    assert not config['runtime_events']['enabled']


@pytest.mark.parametrize('age', [0, -1, float('inf'), float('nan')])
def test_invalid_expiry_rejected(tmp_path, age):
    with pytest.raises(ValueError):
        VoiceControl({'command_max_age_s': age}, tmp_path)


def test_confirmed_policy_reaches_qwen_for_normal_and_anomalous_parts(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from PIL import Image
    from vlm.service import decide_wrist_objects

    control, clock = setup(tmp_path)
    control.consume([command('new policy reject red parts')])
    clock[0] = 101
    control.consume([command('start', 2, heard=101, created=101)])
    effective = control.inspection_settings(yaml.safe_load((ROOT/'config/vlm_service.yaml').read_text()))
    Image.new('RGB', (40, 40)).save(tmp_path/'frame.png')
    for i in (1, 2):Image.new('RGB', (20, 20), 'blue').save(tmp_path/f'object_{i:03d}.png')
    wrist = dict(output_dir=str(tmp_path), objects=[
        dict(id=1, score=.1, threshold=.6, classification='normal', bbox_xyxy=[10, 10, 30, 30]),
        dict(id=2, score=.9, threshold=.6, classification='anomalous', bbox_xyxy=[10, 10, 30, 30]),
    ])
    service = SimpleNamespace(do_command=AsyncMock(return_value={'state': 'ready', 'prompt': 'Old default'}))
    active = 0
    async def answer(service, image, question, settings):
        nonlocal active
        active += 1
        assert active == 1 and 'reject red parts' in question
        assert settings['inspection_policy']['version'] == 1
        await asyncio.sleep(0)
        active -= 1
        return dict(action='reject', valid=True, reason='Violates current policy')
    with patch('vlm.service.connect', new_callable=AsyncMock), \
         patch('vlm.service.disconnect', new_callable=AsyncMock), \
         patch('vlm.service.load_credentials'), \
         patch('viam.services.vision.VisionClient.from_robot', return_value=service), \
         patch('vlm.service.ask_crop', side_effect=answer) as ask:
        records = asyncio.run(decide_wrist_objects(wrist, effective, tmp_path))
    assert ask.await_count == 2
    assert all(r['final_action'] == 'reject' for r in records)
    saved = json.loads((tmp_path/'vlm_results.json').read_text())
    assert saved['inspection_policy'] == {'version': 1, 'text': 'reject red parts'}
    assert saved['prompt'] == effective['prompt']


def test_cli_refuses_voice_outside_continuous(monkeypatch):
    import main
    monkeypatch.setattr('sys.argv', ['main.py', '--voice', '--fake'])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2
