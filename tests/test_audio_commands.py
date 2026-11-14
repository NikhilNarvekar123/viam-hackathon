import asyncio
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from core.runtime_events import EventKind, ProducerContext
from monitors import audio


def context(**settings):
    return ProducerContext(publish=Mock(return_value=True), stop=threading.Event(),
                           settings=dict(print_transcripts=False, **settings), fake=False)


def test_listener_publishes_readiness_and_typed_command_then_closes_connection(tmp_path):
    ctx = context(commands_enabled=True, step_seconds=2, device='CPU')
    events = []
    def publish(kind, detail=''):
        events.append((kind, detail))
        if kind == EventKind.VOICE_COMMAND:ctx.stop.set()
        return True
    ctx = ProducerContext(publish, ctx.stop, ctx.settings, False)
    async def chunks():
        yield SimpleNamespace(audio=SimpleNamespace(audio_data=np.full(32000, 1000, np.int16).tobytes()))
    mic = SimpleNamespace(get_properties=AsyncMock(return_value=SimpleNamespace(sample_rate_hz=16000, num_channels=1)),
                          get_audio=AsyncMock(side_effect=lambda **kwargs: chunks()))
    pipe = SimpleNamespace(generate=Mock(return_value=SimpleNamespace(texts=['Start.'])))
    module = SimpleNamespace(AudioIn=SimpleNamespace(from_robot=Mock(return_value=mic)))
    with patch.dict(sys.modules, {'viam.components.audio_in': module}), \
         patch('robot.viam_hw.load_credentials', return_value={}), \
         patch('robot.viam_hw.connect', new_callable=AsyncMock), \
         patch('robot.viam_hw.disconnect', new_callable=AsyncMock) as close, \
         patch.object(audio, '_load_whisper_pipeline', return_value=(pipe, object(), 'fake-whisper')) as load:
        asyncio.run(audio._listen_loop(ctx, tmp_path))
    assert [e[0] for e in events] == [EventKind.AUDIO_READY, EventKind.VOICE_COMMAND]
    payload = json.loads(events[1][1])
    assert payload['command'] == 'start' and payload['sequence'] == 1 and payload['heard_at'] > 0
    assert len(pipe.generate.call_args.args[0]) == 32000
    assert np.isclose(pipe.generate.call_args.args[0][0], 1000/32768)
    load.assert_called_once();close.assert_awaited_once()


def test_microphone_failure_propagates_and_disconnects(tmp_path):
    ctx = context()
    mic = SimpleNamespace(get_properties=AsyncMock(side_effect=RuntimeError('mic unavailable')))
    module = SimpleNamespace(AudioIn=SimpleNamespace(from_robot=Mock(return_value=mic)))
    with patch.dict(sys.modules, {'viam.components.audio_in': module}), \
         patch('robot.viam_hw.load_credentials', return_value={}), \
         patch('robot.viam_hw.connect', new_callable=AsyncMock), \
         patch('robot.viam_hw.disconnect', new_callable=AsyncMock) as close, \
         patch.object(audio, '_load_whisper_pipeline', return_value=(object(), object(), 'fake')):
        with pytest.raises(RuntimeError, match='mic unavailable'):
            asyncio.run(audio._listen_loop(ctx, tmp_path))
    close.assert_awaited_once();ctx.publish.assert_not_called()


def test_shutdown_cancels_blocked_listener(tmp_path):
    ctx = context();closed = []
    async def listener(*args):
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)
    async def exercise():
        task = asyncio.create_task(audio._run_until_stopped(ctx, tmp_path))
        await asyncio.sleep(.01)
        ctx.stop.set()
        await asyncio.wait_for(task, 1)
    with patch.object(audio, '_listen_loop', listener):
        asyncio.run(exercise())
    assert closed == [True]


def test_fake_producer_never_accesses_microphone_or_models():
    ctx = context();ctx.stop.set()
    ctx = ProducerContext(ctx.publish, ctx.stop, ctx.settings, True)
    with patch.object(audio, '_listen_loop', side_effect=AssertionError('No microphone')), \
         patch.object(audio, '_load_whisper_pipeline', side_effect=AssertionError('No model')):
        audio.run(ctx)
    ctx.publish.assert_not_called()


def test_pcm_conversion_and_invalid_metadata():
    samples, rms = audio._prepare_16k_mono(np.array([0, 16384]*4, np.int16).tobytes(), 8000, 2)
    assert len(samples) == 8 and np.allclose(samples, .5) and rms == 16384
    with pytest.raises(ValueError):audio._prepare_16k_mono(b'', 0, 1)
