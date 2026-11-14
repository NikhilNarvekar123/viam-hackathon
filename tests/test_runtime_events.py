import asyncio
from pathlib import Path
import threading
from unittest.mock import patch

import pytest
import yaml

from core.runtime_events import EventKind, EventRuntime, managed_runtime

ROOT = Path(__file__).resolve().parents[1]


def settings():
    result = yaml.safe_load((ROOT / 'config/config.yaml').read_text())['runtime_events']
    result['idle_poll_s'] = .001
    result['join_timeout_s'] = .5
    return result


def test_disabled_runtime_starts_no_producers():
    with patch('monitors.audio.run') as audio, patch('monitors.overview.run') as overview:
        with managed_runtime(settings(), fake=True) as runtime:
            assert runtime is None
        audio.assert_not_called()
        overview.assert_not_called()


def test_two_background_threads_publish_to_one_controller_and_shut_down():
    barrier = threading.Barrier(3)
    ids = set()
    def producer(context):
        assert context.fake
        ids.add(threading.get_ident())
        context.publish(EventKind.NOTICE, 'ready')
        barrier.wait(timeout=2)
        context.stop.wait()
    runtime = EventRuntime(settings(), {'audio': producer, 'overview': producer}, fake=True)
    with runtime:
        barrier.wait(timeout=2)
        signal = runtime.checkpoint()
        assert not signal.reset_requested
        assert {e.source for e in signal.events} == {'audio', 'overview'}
        assert len(ids) == 2 and threading.get_ident() not in ids
    assert all(not t.is_alive() for t in runtime._threads)
    assert not runtime.publish('reset_batch', 'audio')


def test_reset_events_collapse_into_one_reset_and_are_consumed_once():
    runtime = EventRuntime(settings())
    runtime.publish('reset_batch', 'audio')
    runtime.publish('scene_changed', 'overview')
    signal = runtime.checkpoint()
    assert signal.reset_requested and len(signal.events) == 2
    assert not runtime.checkpoint().reset_requested
    with pytest.raises(ValueError):
        runtime.publish('arbitrary_move_command', 'audio')


def test_queue_overflow_is_latched_failure_not_silent_event_loss():
    config = settings()
    config['queue_capacity'] = 1
    runtime = EventRuntime(config)
    assert runtime.publish('notice', 'audio')
    assert not runtime.publish('reset_batch', 'overview')
    for _ in range(2):
        with pytest.raises(RuntimeError, match='overflow'):
            runtime.checkpoint()


@pytest.mark.parametrize('raises', [False, True])
def test_failed_or_unexpectedly_exited_producer_faults_controller(raises):
    def producer(context):
        if raises:
            raise ValueError('synthetic adapter error')
    with EventRuntime(settings(), {'audio': producer}, fake=True) as runtime:
        runtime._threads[0].join(timeout=2)
        with pytest.raises(RuntimeError, match='Producer audio failed'):
            runtime.checkpoint()


def test_idle_ignores_notice_and_wakes_on_reset():
    async def exercise():
        runtime = EventRuntime(settings())
        task = asyncio.create_task(runtime.wait_for_reset())
        runtime.publish('notice', 'audio')
        await asyncio.sleep(.01)
        assert not task.done()
        runtime.publish('scene_changed', 'overview')
        assert (await asyncio.wait_for(task, 1)).reset_requested
    asyncio.run(exercise())


def test_closed_runtime_wakes_idle_instead_of_hanging():
    async def exercise():
        runtime = EventRuntime(settings())
        task = asyncio.create_task(runtime.wait_for_reset())
        await asyncio.sleep(.01)
        runtime.stop.set()
        with pytest.raises(RuntimeError, match='closed'):
            await asyncio.wait_for(task, 1)
    asyncio.run(exercise())


def test_enabled_config_starts_both_inert_hooks_and_closes_them():
    config = settings()
    config['enabled'] = True
    for value in config['producers'].values():
        value['enabled'] = True
    with managed_runtime(config, fake=True) as runtime:
        assert {t.name for t in runtime._threads} == {'pipeline-audio', 'pipeline-overview'}
        assert all(t.is_alive() for t in runtime._threads)
        assert not runtime.checkpoint().events
    assert all(not t.is_alive() for t in runtime._threads)


def test_producers_shut_down_when_main_worker_raises():
    runtime = EventRuntime(settings(), {'audio': lambda ctx: ctx.stop.wait()}, fake=True)
    with pytest.raises(ValueError):
        with runtime:
            raise ValueError('main failed')
    assert runtime.stop.is_set()
    assert all(not t.is_alive() for t in runtime._threads)


@pytest.mark.parametrize('key,value', [('queue_capacity', 0), ('queue_capacity', True),
                                      ('idle_poll_s', 0), ('join_timeout_s', float('inf'))])
def test_bad_settings_rejected(key, value):
    config = settings()
    config[key] = value
    with pytest.raises(ValueError):
        EventRuntime(config)
