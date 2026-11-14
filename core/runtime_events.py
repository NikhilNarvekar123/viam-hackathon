"""Threaded producer hooks and checkpoint-only events. No hardware commands.

This is NOT an immediate stop mechanism. The controller is the only consumer;
producers must use bounded I/O waits and honour their shutdown Event.
"""
import asyncio
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
import math
import threading
import time
from typing import Callable

LOG = logging.getLogger(__name__)


class EventKind(str, Enum):
    RESET_BATCH = 'reset_batch'
    SCENE_CHANGED = 'scene_changed'
    NOTICE = 'notice'
    AUDIO_READY = 'audio_ready'
    VOICE_COMMAND = 'voice_command'


@dataclass(frozen=True)
class RuntimeEvent:
    kind: EventKind
    source: str
    created_at: float
    detail: str = ''


@dataclass(frozen=True)
class Checkpoint:
    events: tuple[RuntimeEvent, ...]
    reset_requested: bool


@dataclass(frozen=True)
class ProducerContext:
    publish: Callable
    stop: threading.Event
    settings: dict
    fake: bool


class EventRuntime:
    def __init__(self, settings, producers=None, fake=False):
        self.capacity = settings['queue_capacity']
        self.poll_s = float(settings['idle_poll_s'])
        self.join_s = float(settings['join_timeout_s'])
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError('runtime_events.queue_capacity must be a positive integer')
        if any(not math.isfinite(v) or v <= 0 for v in (self.poll_s, self.join_s)):
            raise ValueError('Runtime wait intervals must be positive and finite')
        self.reset_kinds = {EventKind(k) for k in settings['reset_events']}
        self.settings = settings
        self.producers = dict(producers or {})
        self.fake = fake
        self.stop = threading.Event()
        self._events = deque()
        self._lock = threading.Lock()
        self._fault = None
        self._threads = []

    def publish(self, kind, source, detail=''):
        """Nonblocking. False means closed/overflowed; overflow faults the next checkpoint."""
        event = RuntimeEvent(EventKind(kind), str(source), time.monotonic(), str(detail))
        with self._lock:
            if self.stop.is_set():
                return False
            if len(self._events) >= self.capacity:
                self._fault = 'Runtime event queue overflow: refusing further picks'
                return False
            self._events.append(event)
        return True

    def checkpoint(self):
        """Called by the controller only, never by a producer. Faults stay latched."""
        with self._lock:
            if self._fault:
                raise RuntimeError(self._fault)
            if self.stop.is_set():
                raise RuntimeError('Event runtime is closed')
            events = tuple(self._events)
            self._events.clear()
        for event in events:
            LOG.info('Checkpoint event: %s from %s', event.kind.value, event.source)
        return Checkpoint(events, any(e.kind in self.reset_kinds for e in events))

    async def wait_for_reset(self):
        """Idle without recapturing/reprocessing returned rejects or blocked objects."""
        while True:
            result = self.checkpoint()
            if result.reset_requested:
                return result
            await asyncio.sleep(self.poll_s)

    def _run_producer(self, name, run):
        context = ProducerContext(
            publish=lambda kind, detail='': self.publish(kind, name, detail),
            stop=self.stop, settings=self.settings.get('producers', {}).get(name, {}),
            fake=self.fake,
        )
        try:
            run(context)
            if not self.stop.is_set():
                raise RuntimeError('Producer exited before shutdown')
        except Exception as exc:
            with self._lock:
                self._fault = f'Producer {name} failed ({type(exc).__name__}); refusing further picks'
            LOG.exception('Background producer %s failed', name)

    def __enter__(self):
        if self._threads or self.stop.is_set():
            raise RuntimeError('EventRuntime instances cannot be restarted')
        try:
            for name, run in self.producers.items():
                thread = threading.Thread(target=self._run_producer, args=(name, run),
                                          name=f'pipeline-{name}', daemon=True)
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        self.stop.set()
        for thread in self._threads:
            thread.join(timeout=self.join_s)
            if thread.is_alive():
                LOG.error('Producer %s ignored shutdown; thread cannot be forcibly killed', thread.name)


@contextmanager
def managed_runtime(settings, supplied=None, fake=False):
    """A supplied runtime is caller-owned; otherwise start only configured hooks."""
    if supplied is not None:
        yield supplied
    elif not settings.get('enabled', False):
        yield None
    else:
        from monitors import audio, overview
        hooks = {'audio': audio.run, 'overview': overview.run}
        configured = settings.get('producers', {})
        if set(configured) - set(hooks):
            raise ValueError('Unknown runtime producer')
        hooks = {name: run for name, run in hooks.items()
                 if configured.get(name, {}).get('enabled', False)}
        with EventRuntime(settings, hooks, fake=fake) as runtime:
            yield runtime
