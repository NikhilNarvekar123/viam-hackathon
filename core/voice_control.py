"""Checkpoint-only voice control. No hardware, model calls, or arbitrary dispatch.

A start must be spoken AFTER a stop/gesture has been acknowledged. Buffered
starts cannot cancel it. A voice stop is a soft pause, never an emergency stop.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from core.runtime_events import EventKind

MAX_POLICY_CHARS = 600
COMMANDS = frozenset(('hello', 'start', 'stop', 'dance', 'new_policy'))


def normalize_transcript(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text.lower())).strip()


def parse_voice_command(raw_text):
    """Whole utterances only: 'new policy <rule> over', standalone 'dance', and robot/robo hello/start/stop."""
    if not isinstance(raw_text, str) or len(raw_text) > 2048:
        return None
    text = raw_text.strip()
    policy = re.search(
        r'(?:please\s+)?new\s+polic(?:y|ies)\b\s*[:.,-]?\s*(.+?)\s*\bover\b[.!?]*\s*$',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if policy:
        rule = policy.group(1).strip(' \t:.,-')
        if rule and any(c.isalpha() for c in rule) and len(rule) <= MAX_POLICY_CHARS and not any(ord(c) < 32 for c in rule):
            return dict(command='new_policy', policy=rule, raw_text=text)
        return None
    norm = normalize_transcript(text)
    if re.fullmatch(r'(?:please )?dance(?: please)?', norm):
        return dict(command='dance', raw_text=text)
    match = re.fullmatch(
        r'(?:(?:robot|robo) (?:please )?(hello|start|stop)|(?:please )?(hello|start|stop) (?:robot|robo))(?: please)?',
        norm,
    )
    if match:
        return dict(command=match.group(1) or match.group(2), raw_text=text)
    return None


def voice_runtime_settings(config):
    settings = deepcopy(config['runtime_events'])
    settings['enabled'] = True
    settings['producers']['audio'].update(enabled=True, commands_enabled=True)
    return settings


class VoiceControl:
    """One controller-owned state machine; consume all events through its checkpoint.

    Policies live only in this process. New policies reset a running inspection at
    its next safe boundary, never an in-flight request or organizer service config.
    """
    def __init__(self, settings, root, *, clock=time.monotonic):
        self.clock = clock
        self.max_age = float(settings.get('command_max_age_s', 30.0))
        if not math.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError('Invalid voice command_max_age_s')
        self.ready = False
        self.ready_after = math.inf
        self.paused = True
        self.restart_blocked = True  # Initial wait, explicit stop, and gestures require a fresh start.
        self.pending_policy = None
        self.pending_gesture = None
        self.gesture_in_progress = False
        self.gesture_cancel_requested = False
        self.active_policy = None
        self.policy_version = 0
        self.resume_after = -math.inf
        self.last_sequence = 0
        folder = Path(root) / 'logs/voice'
        folder.mkdir(parents=True, exist_ok=True)
        self.log_path = folder / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.jsonl')
        self.log('waiting', reason='Waiting for audio readiness. Start picks; hello/dance request taught gestures.')

    def log(self, event, **fields):
        row = dict(event=event, time=datetime.now(timezone.utc).isoformat(),
                   paused=self.paused, policy_version=self.policy_version, **fields)
        # Serialize terminal output too: transcripts must not inject terminal control sequences.
        line = json.dumps(row, ensure_ascii=True, allow_nan=False)
        with self.log_path.open('a') as stream:
            stream.write(line + '\n')
        print('[VOICE] ' + line, flush=True)

    def consume(self, events):
        """Return True when pending inspection work must be abandoned."""
        reset = False
        for event in events:
            if event.source != 'audio':
                continue
            if event.kind == EventKind.AUDIO_READY:
                if not self.ready:
                    self.ready = True
                    self.ready_after = event.created_at
                    self.log('ready', reason='Say start to begin. Voice stop is NOT an emergency stop.')
                continue
            if event.kind != EventKind.VOICE_COMMAND:
                continue
            try:
                if len(event.detail) > 4096:
                    raise ValueError('oversized envelope')
                payload = json.loads(event.detail)
                if not isinstance(payload, dict):
                    raise ValueError('invalid envelope')
                command = payload.get('command')
                sequence = payload.get('sequence')
                heard = payload.get('heard_at')
                if (command not in COMMANDS or type(sequence) is not int or sequence < 1
                        or type(heard) not in (int, float) or not math.isfinite(heard)):
                    raise ValueError('invalid command metadata')
                parsed = parse_voice_command(payload.get('raw_text'))
                if not parsed or parsed['command'] != command or parsed.get('policy') != payload.get('policy'):
                    raise ValueError('transcript and command disagree')
                if not math.isfinite(event.created_at) or heard > event.created_at or event.created_at > self.clock():
                    raise ValueError('invalid command timestamps')
            except (ValueError, TypeError):
                self.log('ignored', reason='Malformed or ambiguous voice command')
                continue
            if sequence <= self.last_sequence:
                self.log('ignored', command=command, reason='Duplicate/out-of-order command')
                continue
            self.last_sequence = sequence
            # Never expire a stop: a long delivery/inference must still end paused.
            if command != 'stop' and self.clock() - heard > self.max_age:
                self.log('ignored', command=command, reason='Stale command; repeat it after acknowledgement')
                continue
            if command in ('hello', 'dance') and (not self.ready or self.gesture_in_progress
                    or heard < self.ready_after or heard <= self.resume_after):
                self.log('ignored', command=command, reason='Wait for audio readiness and any active gesture to finish')
                continue
            if command == 'start':
                if (not self.ready or heard < self.ready_after or heard <= self.resume_after
                        or self.pending_gesture is not None or self.gesture_in_progress):
                    self.log('ignored', command=command, reason='Say start AFTER ready/stop/policy acknowledgement or gesture completion')
                    continue
                self.paused = False
                self.restart_blocked = False
                reset = True
                self.log('start', reason='Discard all pending targets/verdicts; fresh inspection')
            else:
                self.paused = True
                self.resume_after = self.clock()
                reset = True
                if command in ('stop', 'new_policy'):
                    self.pending_gesture = None
                    self.gesture_cancel_requested = self.gesture_in_progress
                if command == 'new_policy':
                    self.pending_policy = None
                    self.active_policy = parsed['policy']
                    self.policy_version += 1
                    self.paused = self.restart_blocked or not self.ready
                    self.log('policy_activated', policy=self.active_policy, stage1_fast_pass=False,
                             automatic_restart=not self.paused,
                             reason='Discard old batch; rescan with this policy unless explicitly stopped')
                elif command in ('hello', 'dance'):
                    self.restart_blocked = True
                    self.pending_gesture = command
                    self.log('gesture_requested', gesture=command,
                             reason='Replay taught poses at empty-gripper safe boundary; stay stopped afterward')
                else:
                    self.restart_blocked = True
                    self.log('stopped', reason='Safe checkpoint reached; remain stopped until start')
        return reset

    def begin_gesture(self):
        if self.pending_gesture is None or self.gesture_in_progress or not self.paused:
            raise ValueError('No pending paused gesture')
        name = self.pending_gesture
        self.pending_gesture = None
        self.gesture_in_progress = True
        self.gesture_cancel_requested = False
        self.log('gesture_started', gesture=name)
        return name

    def finish_gesture(self, name, status):
        self.pending_gesture = None
        self.gesture_in_progress = False
        self.gesture_cancel_requested = False
        self.paused = True
        self.restart_blocked = True
        # Commands spoken during the gesture cannot resume picking afterward.
        self.resume_after = self.clock()
        self.log('gesture_' + status, gesture=name, reason='Remain stopped; say a fresh start to inspect')

    def inspection_settings(self, base):
        """Immutable-by-copy settings snapshot for a single inspection epoch."""
        settings = deepcopy(base)
        settings['inspection_policy'] = dict(version=self.policy_version, text=self.active_policy)
        if self.active_policy is not None:
            template = settings.get('custom_policy_prompt')
            if not isinstance(template, str) or template.count('{policy}') != 1:
                raise ValueError('custom_policy_prompt must contain exactly one {policy}')
            settings['prompt'] = template.replace('{policy}', self.active_policy)
            settings['stage1_fast_pass'] = False
        return settings
