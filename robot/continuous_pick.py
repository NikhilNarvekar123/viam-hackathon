"""Opt-in indefinite wrist/home picking loop; the sole motion owner stays here.

No overview counting dependency, keyboard pauses, or cycle limit. Optional voice
commands are checkpoint-only; empty pending work idles without retrying rejects.
"""
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from anomaly.wrist import WristInspector
from core.runtime_events import Checkpoint, managed_runtime
from core.voice_control import VoiceControl, voice_runtime_settings
from robot.external_box_pick import ExternalBoxBatch
from robot.home_scene import HomeScene
from robot.gestures import run_gesture
from vlm.service import decide_wrist_objects


def validate_settings(settings):
    for name in ('idle_poll_s', 'sample_interval_s', 'capture_timeout_s', 'max_frame_age_s', 'home_joint_tolerance_deg'):
        value = settings[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid continuous {name}')
    if not math.isfinite(settings['settle_s']) or settings['settle_s'] < 0:
        raise ValueError('Invalid continuous settle_s')


class HomeMonitor:
    def __init__(self, batch, inspector, settings, checkpoint, voice=None):
        self.batch, self.settings, self.checkpoint = batch, settings, checkpoint
        self.voice = voice
        self.folder = Path(batch.wrist['output_dir']) / 'home_checks'
        self.folder.mkdir()
        self.index = 0
        with Image.open(Path(batch.wrist['output_dir']) / 'frame.png') as image:
            reference = image.convert('RGB')
        with Image.open(Path(batch.wrist['output_dir']) / 'foreground_mask.png') as mask:
            self.foreground = np.asarray(mask) > 0
        self.objects = batch.wrist['localization']['objects']
        self.scene = HomeScene(reference, self.foreground, self.objects,
                               inspector.settings['auto_crop'], settings['validation'])
        (self.folder / 'settings.json').write_text(json.dumps(settings, indent=2)+'\n')

    async def sample(self, expected_removed_id=None, expired=False, idle=False):
        """Channel resets take priority over image comparison, before AND after capture."""
        events = []
        def drain():
            signal = self.checkpoint()
            events.extend(asdict(e) for e in signal.events)
            return signal.reset_requested
        sample_index = self.index
        folder = self.folder / ('idle_latest' if idle else f'{sample_index:06d}')
        folder.mkdir(exist_ok=idle)
        if idle:
            # Unchanged idle polling must not accumulate images indefinitely.
            for name in ('first.png','second.png'):
                (folder/name).unlink(missing_ok=True)
        self.index += 1
        if drain():
            result = dict(status='reset_requested', reason='channel_request_or_shutdown')
        elif expired:
            result = dict(status='reset_requested', reason='original_batch_expired')
        else:
            first, second = await self.batch.capture_home_frames(self.settings)
            first.save(folder / 'first.png')
            second.save(folder / 'second.png')
            if drain():
                result = dict(status='reset_requested', reason='channel_request_or_shutdown')
            else:
                scene = self.scene.check(first, second, expected_removed_id)
                result = dict(status='unchanged' if scene['status']=='unchanged' else 'reset_requested',
                              reason=scene['reason'], scene=scene)
        result.update(runtime_events=events,sample_index=sample_index,idle=idle)
        (folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        if idle and result['status']=='reset_requested':
            folder.rename(self.folder / f'{sample_index:06d}')
        return result

    async def after_attempt(self, entry):
        # execute() calls this ONLY after execute_result() has returned successfully.
        # home_for_check checks gripper emptiness BEFORE any home/open operation.
        if self.voice is not None:
            signal = self.checkpoint()
            if signal.reset_requested:
                # A completed delivery includes the picker's return-to-scan. Do not
                # add another home/open movement after acknowledging a voice stop.
                return dict(status='reset_requested', reason='voice_or_channel_checkpoint',
                            runtime_events=[asdict(e) for e in signal.events])
        await self.batch.home_for_check()
        removed = entry['crop_id'] if entry.get('removed_from_source', False) else None
        if removed is not None and self.batch.fake:
            obj = next(o for o in self.objects if o['id']==removed)
            pixels = np.asarray(self.batch.fake_scene)
            table = tuple(np.median(pixels[~self.foreground], axis=0).astype(np.uint8))
            x0, y0, x1, y1 = obj['bbox_xyxy']
            ImageDraw.Draw(self.batch.fake_scene).rectangle((x0,y0,x1-1,y1-1), fill=table)
        return await self.sample(removed, expired=entry['status']=='batch_expired')


async def run_continuous(config, root, *, fake=False, confirm_motion=False,
                         settings=None, event_runtime=None, stop=None):
    """Run until cancellation, caller shutdown, capacity, or fault; never auto-resume faults.

    `stop` is an optional caller-owned asyncio.Event for embedding/tests. It is
    observed at safe checkpoints, NOT an emergency stop or mid-delivery command.
    """
    root = Path(root)
    if not fake and not confirm_motion:
        raise ValueError('Continuous live picking requires --confirm-motion')
    if not config.get('pick_after_inspection') or config.get('pick_target_source')!='pipeline':
        raise ValueError('Continuous mode requires supplied pipeline targets and picking enabled')
    settings = settings if settings is not None else yaml.safe_load((root/'config/continuous_pick.yaml').read_text())
    validate_settings(settings)
    # Reject invalid image settings before connecting/moving, even with an empty scene.
    HomeScene(Image.new('RGB',(640,480)), np.zeros((480,640),dtype=bool), [], {}, settings['validation'])
    stop = stop if stop is not None else asyncio.Event()
    wrist_settings = yaml.safe_load((root/config['wrist_inspection_config']).read_text())
    inspector = WristInspector(wrist_settings, root, fake=fake)
    vlm_settings = yaml.safe_load((root/config['vlm_service_config']).read_text())
    voice = VoiceControl(config['voice_control'], root) if config.get('voice_control',{}).get('enabled') else None
    runtime_settings = voice_runtime_settings(config) if voice is not None else config.get('runtime_events',{})
    state = {}
    batches = 0
    with managed_runtime(runtime_settings, event_runtime, fake=fake) as runtime:
        def checkpoint():
            signal = runtime.checkpoint() if runtime is not None else Checkpoint((),False)
            voice_reset = voice.consume(signal.events) if voice is not None else False
            return Checkpoint(signal.events, signal.reset_requested or stop.is_set() or voice_reset
                              or voice is not None and voice.paused)
        while not stop.is_set():
            # Faults abort BEFORE home. Earlier reset requests are fulfilled by this
            # fresh capture; requests arriving during inspection are checked again.
            initial = checkpoint()
            while voice is not None and voice.paused and not stop.is_set():
                if voice.pending_gesture is not None:
                    name = voice.begin_gesture()
                    def gesture_checkpoint():
                        signal = runtime.checkpoint()  # Faults still abort, including while gesturing.
                        voice.consume(signal.events)
                        return stop.is_set() or signal.reset_requested or voice.gesture_cancel_requested
                    try:
                        outcome = await run_gesture(name, config, root, fake=fake,
                            confirm_motion=confirm_motion, checkpoint=gesture_checkpoint)
                    except BaseException:
                        voice.finish_gesture(name, 'failed')
                        raise
                    voice.finish_gesture(name, outcome['status'])
                await asyncio.sleep(runtime.poll_s)
                initial = checkpoint()  # Also checks producer/overflow faults while stopped.
            if stop.is_set():break
            async with ExternalBoxBatch(root, config, fake, confirm_motion, state) as batch:
                if state.get('placed',{}).get('box',0) >= batch.settings['max_good_places']:
                    return dict(status='capacity', batches=batches, state=state)
                await batch.home_for_check()
                if voice is not None and checkpoint().reset_requested:
                    continue
                wrist = await batch.capture_and_inspect(inspector)
                if voice is not None and checkpoint().reset_requested:
                    continue
                epoch_settings = voice.inspection_settings(vlm_settings) if voice is not None else vlm_settings
                decisions = await decide_wrist_objects(wrist, epoch_settings, root, fake=fake)
                monitor = HomeMonitor(batch, inspector, settings, checkpoint, voice=voice)
                # Also check the home view after inference, before using its decisions.
                preflight = await monitor.sample()
                if preflight['status']=='reset_requested':
                    report = dict(status='reset_requested', reason=preflight['reason'], objects=[],
                                  pending_crop_ids=[r['crop_id'] for r in batch.rows],
                                  sorting_executed=False, pre_pick_check=preflight)
                else:
                    report = await batch.execute(decisions, checkpoint=checkpoint,
                                                 after_attempt=monitor.after_attempt)
                    report['pre_pick_check'] = preflight
                batches += 1
                report.update(continuous=True, inspection_epoch=batches,
                              inspection_policy=epoch_settings.get('inspection_policy'),
                              initial_events=[asdict(e) for e in initial.events])
                folder = Path(wrist['output_dir'])
                (folder/'batch_pick.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                cycle=dict(cycle=batches,continuous=True,wrist_output=str(folder),
                           wrist_count=wrist['candidate_count'],vlm_decisions=decisions,
                           sorting_executed=report['sorting_executed'],pick_result=report,
                           overview_used=False)
                (folder/'cycle.json').write_text(json.dumps(cycle,indent=2,allow_nan=False)+'\n')
                if stop.is_set():break
                if state['placed']['box'] >= batch.settings['max_good_places']:
                    return dict(status='capacity', batches=batches, state=state)
                if report['status']=='reset_requested':
                    print('Continuous: discard pending targets; fresh home capture/inspection.',flush=True)
                    await asyncio.sleep(settings['idle_poll_s'])
                    continue
                print('Continuous: no pending targets; watching at home (not proof of empty table).',flush=True)
                while not stop.is_set():
                    # Wake promptly on caller shutdown; ordinary sampling stays bounded.
                    try:
                        await asyncio.wait_for(stop.wait(),settings['idle_poll_s'])
                    except asyncio.TimeoutError:
                        pass
                    if stop.is_set():break
                    result = await monitor.sample(idle=True)
                    if result['status']=='reset_requested':break
    return dict(status='stopped', batches=batches, state=state)
