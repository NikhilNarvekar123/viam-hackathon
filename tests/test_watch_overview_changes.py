import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from tools import watch_overview_changes as watch
from tools.replay_overview_events import replay


def options(tmp_path, *extra):
    return watch.parser().parse_args(['--fake', '--manual', '--hz', '10', '--seconds', '5',
                                     '--out', str(tmp_path), *extra])


def commands(values):
    return iter(values).__next__


def test_fake_camera_watch_and_saved_replay_agree(tmp_path):
    args = options(tmp_path, '--record')
    async def exercise():
        async with watch.source(args) as camera:
            return await watch.watch(args, camera, commands([None] * 8 + ['q']))
    # Fake source must not even read a config or construct a hardware connection.
    args.config = tmp_path / 'nonexistent.yaml'
    folder = asyncio.run(exercise())
    live = [json.loads(line) for line in (folder/'detections.jsonl').read_text().splitlines()]
    saved = list(replay(folder/'replay.json'))
    assert [r['status'] for r in live] == [r['status'] for r in saved]
    assert sum(r['would_publish'] == 'scene_changed' for r in live) == 1
    assert all(r['fake'] for r in live)
    assert (folder/'latest.png').exists()
    assert len(list(folder.glob('change_*.png'))) == 1


def test_manual_occlusion_preserves_reference_until_clear_and_r_rearms(tmp_path):
    args = options(tmp_path, '--record')
    cmds = commands([None, 'o', None, 'r', None, 'c', None, None, None, 'r', None, 'q'])
    folder = asyncio.run(watch.watch(args, watch.SyntheticSource(), cmds))
    rows = [json.loads(line) for line in (folder/'detections.jsonl').read_text().splitlines()]
    assert rows[3]['status'] == 'obscured'  # r refused, not queued for later.
    assert rows[5]['status'] == 'pending'
    assert rows[7]['would_publish'] == 'scene_changed'
    assert rows[8]['status'] == 'latched'
    assert rows[9]['status'] == 'reference'
    assert rows[10]['status'] == 'unchanged'
    assert [r['status'] for r in rows] == [r['status'] for r in replay(folder/'replay.json')]


def test_defaults_poll_every_two_seconds_without_manual_input():
    args = watch.parser().parse_args([])
    assert args.hz == 0.5
    assert not args.manual


def test_automatic_mode_reports_multiple_changes_and_replay_retains_rearms(tmp_path):
    args = options(tmp_path, '--record')
    args.manual = False
    class Camera(watch.SyntheticSource):
        async def image(self):
            image = await super().image()
            # Add at sample 3, then remove at sample 8.
            if self.index > 8:
                image.paste((160, 160, 160), (140, 90, 161, 111))
            return image
    folder = asyncio.run(watch.watch(args, Camera(), commands([None] * 13 + ['q'])))
    rows = [json.loads(line) for line in (folder/'detections.jsonl').read_text().splitlines()]
    events = [row for row in rows if row['would_publish']]
    assert [row['sample'] for row in events] == [5, 10]
    assert all(row['auto_rearmed'] for row in events)
    assert not any(row['status'] == 'latched' for row in rows)
    saved = list(replay(folder/'replay.json'))
    assert sum(row['would_publish'] == 'scene_changed' for row in saved) == 2
    assert sum(row['operation'] == 'reference' for row in saved) == 3


def test_automatic_run_does_not_read_terminal(tmp_path):
    args = options(tmp_path)
    args.manual = False
    args.seconds = .01
    with patch.object(watch, 'terminal_command', side_effect=AssertionError('unexpected stdin')):
        asyncio.run(watch.run(args))


def test_capture_timeout_is_not_retried_and_recording_is_finalized(tmp_path):
    args = options(tmp_path, '--record', '--timeout', '0.01')
    class HangingCamera:
        calls = 0
        async def image(self):
            self.calls += 1
            await asyncio.sleep(10)
    camera = HangingCamera()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(watch.watch(args, camera))
    assert camera.calls == 1
    manifest = next(tmp_path.glob('watch_*/replay.json'))
    assert json.loads(manifest.read_text())['samples'] == []


def test_changed_dimensions_fail_not_silent_reference_reset(tmp_path):
    args = options(tmp_path)
    class Camera:
        calls = 0
        async def image(self):
            self.calls += 1
            return Image.new('RGB', (50 + self.calls, 40))
    with pytest.raises(ValueError, match='dimensions'):
        asyncio.run(watch.watch(args, Camera()))


def test_live_source_uses_camera_only_config_override_and_closes(tmp_path):
    import yaml
    args = options(tmp_path)
    args.fake = False
    args.camera = 'overview-test'
    args.config = tmp_path/'config.yaml'
    args.config.write_text(yaml.safe_dump({'hardware': {'viam': {'camera': 'wrist'}}}))
    states = []
    @asynccontextmanager
    async def capture(config, root, fake):
        assert config['hardware']['viam']['camera'] == 'overview-test'
        assert not fake
        states.append('opened')
        class Camera:
            async def get_frame(self):
                return SimpleNamespace(image=Image.new('RGB', (30, 20)))
        try:
            yield Camera()
        finally:
            states.append('closed')
    async def exercise():
        async with watch.source(args) as camera:
            assert (await camera.image()).size == (30, 20)
            raise ValueError('test failure')
    with patch('robot.camera.capture_camera', capture), pytest.raises(ValueError, match='test failure'):
        asyncio.run(exercise())
    assert states == ['opened', 'closed']
    assert yaml.safe_load(args.config.read_text())['hardware']['viam']['camera'] == 'wrist'


def test_roi_and_fixed_base_exclusion():
    workspace, excluded = watch.masks((80, 60), [10, 10, 70, 50], [[10, 10, 20, 30]])
    assert workspace.sum() == 60 * 40
    assert excluded.sum() == 10 * 20
    assert not workspace[0, 0]
    assert excluded[15, 15]


def test_bad_rectangle_is_refused():
    with pytest.raises(ValueError, match='fit processing image'):
        watch.masks((80, 60), [0, 0, 90, 60], [])


def test_resize_preserves_aspect_ratio_without_upscaling():
    assert watch.prepare(Image.new('RGB', (1280, 720)), 640).size == (640, 360)
    assert watch.prepare(Image.new('RGB', (320, 240)), 640).size == (320, 240)
    assert np.asarray(watch.prepare(Image.new('L', (320, 240)), 640)).shape == (240, 320, 3)


@pytest.mark.parametrize('flag,value', [('--hz','0'), ('--hz','11'), ('--seconds','nan'),
                                       ('--timeout','-1'), ('--width','0'),
                                       ('--min-area','0'), ('--persistence','0')])
def test_bad_arguments(tmp_path, flag, value):
    with pytest.raises(ValueError):
        watch.validate(options(tmp_path, flag, value))
