"""Synthetic CPU checks only; no validation of camera masks/arm visibility."""
from dataclasses import replace
import threading

import numpy as np
import pytest

from core.runtime_events import EventKind, EventRuntime
from monitors.overview_detection import DetectionSettings, OverviewChangeDetector


def scene():
    return np.full((60, 80, 3), 160, dtype=np.uint8)


def patch(x=20, y=20):
    mask = np.zeros((60, 80), dtype=bool)
    mask[y:y + 6, x:x + 8] = True
    return mask


def detector(reference=None, **kwargs):
    result = OverviewChangeDetector(np.ones((60, 80), dtype=bool), **kwargs)
    result.set_reference(scene() if reference is None else reference, clear_view=True)
    return result


def repeated(det, frame, count=3, clear_view=True):
    return [det.observe(frame, clear_view=clear_view) for _ in range(count)]


@pytest.mark.parametrize('operation', ['added', 'removed', 'moved'])
def test_persistent_object_changes_publish_once(operation):
    before, after = scene(), scene()
    if operation in ('removed', 'moved'):
        before[patch()] = 30
    if operation == 'added':
        after[patch()] = 30
    elif operation == 'moved':
        after[patch(45, 35)] = 30
    det = detector(before)
    results = repeated(det, after, 6)
    assert [r.status for r in results] == ['pending', 'pending', 'changed',
                                          'latched', 'latched', 'latched']
    assert sum(r.publish_scene_changed for r in results) == 1
    assert results[2].changed_pixels == (96 if operation == 'moved' else 48)
    assert (20, 20, 28, 26) in results[2].boxes_xyxy


def test_workspace_and_base_are_excluded_even_if_their_pixels_change():
    workspace = np.zeros((60, 80), dtype=bool)
    workspace[10:50, 10:70] = True
    base = patch()
    det = OverviewChangeDetector(workspace, base)
    det.set_reference(scene(), clear_view=True)
    frame = scene()
    frame[~workspace | base] = 0
    assert all(r.status == 'unchanged' for r in repeated(det, frame, 5))


def test_unchanged_low_noise_and_bounded_uniform_lighting_shift():
    det = detector()
    rng = np.random.default_rng(8)
    for _ in range(6):
        noise = rng.integers(-3, 4, scene().shape)
        frame = (scene().astype(int) + 10 + noise).astype(np.uint8)
        assert det.observe(frame, clear_view=True).status == 'unchanged'


def test_brightness_compensation_does_not_hide_local_changes():
    det = detector()
    frame = scene() + 10
    frame[patch()] = 30
    assert repeated(det, frame)[-1].publish_scene_changed


def test_broad_uncertain_change_is_reported_not_silently_rebaselined():
    det = detector()
    frame = np.full_like(scene(), 220)
    result = repeated(det, frame)[-1]
    assert result.publish_scene_changed
    assert result.reason == 'broad_or_uncertain_change'


def test_transient_change_resets_persistence():
    det = detector()
    frame = scene()
    frame[patch()] = 0
    assert not any(r.publish_scene_changed for r in repeated(det, frame, 2))
    assert det.observe(scene(), clear_view=True).status == 'unchanged'
    assert not any(r.publish_scene_changed for r in repeated(det, frame, 2))
    assert det.observe(frame, clear_view=True).publish_scene_changed


def test_disconnected_noise_is_not_an_object():
    det = detector()
    frame = scene()
    frame[::5, ::5] = 0
    assert not any(r.publish_scene_changed for r in repeated(det, frame, 5))


def test_motion_is_not_learned_and_change_during_occlusion_is_detected():
    det = detector()
    # Caller labels these views obscured; detector does NOT identify an arm.
    obscured = np.zeros_like(scene())
    assert all(r.status == 'obscured' for r in repeated(det, obscured, 10, False))
    after = scene()
    after[patch()] = 0
    result = repeated(det, after)[-1]
    assert result.publish_scene_changed


def test_arm_leaves_unchanged_scene_without_event():
    det = detector()
    repeated(det, np.zeros_like(scene()), 10, False)
    assert all(r.status == 'unchanged' for r in repeated(det, scene(), 5))


def test_occlusion_breaks_persistence_but_keeps_reference():
    det = detector()
    frame = scene()
    frame[patch()] = 0
    repeated(det, frame, 2)
    det.observe(frame, clear_view=False)
    assert [r.status for r in repeated(det, frame)] == ['pending', 'pending', 'changed']


def test_expected_removal_updates_only_its_patch_not_other_object_changes():
    before = scene()
    before[patch()] = 0
    det = detector(before)
    after = scene()
    after[patch(45, 35)] = 0
    det.accept_expected_change(after, patch(), clear_view=True)
    result = repeated(det, after)[-1]
    assert result.boxes_xyxy == ((45, 35, 53, 41),)


def test_verified_expected_removal_alone_does_not_trigger():
    before = scene()
    before[patch()] = 0
    det = detector(before)
    det.accept_expected_change(scene(), patch(), clear_view=True)
    assert all(r.status == 'unchanged' for r in repeated(det, scene(), 4))


def test_expected_patch_does_not_reset_evidence_elsewhere():
    det = detector()
    frame = scene()
    frame[patch(45, 35)] = 0
    repeated(det, frame, 2)
    det.accept_expected_change(frame, patch(), clear_view=True)
    assert det.observe(frame, clear_view=True).publish_scene_changed


def test_explicit_fresh_reference_rearms_after_latch():
    det = detector()
    changed = scene()
    changed[patch()] = 0
    repeated(det, changed)
    with pytest.raises(ValueError, match='unlatched'):
        det.accept_expected_change(changed, patch(), clear_view=True)
    det.set_reference(changed, clear_view=True)
    assert det.observe(changed, clear_view=True).status == 'unchanged'
    assert repeated(det, scene())[-1].publish_scene_changed


def test_frames_and_masks_are_owned_not_mutable_external_references():
    frame, workspace = scene(), np.ones((60, 80), dtype=bool)
    det = OverviewChangeDetector(workspace)
    det.set_reference(frame, clear_view=True)
    workspace[:] = False
    frame[patch()] = 0
    assert repeated(det, frame)[-1].publish_scene_changed


@pytest.mark.parametrize('bad', [None, 1, 'yes', np.bool_(True)])
def test_visibility_must_be_explicit_boolean(bad):
    with pytest.raises(ValueError, match='boolean'):
        detector().observe(scene(), clear_view=bad)


def test_no_implicit_baseline_or_obscured_reference():
    det = OverviewChangeDetector(np.ones((60, 80), dtype=bool))
    with pytest.raises(ValueError, match='reference'):
        det.observe(scene(), clear_view=True)
    with pytest.raises(ValueError, match='obscured'):
        det.set_reference(scene(), clear_view=False)
    with pytest.raises(ValueError, match='active'):
        det.accept_expected_change(scene(), patch(), clear_view=True)


def test_obscured_or_outside_expected_patch_is_refused():
    det = detector(excluded_mask=patch(45, 35))
    with pytest.raises(ValueError, match='occlusion'):
        det.accept_expected_change(scene(), patch(), clear_view=False)
    with pytest.raises(ValueError, match='inside workspace'):
        det.accept_expected_change(scene(), patch(45, 35), clear_view=True)
    with pytest.raises(ValueError, match='nonempty'):
        det.accept_expected_change(scene(), np.zeros((60, 80), dtype=bool), clear_view=True)


@pytest.mark.parametrize('mask', [np.zeros((2, 2), dtype=bool), np.ones((2, 2)),
                                 np.ones((2, 2, 3), dtype=bool)])
def test_invalid_workspace(mask):
    with pytest.raises(ValueError):
        OverviewChangeDetector(mask)


@pytest.mark.parametrize('frame', [np.zeros((10, 10, 3), dtype=np.uint8),
                                  np.zeros((60, 80, 3), dtype=float),
                                  np.zeros((60, 80), dtype=np.uint8)])
def test_bad_frames_fail_instead_of_looking_unchanged(frame):
    with pytest.raises(ValueError, match='Frame'):
        detector().observe(frame, clear_view=True)


@pytest.mark.parametrize('key,value', [('difference_threshold', 0),
                                      ('difference_threshold', float('nan')),
                                      ('min_area_px', True), ('min_area_px', 0),
                                      ('persistence_frames', 0),
                                      ('illumination_shift_limit', -0.1),
                                      ('broad_change_fraction', float('inf'))])
def test_bad_settings(key, value):
    with pytest.raises(ValueError):
        replace(DetectionSettings(), **{key: value})


def test_cpu_detector_can_publish_one_event_from_fake_producer_thread():
    ready = threading.Event()
    def producer(context):
        assert context.fake
        det = detector()
        frame = scene()
        frame[patch()] = 0
        for result in repeated(det, frame, 8):
            if result.publish_scene_changed:
                context.publish(EventKind.SCENE_CHANGED, result.reason)
        ready.set()
        context.stop.wait()
    settings = dict(queue_capacity=4, idle_poll_s=.01, join_timeout_s=1,
                    reset_events=['scene_changed'], producers={'overview': {}})
    with EventRuntime(settings, {'overview': producer}, fake=True) as runtime:
        assert ready.wait(2)
        signal = runtime.checkpoint()
        assert signal.reset_requested and len(signal.events) == 1
        assert signal.events[0].source == 'overview'
        assert not runtime.checkpoint().events
    assert all(not thread.is_alive() for thread in runtime._threads)
