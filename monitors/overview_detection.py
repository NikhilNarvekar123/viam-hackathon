"""CPU-only overview comparison, with explicit visibility and reference inputs.

No camera, robot, model, thread, or runtime configuration access. This does NOT
identify the arm: the caller must supply trustworthy clear-view information.
Pixel coordinates belong to this overview image, never to a wrist-image box.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionSettings:
    difference_threshold: float = 25.0
    min_area_px: int = 12
    persistence_frames: int = 3
    illumination_shift_limit: float = 12.0
    broad_change_fraction: float = 0.35

    def __post_init__(self):
        for name in ('min_area_px', 'persistence_frames'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, low, high in (('difference_threshold', 0, 255),
                                ('illumination_shift_limit', -1, 255),
                                ('broad_change_fraction', 0, 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f'Invalid {name}')
        if self.illumination_shift_limit < 0:
            raise ValueError('illumination_shift_limit must be nonnegative')


@dataclass(frozen=True)
class Detection:
    status: str
    boxes_xyxy: tuple = ()
    changed_pixels: int = 0
    reason: str = ''

    @property
    def publish_scene_changed(self):
        # Only the transition publishes. The latch needs an explicit fresh reference.
        return self.status == 'changed'


def _mask(value, shape=None):
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.ndim != 2 or 0 in result.shape:
        raise ValueError('Masks must be nonempty 2D boolean arrays')
    if shape is not None and result.shape != shape:
        raise ValueError('Mask dimensions differ from workspace')
    return result.copy()


class OverviewChangeDetector:
    """Compare against an explicitly accepted reference; never learn changes away.

    Clear-view=False resets persistence, NOT the reference. Once a change is
    confirmed, one event is emitted and further events are latched until a new
    reference is explicitly accepted after fresh inspection. No object identities
    or confidence about hidden pixels are inferred.
    """

    def __init__(self, workspace_mask, excluded_mask=None, settings=None):
        self.settings = settings or DetectionSettings()
        self.workspace = _mask(workspace_mask)
        if excluded_mask is not None:
            self.workspace &= ~_mask(excluded_mask, self.workspace.shape)
        if not self.workspace.any():
            raise ValueError('Workspace contains no observable pixels')
        self._reference = None
        self._streak = np.zeros(self.workspace.shape, dtype=np.int64)
        self._latched = None

    def _frame(self, frame):
        value = np.asarray(frame)
        if value.dtype != np.uint8 or value.shape != (*self.workspace.shape, 3):
            raise ValueError('Frame must be uint8 RGB with workspace dimensions')
        return value.astype(np.float32)

    @staticmethod
    def _visibility(clear_view):
        if type(clear_view) is not bool:
            raise ValueError('clear_view must be an explicit boolean')

    def set_reference(self, frame, *, clear_view):
        """Explicit new inspection epoch. Caller must verify a clear, stable frame.

        Do not call just because motion ended: that would erase changes made
        during occlusion. A reference should correspond to fresh inspection.
        """
        self._visibility(clear_view)
        if not clear_view:
            raise ValueError('Cannot accept an obscured reference')
        self._reference = self._frame(frame)
        self._streak.fill(0)
        self._latched = None

    def accept_expected_change(self, frame, expected_mask, *, clear_view):
        """Update ONLY a verified expected-change patch, not the entire frame.

        For a future controller adapter: the mask must be in overview pixels and
        the expected pick must be independently verified. Never guess an identity
        from similar fasteners. Unrelated changes inside this patch cannot be
        distinguished by this primitive; uncertain outcomes require reinspection.
        """
        self._visibility(clear_view)
        if not clear_view:
            raise ValueError('Cannot accept expected change through occlusion')
        if self._reference is None or self._latched is not None:
            raise ValueError('Expected change requires an active, unlatched reference')
        current = self._frame(frame)
        patch = _mask(expected_mask, self.workspace.shape)
        if not patch.any() or (patch & ~self.workspace).any():
            raise ValueError('Expected patch must be nonempty and inside workspace')
        self._reference[patch] = current[patch]
        self._streak[patch] = 0

    def observe(self, frame, *, clear_view):
        self._visibility(clear_view)
        current = self._frame(frame)
        if self._reference is None:
            raise ValueError('An explicit clear reference is required')
        if self._latched is not None:
            return Detection('latched', self._latched.boxes_xyxy,
                             self._latched.changed_pixels, self._latched.reason)
        if not clear_view:
            self._streak.fill(0)
            return Detection('obscured', reason='Workspace visibility is unverified')

        delta = current - self._reference
        # Tolerate only a bounded uniform per-channel brightness offset. Large
        # lighting/camera changes must remain visible as uncertainty, not be learned.
        offset = np.median(delta[self.workspace], axis=0)
        if np.max(np.abs(offset)) <= self.settings.illumination_shift_limit:
            delta -= offset
        changed = (np.max(np.abs(delta), axis=2) >= self.settings.difference_threshold)
        changed &= self.workspace
        self._streak[~changed] = 0
        self._streak[changed] = np.minimum(
            self._streak[changed] + 1, self.settings.persistence_frames)
        persistent = self._streak >= self.settings.persistence_frames
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            persistent.astype(np.uint8), connectivity=8)
        boxes, pixels = [], 0
        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            if area >= self.settings.min_area_px:
                boxes.append((x, y, x + width, y + height))
                pixels += area
        if not boxes:
            return Detection('pending' if changed.any() else 'unchanged')
        reason = ('broad_or_uncertain_change'
                  if pixels / int(self.workspace.sum()) >= self.settings.broad_change_fraction
                  else 'persistent_workspace_change')
        result = Detection('changed', tuple(sorted(boxes)), pixels, reason)
        self._latched = result
        return result
