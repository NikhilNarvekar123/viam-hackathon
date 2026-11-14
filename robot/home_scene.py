"""Conservative same-home-view CPU comparison; NOT identity tracking or safety.

Only a verified expected removal can modify a small reference patch. All other
changes request new inspection. No model calls, motion, or target substitution.
"""
import math

import cv2
import numpy as np
from PIL import Image

from anomaly.object_crops import automatic_crops


class HomeScene:
    def __init__(self, image, foreground, objects, crop_settings, settings):
        self.settings = dict(settings)
        self.crop_settings = crop_settings
        self.native_size = image.size
        size = settings['reference_size_px']
        if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ValueError('Invalid scene reference_size_px')
        self.size = tuple(size)
        for name in ('min_change_area_px', 'max_changed_pixels', 'stability_max_changed_pixels', 'table_ring_px'):
            if type(settings[name]) is not int or settings[name] < 1:
                raise ValueError(f'Invalid scene {name}')
        for name in ('expected_margin_px', 'max_remaining_foreground_px'):
            if type(settings[name]) is not int or settings[name] < 0:
                raise ValueError(f'Invalid scene {name}')
        for name in ('difference_threshold', 'stability_threshold', 'max_table_deviation'):
            if not math.isfinite(settings[name]) or not 0 < settings[name] <= 255:
                raise ValueError(f'Invalid scene {name}')
        if not math.isfinite(settings['max_illumination_shift']) or not 0 <= settings['max_illumination_shift'] <= 255:
            raise ValueError('Invalid illumination limit')
        if not math.isfinite(settings['min_removal_fraction']) or not 0 < settings['min_removal_fraction'] <= 1:
            raise ValueError('Invalid removal fraction')
        if np.asarray(foreground).shape != (image.height, image.width):
            raise ValueError('Scene foreground dimensions differ')
        self.reference = self._pixels(image)
        self.foreground = np.asarray(Image.fromarray(np.asarray(foreground, dtype=bool)).resize(
            self.size, Image.Resampling.NEAREST), dtype=bool).copy()
        sx, sy = self.size[0] / image.width, self.size[1] / image.height
        self.boxes = {}
        for obj in objects:
            box = [int(round(v * (sx if i % 2 == 0 else sy))) for i, v in enumerate(obj['bbox_xyxy'])]
            self._region(box)
            if obj['id'] in self.boxes:
                raise ValueError('Duplicate original scene ID')
            self.boxes[obj['id']] = box

    def _pixels(self, image):
        if image.size != self.native_size:
            raise ValueError('Home-view image dimensions changed')
        return np.asarray(image.convert('RGB').resize(self.size, Image.Resampling.BILINEAR), dtype=np.float32)

    def _region(self, box, margin=0):
        x0, y0, x1, y1 = box
        width, height = self.size
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError('Original scene box outside image')
        mask = np.zeros((height, width), dtype=bool)
        mask[max(0, y0-margin):min(height, y1+margin), max(0, x0-margin):min(width, x1+margin)] = True
        return mask

    def _significant(self, mask):
        if int(mask.sum()) >= self.settings['max_changed_pixels']:
            return True
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return any(int(stats[i, cv2.CC_STAT_AREA]) >= self.settings['min_change_area_px'] for i in range(1, count))

    def check(self, first, second, expected_removed_id=None):
        """Return unchanged/changed/uncertain; never silently refresh whole scene.

        The caller must verify home pose, stationary captures and empty gripper.
        Two nearby stable images are a heuristic, not human/occlusion detection.
        """
        try:
            a, current = self._pixels(first), self._pixels(second)
        except ValueError as exc:
            return dict(status='uncertain', reason=str(exc))
        unstable = np.max(np.abs(a-current), axis=2) >= self.settings['stability_threshold']
        if int(unstable.sum()) >= self.settings['stability_max_changed_pixels']:
            return dict(status='uncertain', reason='home_view_not_stable')
        delta = current-self.reference
        shift = np.median(delta, axis=(0, 1))
        if np.max(np.abs(shift)) > self.settings['max_illumination_shift']:
            return dict(status='uncertain', reason='large_illumination_or_scene_change')
        normalized = current-shift
        changed = np.max(np.abs(normalized-self.reference), axis=2) >= self.settings['difference_threshold']
        allowed = np.zeros(changed.shape, dtype=bool)
        if expected_removed_id is not None:
            if expected_removed_id not in self.boxes:
                return dict(status='uncertain', reason='unknown_expected_removal')
            allowed = self._region(self.boxes[expected_removed_id], self.settings['expected_margin_px'])
            # Never hide a neighbor (including earlier REJECT/HOLD objects).
            for ident, box in self.boxes.items():
                if ident != expected_removed_id and (allowed & self._region(box)).any():
                    return dict(status='uncertain', reason='expected_patch_overlaps_other_original_object')
            target_region = self._region(self.boxes[expected_removed_id])
            if (allowed & ~target_region & self.foreground).any():
                return dict(status='uncertain', reason='expected_patch_contains_other_foreground')
            original = target_region & self.foreground
            if not original.any():
                return dict(status='uncertain', reason='missing_original_foreground')
            if float((changed & original).sum()) / int(original.sum()) < self.settings['min_removal_fraction']:
                return dict(status='uncertain', reason='expected_removal_not_visible')
            try:
                _, _, foreground = automatic_crops(second, self.crop_settings)
            except ValueError:
                return dict(status='uncertain', reason='post_pick_segmentation_uncertain')
            foreground = np.asarray(Image.fromarray(foreground).resize(self.size, Image.Resampling.NEAREST), dtype=bool)
            if int((foreground & allowed).sum()) > self.settings['max_remaining_foreground_px']:
                return dict(status='uncertain', reason='foreground_remains_in_expected_removal_patch')
            # Require the revealed patch to resemble the nearby original tabletop,
            # not just to lose the old silhouette. No new object identity is guessed.
            ring = self._region(self.boxes[expected_removed_id], self.settings['expected_margin_px'] + self.settings['table_ring_px'])
            ring &= ~allowed & ~self.foreground
            if not ring.any():
                return dict(status='uncertain', reason='no_local_table_reference')
            table = np.median(self.reference[ring], axis=0)
            residue = np.max(np.abs(normalized[allowed]-table), axis=1) >= self.settings['max_table_deviation']
            if int(residue.sum()) > self.settings['max_remaining_foreground_px']:
                return dict(status='uncertain', reason='revealed_patch_not_table_like')
        outside = changed & ~allowed
        if self._significant(outside):
            return dict(status='changed', reason='unexpected_home_view_change', changed_pixels=int(outside.sum()))
        if expected_removed_id is not None:
            self.reference[allowed] = normalized[allowed]
            self.foreground[allowed] = False
            del self.boxes[expected_removed_id]
        return dict(status='unchanged', reason='remaining_scene_consistent', expected_removed_id=expected_removed_id)
