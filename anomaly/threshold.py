"""Calibration in exported-model score units; anomalous means score >= threshold."""
import math
import numpy as np


def scores_array(scores, *, required=False):
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional sequence")
    if required and values.size == 0:
        raise ValueError("At least one good score is required")
    return values


def score_stats(scores):
    values = scores_array(scores)
    if not values.size:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {"count": int(values.size), "min": float(values.min()),
            "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std())}


def classification_counts(good, defect, threshold):
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    good, defect = scores_array(good), scores_array(defect)
    return {"good_count": int(good.size), "defect_count": int(defect.size),
            "false_accepts": int(np.sum(defect < threshold)),
            "false_rejects": int(np.sum(good >= threshold))}


def suggest_threshold(good, defect=(), margin=0.01):
    good, defect = scores_array(good, required=True), scores_array(defect)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and nonnegative")
    if not defect.size:
        threshold = max(float(good.max()) + margin,
                        float(np.nextafter(good.max(), np.inf)))
        if not math.isfinite(threshold):
            raise ValueError("Cannot choose a finite threshold above the good scores")
        method = "maximum_good_plus_margin"
    else:
        # Evaluate every distinct classification boundary; midpoint for readability.
        # Minimize balanced error (avoid favoring the larger class). On ties,
        # prefer fewer false accepts, then a lower/more conservative threshold.
        unique = np.unique(np.concatenate((good, defect)))
        candidates = [float(unique[0])]
        for left, right in zip(unique[:-1], unique[1:]):
            midpoint = float(left / 2 + right / 2)
            candidates.append(midpoint if midpoint > left else float(right))
        above = float(np.nextafter(unique[-1], np.inf))
        if math.isfinite(above):
            candidates.append(above)
        def objective(value):
            counts = classification_counts(good, defect, value)
            error = counts["false_rejects"] / good.size + counts["false_accepts"] / defect.size
            return error, counts["false_accepts"], value
        threshold = min(candidates, key=objective)
        method = "minimum_balanced_error_tiebreak_fewer_false_accepts"
    return {"threshold": threshold, "method": method, "margin": margin,
            **classification_counts(good, defect, threshold)}
