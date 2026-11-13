import numpy as np
import pytest

from anomaly.threshold import classification_counts, score_stats, suggest_threshold


def test_good_only_margin_and_zero_margin():
    result = suggest_threshold([0.1, 0.2, 0.3], margin=0.05)
    assert result["threshold"] == pytest.approx(0.35)
    assert result["false_rejects"] == 0
    assert result["method"] == "maximum_good_plus_margin"
    assert suggest_threshold([1.0], margin=0)["threshold"] > 1.0


def test_separable_classes():
    result = suggest_threshold([0.1, 0.2], [0.8, 0.9])
    assert result["threshold"] == pytest.approx(0.5)
    assert result["false_accepts"] == result["false_rejects"] == 0


def test_overlapping_scores_and_tie_safety():
    result = suggest_threshold([0.2, 0.7], [0.6, 0.9])
    assert result["threshold"] == pytest.approx(0.4)
    assert result["false_accepts"] == 0 and result["false_rejects"] == 1
    assert suggest_threshold([0.5], [0.5])["threshold"] == 0.5


def test_balanced_error_not_dominated_by_good_samples():
    result = suggest_threshold([0.1] * 100 + [0.9], [0.8])
    assert result["false_accepts"] == 0
    assert result["false_rejects"] == 1


def test_threshold_equality_is_anomalous():
    counts = classification_counts([0.5], [0.5], 0.5)
    assert counts["false_accepts"] == 0 and counts["false_rejects"] == 1


def test_stats():
    assert score_stats([])["count"] == 0
    result = score_stats([1, 2, 3])
    assert result["mean"] == 2 and result["min"] == 1 and result["max"] == 3


@pytest.mark.parametrize("good,defect,margin", [([], [], 0), ([np.nan], [], 0),
    ([1], [np.inf], 0), ([1], [], -1), ([1], [], np.nan)])
def test_invalid_scores(good, defect, margin):
    with pytest.raises(ValueError):
        suggest_threshold(good, defect, margin)
