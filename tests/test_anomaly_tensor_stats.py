import numpy as np
import pytest
from tools.inspect_anomaly_tensors import tensor_stats


def test_scalar_score_statistics():
    s=tensor_stats(np.array([.5],dtype=np.float32))
    assert s['shape']==[1] and s['dtype']=='float32'
    assert s['min']==s['mean']==s['median']==s['max']==.5
    assert s['max_over_median']==1


def test_map_statistics():
    s=tensor_stats(np.array([0,.25,.75,1],dtype=np.float32).reshape(1,1,2,2))
    assert s['min']==0 and s['mean']==s['median']==.5 and s['max']==1
    assert s['max_over_median']==2 and s['fraction_at_one']==.25


def test_zero_median_ratio_not_fabricated():
    assert tensor_stats(np.zeros((2,2)))['max_over_median'] is None


@pytest.mark.parametrize('values',[[],[float('nan')],[float('inf')]])
def test_invalid_tensors_rejected(values):
    with pytest.raises(ValueError):tensor_stats(values)
