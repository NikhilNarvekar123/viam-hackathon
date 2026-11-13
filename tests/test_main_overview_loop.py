import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from main import run_overview_loop


def test_cycles_refresh_even_when_empty():
    stage=AsyncMock(side_effect=[{'candidate_count':0},{'candidate_count':2}])
    with patch('robot.overview.overview_stage',stage), patch('main.asyncio.sleep',new_callable=AsyncMock) as sleep:
        asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':2}},fake=True,cycles=2))
    assert stage.await_count==2
    sleep.assert_awaited_once_with(2)


def test_failure_exits_without_next_cycle():
    stage=AsyncMock(side_effect=RuntimeError('capture failed'))
    with patch('robot.overview.overview_stage',stage):
        with pytest.raises(RuntimeError):
            asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':2}},cycles=3))
    assert stage.await_count==1


def test_continuous_until_cancelled():
    stage=AsyncMock(side_effect=[{'candidate_count':0},asyncio.CancelledError()])
    with patch('robot.overview.overview_stage',stage),patch('main.asyncio.sleep',new_callable=AsyncMock):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':2}}))
    assert stage.await_count==2


def test_bad_delay():
    with pytest.raises(ValueError):
        asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':float('nan')}},cycles=1))
