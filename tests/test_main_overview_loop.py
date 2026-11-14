import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from main import run_overview_loop,wait_for_next_cycle,read_operator_line


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


def test_pause_requires_enter_before_next_capture():
    events=[]
    async def capture(*args):
        events.append('capture');return {'candidate_count':0}
    async def advance():
        events.append('enter');return True
    with patch('robot.overview.overview_stage',side_effect=capture), \
         patch('main.wait_for_next_cycle',side_effect=advance) as pause, \
         patch('main.sys.stdin.isatty',return_value=True), \
         patch('main.asyncio.sleep',new_callable=AsyncMock) as sleep:
        asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':2}},fake=True,cycles=2,pause_between_cycles=True))
    assert events==['capture','enter','capture']
    pause.assert_awaited_once();sleep.assert_not_awaited()


def test_quit_does_not_capture_again():
    stage=AsyncMock(return_value={'candidate_count':0})
    with patch('robot.overview.overview_stage',stage), \
         patch('main.wait_for_next_cycle',new_callable=AsyncMock,return_value=False), \
         patch('main.sys.stdin.isatty',return_value=True):
        asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':2,'pause_between_cycles':True}},cycles=3))
    stage.assert_awaited_once()


def test_noninteractive_pause_fails_before_any_hardware():
    with patch('main.sys.stdin.isatty',return_value=False), \
         patch('robot.overview.overview_stage',new_callable=AsyncMock) as stage:
        with pytest.raises(ValueError,match='interactive terminal'):
            asyncio.run(run_overview_loop({'overview':{'pause_between_cycles':True}},cycles=2))
    stage.assert_not_awaited()


@pytest.mark.parametrize('line,expected',[('\n',True),('q\n',False),('',False)])
def test_pause_enter_quit_and_eof(line,expected):
    with patch('main.read_operator_line',new_callable=AsyncMock,return_value=line):
        assert asyncio.run(wait_for_next_cycle()) is expected


def test_cancel_during_pause_propagates_without_next_capture():
    stage=AsyncMock(return_value={'candidate_count':0})
    with patch('robot.overview.overview_stage',stage), \
         patch('main.wait_for_next_cycle',new_callable=AsyncMock,side_effect=asyncio.CancelledError()), \
         patch('main.sys.stdin.isatty',return_value=True):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run_overview_loop({'overview':{'pause_between_cycles':True}},cycles=2))
    stage.assert_awaited_once()


def test_async_terminal_reader_with_pipe():
    import os
    read_fd,write_fd=os.pipe()
    try:
        with os.fdopen(read_fd,'r') as stream,patch('main.sys.stdin',stream):
            async def run():
                asyncio.get_running_loop().call_soon(os.write,write_fd,b'\n')
                assert await asyncio.wait_for(read_operator_line(),1)=='\n'
            asyncio.run(run())
    finally:
        os.close(write_fd)


def test_bad_delay():
    with pytest.raises(ValueError):
        asyncio.run(run_overview_loop({'overview':{'cycle_delay_s':float('nan')}},cycles=1))
