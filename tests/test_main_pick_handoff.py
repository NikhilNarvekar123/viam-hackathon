import asyncio
from pathlib import Path
from unittest.mock import AsyncMock,patch
import pytest
import yaml
import main

@pytest.mark.parametrize('action,expected_picks',[('accept',2),('reject',2),('hold',0)])
def test_main_runs_pick_after_vlm_once_per_cycle(tmp_path,action,expected_picks):
    config=yaml.safe_load((main.ROOT/'config/config.yaml').read_text())
    config['pick_target_source']='picker'  # retained optional independent-selection mode
    events=[]
    async def overview(*args):events.append('overview');return {'candidate_count':1}
    async def wrist():
        events.append('wrist');return dict(output_dir=str(tmp_path),candidate_count=1,anomalous_count=1)
    async def vlm(*args,**kwargs):events.append('vlm');return [{'final_action':action}]
    async def pick(*args,**kwargs):
        events.append('pick');return {'status':'held','sorting_executed':False}
    with patch('robot.overview.overview_stage',side_effect=overview), \
         patch('anomaly.wrist.WristInspector') as inspector, \
         patch('robot.inspection_pick.InspectionBridge') as bridge, \
         patch('vlm.service.decide_wrist_objects',side_effect=vlm), \
         patch('robot.inspection_pick.run_pick_loop',side_effect=pick) as pickup, \
         patch('main.asyncio.sleep',new_callable=AsyncMock):
        inspector.return_value.capture_and_inspect=AsyncMock(side_effect=wrist)
        asyncio.run(main.run_overview_loop(config,fake=True,cycles=2))
    assert pickup.await_count==expected_picks
    assert events==(['overview','wrist','vlm']+(['pick'] if expected_picks else []))*2
    inspector.assert_called_once()
    assert bridge.call_args.kwargs['inspector'] is inspector.return_value
    if expected_picks:
        assert all(c.kwargs['cycles']==1 for c in pickup.call_args_list)
        assert pickup.call_args_list[0].kwargs['state'] is pickup.call_args_list[1].kwargs['state']


def test_failed_preflight_prevents_home_or_capture():
    config=yaml.safe_load((main.ROOT/'config/config.yaml').read_text())
    with patch('anomaly.wrist.WristInspector'),patch('robot.inspection_pick.InspectionBridge'), \
         patch('robot.inspection_pick.verify_main_pick_start',new_callable=AsyncMock,side_effect=ValueError('holding item')), \
         patch('robot.overview.overview_stage',new_callable=AsyncMock) as stage:
        with pytest.raises(ValueError,match='holding item'):
            asyncio.run(main.run_overview_loop(config,confirm_motion=True,cycles=1))
    stage.assert_not_awaited()


def test_real_main_refuses_unpaused_multi_pick_before_hardware():
    config={'overview':{'cycle_delay_s':2},'pick_after_inspection':True}
    with patch('robot.overview.overview_stage',new_callable=AsyncMock) as stage:
        with pytest.raises(ValueError,match='operator pauses'):
            asyncio.run(main.run_overview_loop(config,confirm_motion=True,cycles=2,pause_between_cycles=False))
    stage.assert_not_awaited()
