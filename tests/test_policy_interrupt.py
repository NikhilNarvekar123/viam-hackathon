"""Regression: never submit the remaining old-policy requests or use partial verdicts."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image
import pytest

from vlm.service import InspectionReset, decide_wrist_objects


def test_reset_after_inflight_request_discards_batch_before_second_request(tmp_path):
    Image.new('RGB', (40, 40)).save(tmp_path/'frame.png')
    for i in (1, 2):Image.new('RGB', (20, 20)).save(tmp_path/f'object_{i:03d}.png')
    wrist = dict(output_dir=str(tmp_path), objects=[
        dict(id=i, score=.9, classification='anomalous', bbox_xyxy=[10, 10, 30, 30]) for i in (1, 2)])
    settings = dict(service_name='mainVLM', prompt='Old policy', stage1_fast_pass=False,
                    routing_policy='vlm_authoritative',
                    action_to_destination=dict(accept='box', reject='reject_bin', hold='hold_area'))
    requested = False
    async def answer(*args):
        nonlocal requested
        requested = True  # Simulates a command arriving while this request is in flight.
        await asyncio.sleep(0)
        return dict(action='accept', valid=True, reason='obsolete verdict')
    service = SimpleNamespace(do_command=AsyncMock(return_value={'state': 'ready'}))
    with patch('vlm.service.connect', new_callable=AsyncMock), \
         patch('vlm.service.disconnect', new_callable=AsyncMock) as close, \
         patch('vlm.service.load_credentials'), \
         patch('viam.services.vision.VisionClient.from_robot', return_value=service), \
         patch('vlm.service.ask_crop', side_effect=answer) as ask:
        with pytest.raises(InspectionReset):
            asyncio.run(decide_wrist_objects(wrist, settings, tmp_path,
                checkpoint=lambda: SimpleNamespace(reset_requested=requested)))
    assert ask.await_count == 1
    close.assert_awaited_once()
    assert json.loads((tmp_path/'inspection_reset.json').read_text())['status'] == 'discarded'
