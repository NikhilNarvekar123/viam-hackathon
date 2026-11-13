import pytest

from vlm.parse import parse_decision


@pytest.mark.parametrize("raw", [
    '{"defect_type":"scratch","action":"reject"}',
    'Result: {"defect_type":"scratch","action":"reject"} Done.',
    '```json\n{"defect_type":"scratch","action":"reject"}\n```',
])
def test_valid(raw):
    decision = parse_decision(raw, ["reject"])
    assert decision.valid
    assert decision.action == "reject"
    assert decision.defect_type == "scratch"
    assert decision.raw_output == raw


@pytest.mark.parametrize("raw", [
    'not JSON',
    '{"action":',
    '{"defect_type":"scratch","action":"unsafe"}',
    '{"defect_type":"scratch","action":["reject"]}',
    '{"action":"reject"}',
    '{"defect_type":null,"action":"reject"}',
    '{"defect_type":"scratch","action":"unsafe",'
    '"nested":{"defect_type":"scratch","action":"reject"}}',
])
def test_fallback(raw):
    decision = parse_decision(raw, ["reject"])
    assert not decision.valid
    assert decision.action == "reject"
    assert decision.raw_output == raw


def test_configured_action():
    assert parse_decision('{"defect_type":"spot","action":"review"}',
                          ["reject", "review"]).valid
