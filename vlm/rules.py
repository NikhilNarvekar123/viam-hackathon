"""No-model fallback: conservatively reject every flagged item."""
import json
from time import perf_counter

from core.types import Decision, Detection, Frame
from vlm.parse import parse_decision


class RulesDecider:
    implementation = "rules"

    def __init__(self, allowed_actions, defect_type):
        if "reject" not in allowed_actions:
            raise ValueError("Rules fallback requires allowed action 'reject'")
        if not isinstance(defect_type, str) or not defect_type.strip():
            raise ValueError("Rules fallback needs a configured defect type")
        self.allowed_actions = allowed_actions
        self.defect_type = defect_type
        self.last_latency_ms = 0.0

    def decide(self, frame: Frame, detection: Detection) -> Decision:
        start = perf_counter()
        raw = json.dumps({"defect_type": self.defect_type, "action": "reject",
                          "reason": "Rules fallback: reject flagged item; no VLM assessment."})
        decision = parse_decision(raw, self.allowed_actions)
        self.last_latency_ms = (perf_counter() - start) * 1000
        return decision
