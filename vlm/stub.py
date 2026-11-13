import json
import random

from core.types import Decision, Detection, Frame
from vlm.parse import parse_decision


class StubDecider:
    implementation = "stub"

    def __init__(self, allowed_actions, defect_types, settings):
        self.allowed_actions = allowed_actions
        self.defect_types = defect_types
        self.settings = settings
        self.rng = random.Random(settings["seed"])
        if not defect_types or settings["valid_action"] not in allowed_actions:
            raise ValueError("Stub requires defect types and a valid configured action")
        rates = [settings["malformed_rate"], settings["disallowed_rate"]]
        if any(not 0 <= rate <= 1 for rate in rates) or sum(rates) > 1:
            raise ValueError("Stub output rates must be nonnegative and sum to <= 1")
        if settings["disallowed_action"] in allowed_actions:
            raise ValueError("Stub disallowed_action must not be allowed")

    def decide(self, frame: Frame, detection: Detection) -> Decision:
        draw = self.rng.random()
        if draw < self.settings["malformed_rate"]:
            raw = '{"action": '
        else:
            disallowed = draw < (self.settings["malformed_rate"]
                                 + self.settings["disallowed_rate"])
            raw = json.dumps({
                "defect_type": self.rng.choice(self.defect_types),
                "action": self.settings["disallowed_action"] if disallowed
                else self.settings["valid_action"],
            })
        return parse_decision(raw, self.allowed_actions)
