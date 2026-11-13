"""Parse one JSON decision, optionally surrounded by prose or markdown."""
import json
from collections.abc import Collection

from core.types import Decision


def parse_decision(raw_output: str, allowed_actions: Collection[str]) -> Decision:
    fallback = Decision(None, "reject", raw_output, False)
    # Decode from the first object boundary. Do not rescue nested/later objects
    # from an invalid decision (which could accidentally accept an unsafe action).
    start = raw_output.find("{")
    if start < 0:
        return fallback
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw_output[start:])
    except (json.JSONDecodeError, ValueError):
        return fallback
    action = obj.get("action")
    defect_type = obj.get("defect_type")
    if (not isinstance(action, str) or action not in allowed_actions
            or not isinstance(defect_type, str) or not defect_type.strip()):
        return fallback
    return Decision(defect_type, action, raw_output, True)
