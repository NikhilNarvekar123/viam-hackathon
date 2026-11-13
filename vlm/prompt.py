import json


def build_prompt(template: str, defect_types: list[str], allowed_actions: list[str]) -> str:
    for name, values in (("defect_types", defect_types), ("allowed_actions", allowed_actions)):
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{name} must be a nonempty list of nonempty strings")
    if "reject" not in allowed_actions:
        raise ValueError("allowed_actions must include safe fallback 'reject'")
    if not isinstance(template, str) or any("{" + name + "}" not in template
                                          for name in ("defect_types", "allowed_actions")):
        raise ValueError("Prompt template must contain {defect_types} and {allowed_actions}")
    # Literal substitution leaves JSON example braces intact, unlike str.format.
    prompt = template.replace("{defect_types}", json.dumps(defect_types))
    prompt = prompt.replace("{allowed_actions}", json.dumps(allowed_actions))
    example = json.dumps({"defect_type": defect_types[0], "action": "reject",
                          "reason": "Visible defect."})
    return (prompt.strip() + "\nInspect the physical object in the unannotated original RGB crop. "
            "No heatmap is supplied. Judge only visible physical evidence.\n"
            "Return JSON only, one object with defect_type, action, and reason. "
            "Choose action only from allowed_actions. Keep reason to at most 15 words. "
            "No markdown or surrounding text.\nExample reply (format only, not a prediction): "
            + example)
