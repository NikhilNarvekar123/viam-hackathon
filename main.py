"""Item-agnostic capture → detect → decide → sort orchestration."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter

import yaml

from core.factory import managed_components
from core.metrics import print_summary
from robot.sorting import build_sort_plan, describe_plan, sort_item

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc)


async def run(config: dict, n: int, fake: bool, confirm_motion: bool = False):
    async with managed_components(config, ROOT, fake, confirm_motion) as components:
        await run_loop(config, n, components, confirm_motion)


async def run_loop(config, n, components, confirm_motion=False):
    motion_enabled = not components.real_hardware or confirm_motion
    log_dir = ROOT / config["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{now():%Y%m%dT%H%M%S_%fZ}.jsonl"
    # Every fake hardware call is recorded, without flooding the summary console.
    handler = logging.FileHandler(log_path.with_suffix(".hardware.log"))
    logger = logging.getLogger("robot")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    records = []
    try:
        with log_path.open("a") as stream:
            for _ in range(n):
                latency = {}
                start = perf_counter()
                frame = await components.camera.get_frame()
                latency["capture"] = (perf_counter() - start) * 1000
                timestamp = now().isoformat()
                start = perf_counter()
                detection = components.detector.detect(frame)
                latency["detect"] = (perf_counter() - start) * 1000
                decision = None
                latency["decide"] = 0.0
                if detection.is_anomalous:
                    start = perf_counter()
                    decision = components.decider.decide(frame, detection)
                    latency["decide"] = (perf_counter() - start) * 1000
                    final_bin = config["action_to_bin"][decision.action]
                else:
                    final_bin = config["sorting"]["pass_bin"]
                start = perf_counter()
                plan = build_sort_plan(frame, final_bin, config["sorting"])
                if motion_enabled:
                    await sort_item(components.arm, components.gripper, plan,
                                    allow_stub=not components.real_hardware)
                else:
                    message = f"DRY RUN {frame.item_id}: would {describe_plan(plan)}"
                    logger.info(message)
                    print(message)
                latency["sort"] = (perf_counter() - start) * 1000
                record = {
                    "item_id": frame.item_id, "timestamp": timestamp,
                    "ground_truth": frame.ground_truth,
                    "anomaly_score": detection.score,
                    "pass_fail": "fail" if detection.is_anomalous else "pass",
                    "raw_vlm_output": decision.raw_output if decision else None,
                    "parsed_action": decision.action if decision else None,
                    "defect_type": decision.defect_type if decision else None,
                    "vlm_valid": decision.valid if decision else None,
                    "decider_implementation": getattr(components.decider, "implementation", "unknown"),
                    "final_bin": final_bin, "latency_ms": latency,
                    "motion_executed": motion_enabled,
                    "sort_plan": describe_plan(plan),
                    "pick_target_is_stub": plan.uses_stub_target,
                    "hardware": "viam" if components.real_hardware else "fake",
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                records.append(record)
    finally:
        logger.removeHandler(handler)
        handler.close()
    print(("REAL" if components.real_hardware else "FAKE")
          + " hardware; detector=" + config["models"]["detector"].get("implementation", "stub")
          + "; decider=" + getattr(components.decider, "implementation", "unknown"))
    if not motion_enabled:
        print("DRY RUN: bins and sorting metrics are planned only; no physical sorting.")
    print_summary(records, config["sorting"]["pass_bin"])
    print(f"Item log: {log_path.relative_to(ROOT)}")
    print(f"Hardware call log: {log_path.with_suffix('.hardware.log').relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--confirm-motion", action="store_true",
                        help="Explicitly authorize real arm/gripper motion")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    try:
        asyncio.run(run(config, args.n, args.fake, args.confirm_motion))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; cleanup attempted.\n")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Setup/run error: {exc}\n")


if __name__ == "__main__":
    main()
