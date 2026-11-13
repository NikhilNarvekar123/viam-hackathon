"""Continuous home -> overview -> wrist anomaly loop; sorting remains separate scaffolding."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
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


async def run_overview_loop(config, fake=False, confirm_motion=False, cycles=None, image_path=None):
    from robot.overview import overview_stage

    delay = float(config['overview'].get('cycle_delay_s', 2.0))
    if not math.isfinite(delay) or delay < 0.1:
        raise ValueError('overview.cycle_delay_s must be finite and >= 0.1 seconds')
    if cycles is not None and cycles < 1:
        raise ValueError('Cycle limit must be positive')
    inspector = None
    if config.get('wrist_inspection_config') and image_path is None:
        from anomaly.wrist import WristInspector
        settings = yaml.safe_load((ROOT / config['wrist_inspection_config']).read_text())
        inspector = WristInspector(settings, ROOT, fake=fake)
    index = 0
    while cycles is None or index < cycles:
        index += 1
        print(f'\n=== Cycle {index}: home -> overview ===', flush=True)
        # Each call homes (or verifies home), then captures a NEW image.
        # Exceptions propagate: never advance after a failed home/capture.
        observation = await overview_stage(config, ROOT, fake, image_path, confirm_motion)
        print(f"Cycle {index}: {observation['candidate_count']} candidates", flush=True)
        if inspector is not None:
            wrist = await inspector.capture_and_inspect()
            cycle_report = {'cycle': index, 'overview_output': observation.get('output_dir'),
                            'overview_count': observation['candidate_count'],
                            'wrist_output': wrist['output_dir'],
                            'wrist_count': wrist['candidate_count'],
                            'wrist_anomalous_count': wrist['anomalous_count'],
                            'cross_camera_object_matching': False}
            (Path(wrist['output_dir']) / 'cycle.json').write_text(json.dumps(cycle_report, indent=2) + '\n')
        # Future stages: match/select -> decide -> sort -> verify.
        # Wrist and overview IDs are NOT associated. Do not infer correspondence.
        # Do not carry frame-local IDs or targets into the next cycle.
        if cycles is None or index < cycles:
            await asyncio.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["overview", "sort"], default="overview",
                        help="Default overview loops home/capture/count; sort is legacy scaffolding")
    parser.add_argument("--overview-image", type=Path, help="Offline overview frame, compared to configured background")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--n", type=int, default=None,
                        help="Cycle limit; default overview runs until Ctrl+C (offline image: once)")
    parser.add_argument("--confirm-motion", action="store_true",
                        help="Explicitly authorize real arm/gripper motion")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    args = parser.parse_args()
    if args.n is not None and args.n < 1:
        parser.error("--n must be at least 1")
    if args.stage == "overview" and not args.fake and not args.overview_image and not args.confirm_motion:
        parser.error("Live loop requires --confirm-motion: every cycle must reach/verify home before capture")
    if args.overview_image and args.confirm_motion:
        parser.error("Offline overview analysis does not move hardware; remove --confirm-motion")
    if args.overview_image and (args.stage != "overview" or args.fake):
        parser.error("--overview-image requires the overview stage without --fake")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    try:
        if args.stage == "overview":
            cycles = args.n if args.n is not None else (1 if args.overview_image else None)
            asyncio.run(run_overview_loop(config, args.fake, args.confirm_motion, cycles, args.overview_image))
        else:
            asyncio.run(run(config, args.n if args.n is not None else 10, args.fake, args.confirm_motion))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; cleanup attempted.\n")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Setup/run error: {exc}\n")


if __name__ == "__main__":
    main()
