"""Home -> overview -> wrist inspection -> planned batch picking, with optional event hooks."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import sys
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


async def read_operator_line():
    """Wait without a worker thread, so Ctrl+C can cancel promptly on Linux/macOS."""
    loop=asyncio.get_running_loop()
    future=loop.create_future()
    fd=sys.stdin.fileno()
    def ready():
        if not future.done():
            try: future.set_result(sys.stdin.readline())
            except Exception as exc: future.set_exception(exc)
    loop.add_reader(fd,ready)
    try:
        return await future
    finally:
        loop.remove_reader(fd)


async def wait_for_next_cycle(sorting_executed=False):
    summary='pick batch complete. Check/clear the GOOD placement area. ' if sorting_executed else 'inspection complete; no sorting actions executed. '
    print('\nPAUSED — '+summary+
          'Reposition parts only with the arm stopped and competing controllers stopped.\n'
          'Clear hands/workspace, then press ENTER to start the next home/capture cycle. '
          'Type Q then ENTER, or Ctrl+C, to stop.',flush=True)
    while True:
        line=await read_operator_line()
        if line=='' or line.strip().lower() in ('q','quit'):
            print('Stopping; no next cycle.',flush=True)
            return False
        if not line.strip(): return True
        print('Press ENTER to continue, or Q then ENTER to stop.',flush=True)


async def run_overview_loop(config, fake=False, confirm_motion=False, cycles=None, image_path=None,
                            pause_between_cycles=None, inspect_only=False, event_runtime=None):
    from core.runtime_events import managed_runtime
    if config.get('voice_control', {}).get('enabled'):
        raise ValueError('Voice control requires --continuous')
    with managed_runtime(config.get('runtime_events',{}),event_runtime,fake=fake) as runtime:
        await _run_overview_loop(config,fake,confirm_motion,cycles,image_path,
                                 pause_between_cycles,inspect_only,runtime)


async def _run_overview_loop(config, fake=False, confirm_motion=False, cycles=None, image_path=None,
                             pause_between_cycles=None, inspect_only=False, event_runtime=None):
    from robot.overview import overview_stage

    delay = float(config['overview'].get('cycle_delay_s', 2.0))
    if not math.isfinite(delay) or delay < 0.1:
        raise ValueError('overview.cycle_delay_s must be finite and >= 0.1 seconds')
    if cycles is not None and cycles < 1:
        raise ValueError('Cycle limit must be positive')
    if pause_between_cycles is None:
        pause_between_cycles=bool(config['overview'].get('pause_between_cycles',False)) and not fake and image_path is None
    if pause_between_cycles and cycles!=1 and not sys.stdin.isatty():
        raise ValueError('Paused loops require an interactive terminal (SSH: use ssh -t). '
                         'Use --no-pause-between-cycles only for explicitly requested automatic cycling.')
    picking=bool(config.get('pick_after_inspection',False)) and image_path is None and not inspect_only
    if picking and not fake and not confirm_motion:
        raise ValueError('Main pipeline picking requires --confirm-motion')
    if picking and not fake and cycles!=1 and not pause_between_cycles:
        raise ValueError('Main pipeline picking requires operator pauses between cycles')
    target_source=config.get('pick_target_source','picker')
    if target_source not in ('pipeline','picker'):raise ValueError('pick_target_source must be pipeline or picker')
    external_boxes=picking and target_source=='pipeline'
    if event_runtime is not None and not external_boxes:
        raise ValueError('Runtime events currently require the pipeline-box main picking mode')
    pick_bridge=None;pick_state={}
    inspector = None
    if config.get('wrist_inspection_config') and image_path is None:
        from anomaly.wrist import WristInspector
        settings = yaml.safe_load((ROOT / config['wrist_inspection_config']).read_text())
        inspector = WristInspector(settings, ROOT, fake=fake)
    if picking:
        if inspector is None or not config.get('vlm_service_config'):
            raise ValueError('Main pipeline picking requires wrist inspection and VLM routing configuration')
        from robot.inspection_pick import InspectionBridge,run_pick_loop,verify_main_pick_start
        pick_settings=yaml.safe_load((ROOT/config['pick_bolt_config']).read_text())
        if not external_boxes:pick_bridge=InspectionBridge(ROOT,config,pick_settings,fake,inspector=inspector)
    index = 0
    while cycles is None or index < cycles:
        # Requests already present are satisfied by the fresh capture below; faults still abort.
        if event_runtime is not None:event_runtime.checkpoint()
        index += 1
        pick_report=None
        if picking:
            if pick_state.get('placed',{}).get('box',0)>=pick_settings['max_good_places']:
                print('GOOD place area capacity reached; stopping.');break
            if not fake: await verify_main_pick_start(config,ROOT)
        print(f'\n=== Cycle {index}: home -> overview ===', flush=True)
        # Each call homes (or verifies home), then captures a NEW image.
        # Exceptions propagate: never advance after a failed home/capture.
        observation = await overview_stage(config, ROOT, fake, image_path, confirm_motion)
        print(f"Cycle {index}: {observation['candidate_count']} candidates", flush=True)
        if inspector is not None:
            if external_boxes:
                from robot.external_box_pick import run_external_box_batch
                options={'checkpoint':event_runtime.checkpoint} if event_runtime is not None else {}
                wrist,decisions,pick_report=await run_external_box_batch(config,ROOT,inspector,fake,confirm_motion,pick_state,**options)
            else:
                wrist = await inspector.capture_and_inspect()
                decisions = None
                if config.get('vlm_service_config'):
                    from vlm.service import decide_wrist_objects
                    vlm_settings = yaml.safe_load((ROOT / config['vlm_service_config']).read_text())
                    decisions = await decide_wrist_objects(wrist, vlm_settings, ROOT, fake=fake)
            cycle_report = {'cycle': index, 'overview_output': observation.get('output_dir'),
                            'overview_count': observation['candidate_count'],
                            'wrist_output': wrist['output_dir'],
                            'wrist_count': wrist['candidate_count'],
                            'wrist_anomalous_count': wrist['anomalous_count'],
                            'cross_camera_object_matching': False,
                            'vlm_decisions': decisions, 'sorting_executed': bool(pick_report and pick_report['sorting_executed'])}
            (Path(wrist['output_dir']) / 'cycle.json').write_text(json.dumps(cycle_report, indent=2) + '\n')
            # Batch IDs cannot be used as grasp targets after moving the camera.
            # Pick ONE freshly located target, recheck its classification, then deliver.
            if picking and not external_boxes and decisions and any(d.get('final_action') in ('accept','reject') for d in decisions):
                print('Inspection complete -> pick and sort ONE target (fresh target verification).',flush=True)
                pick_report=await run_pick_loop(config,ROOT,fake,confirm_motion,cycles=1,
                                                bridge=pick_bridge,state=pick_state)
            if pick_report is not None:
                cycle_report['pick_result']=pick_report
                cycle_report['sorting_executed']=pick_report['sorting_executed']
            (Path(wrist['output_dir']) / 'cycle.json').write_text(json.dumps(cycle_report, indent=2) + '\n')
        if pick_report and pick_report.get('status')=='capacity': break
        reset_requested=bool(pick_report and pick_report.get('status')=='reset_requested')
        if reset_requested:
            print('Reset requested: pending batch discarded; next cycle recaptures and reinspects.',flush=True)
        # Wrist and overview IDs are NOT associated. Do not infer correspondence.
        # Do not carry frame-local IDs or targets into the next cycle.
        if cycles is None or index < cycles:
            if pause_between_cycles:
                if picking:
                    if not await wait_for_next_cycle(sorting_executed=True): break
                elif not await wait_for_next_cycle(): break
            elif event_runtime is not None:
                if not reset_requested:
                    print('IDLE — batch exhausted, not necessarily an empty table; waiting for a reset event.',flush=True)
                    await event_runtime.wait_for_reset()
                # A reset starts fresh, but never bypasses the live operator-pause gate above.
            else:
                await asyncio.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["overview", "pick", "sort"], default="overview",
                        help="overview: full main pipeline; pick: standalone inspected picking; sort: legacy scaffolding")
    parser.add_argument('--continuous',action='store_true',
                        help='Indefinite wrist/home validation loop: no ENTER pauses or cycle limit; no overview counting')
    parser.add_argument('--continuous-config',type=Path,default=ROOT/'config/continuous_pick.yaml')
    parser.add_argument('--voice',action='store_true',
                        help='Continuous mode only: listen for start/stop/policy; initially stopped, gestures disabled')
    parser.add_argument('--inspect-only',action='store_true',help='Disable the post-VLM pick/sort in the main pipeline')
    parser.add_argument('--pick-target-source',choices=['pipeline','picker'],
                        help='pipeline: supplied boxes/verdicts for the batch; picker: own selection/reclassification')
    parser.add_argument("--overview-image", type=Path, help="Offline overview frame, compared to configured background")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--n", type=int, default=None,
                        help="Cycle limit; default overview runs until Ctrl+C (offline image: once)")
    parser.add_argument('--pause-between-cycles',action=argparse.BooleanOptionalAction,default=None,
                        help='Pause for ENTER between inspections; defaults to live config (fake/offline do not pause)')
    parser.add_argument("--confirm-motion", action="store_true",
                        help="Explicitly authorize real arm/gripper motion")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    args = parser.parse_args()
    if args.voice and not args.continuous:
        parser.error('--voice requires --continuous')
    if args.continuous and (args.stage!='overview' or args.overview_image or args.inspect_only
                           or args.pick_target_source=='picker' or args.n is not None
                           or args.pause_between_cycles is not None):
        parser.error('--continuous cannot be combined with stage overrides, offline/inspect-only, picker targets, --n, or pause flags')
    if args.n is not None and args.n < 1:
        parser.error("--n must be at least 1")
    if args.stage in ('overview','pick') and not args.fake and not args.overview_image and not args.confirm_motion:
        parser.error("Live loop requires --confirm-motion: every cycle must reach/verify home before capture")
    if args.overview_image and args.confirm_motion:
        parser.error("Offline overview analysis does not move hardware; remove --confirm-motion")
    if args.overview_image and (args.stage != "overview" or args.fake):
        parser.error("--overview-image requires the overview stage without --fake")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    if args.voice:config.setdefault('voice_control',{})['enabled']=True
    if config.get('voice_control',{}).get('enabled') and not args.continuous:
        parser.error('voice_control.enabled requires --continuous')
    if args.pick_target_source is not None:config['pick_target_source']=args.pick_target_source
    try:
        if args.continuous:
            from robot.continuous_pick import run_continuous
            continuous_settings=yaml.safe_load(args.continuous_config.read_text())
            asyncio.run(run_continuous(config,ROOT,fake=args.fake,confirm_motion=args.confirm_motion,
                                      settings=continuous_settings))
        elif args.stage == "overview":
            cycles = args.n if args.n is not None else (1 if args.overview_image else None)
            asyncio.run(run_overview_loop(config, args.fake, args.confirm_motion, cycles, args.overview_image,
                                          args.pause_between_cycles,args.inspect_only))
        elif args.stage=='pick':
            from robot.inspection_pick import run_pick_loop
            paused=args.pause_between_cycles if args.pause_between_cycles is not None else not args.fake
            if not args.fake and args.n!=1 and not paused:
                raise ValueError('Real picking requires operator pauses between cycles')
            if paused and args.n!=1 and not sys.stdin.isatty():
                raise ValueError('Interactive picking requires a terminal (SSH: use ssh -t)')
            pause=(lambda:wait_for_next_cycle(sorting_executed=True)) if paused else None
            asyncio.run(run_pick_loop(config,ROOT,args.fake,args.confirm_motion,args.n,pause))
        else:
            asyncio.run(run(config, args.n if args.n is not None else 10, args.fake, args.confirm_motion))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; cleanup attempted.\n")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Setup/run error: {exc}\n")


if __name__ == "__main__":
    main()
