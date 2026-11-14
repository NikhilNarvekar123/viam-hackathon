# AGENTS.md

## Repository identity — verify before editing

- The canonical GitHub repository for this pipeline is **https://github.com/NikhilNarvekar123/viam-hackathon** (HTTPS `.git` and equivalent SSH URLs are acceptable).
- Before editing, committing, deploying, or pushing, check `git rev-parse --show-toplevel`, `git remote -v`, and `git status --short`. Work in a verified `viam-hackathon` checkout; a folder named `viam` is not proof of repository identity. Do not publish to `NikhilNarvekar123/viam`.
- Older development/deployment directories can contain newer copied files despite an old Git HEAD or a legacy remote. Resolve mismatches with the user; preserve those files and compare hashes before migration. Never silently repoint remotes or reset/clean/pull over a dirty deployment.
- **User workflow preference: publish directly to `viam-hackathon/main`; do not create feature/review branches unless the user explicitly asks.** Fetch and review current `origin/main` before publishing; preserve teammates' commits and never force-push. If `main` is checked out in another active worktree, coordinate or use an isolated checkout of `origin/main` and push `HEAD:main` without creating a new branch.
- Coordinate shared-file changes with the audio/camera workstreams, preserve their work, and commit only explicitly reviewed paths. Private diagnostic/handoff documents, images, models, credentials, and raw machine logs must not be swept into commits.

## Project

Entry for Viam's "Robot Hackathon: Edge of Intelligence", Track 3: **Spot it, sort it**.

A robot station that inspects items, detects defective ones, decides how to handle each defect, and physically sorts them: good items go into the box, defective items go into the reject bin.

The scenario is scattered fasteners (screws, bolts, nuts, etc.). Tape of any color or marker marks are defects. Keep implementations item-agnostic: item names, defect types, allowed actions, thresholds, camera crop, and arm poses live in config, never in code.

Judging criteria: detection accuracy, correct responses (VLM decisions), latency, how work is split across NPU / GPU / CPU, and a short design explanation.

## Pipeline

1. **Capture**: get a color frame (and optionally depth) from the Viam camera, then crop to the item region of interest (ROI).
2. **Detect**: an Anomalib anomaly model, trained on good images only and exported to OpenVINO IR. Outputs an anomaly score, a heatmap, and pass/fail against a threshold.
3. **Decide**: VLM is re-enabled (`enabled: true` in `config/vlm_service.yaml`) following service recovery and explicit user authorization. Avoid concurrent browser vision polling that previously caused a request backlog. Current policy: Stage 1 scores below the user-selected 0.60 threshold go directly to GOOD without a VLM call. Remaining candidates reach the local Viam-hosted VLM with a proportionally framed square crop, resized to 224x224: side=max(int(36*w/640), int(max(bbox_width,bbox_height)*1.45)). Anomalib uses the same isolated crop resized to 128x128. Other detected contours are filled with median tabletop color after 4-reference-pixel dilation; no heatmaps, annotations, or anomaly scores enter the VLM request. Boundary-crossing square windows stop inspection instead of adding artificial padding. The configured prompt checks surface finish, foreign material and physical/structural damage. Stage 2 returns observation first, then a boolean defect; true maps to reject, false to accept, and malformed/uncertain output to HOLD. Invalid, failed, or uncertain responses route to `hold`. `config/vlm_service.yaml` owns fast-pass and routing settings; `vlm_authoritative` allows valid VLM accept to go to the good box. Accuracy is NOT validated: saved-frame replay still falsely accepted two known defects. Threshold 0.60 is an operating choice, not a validated calibration.
4. **Sort**: the Viam Python SDK moves the arm through taught poses: home → pick → box or reject bin → home.

The current default `main.py` homes, opens the gripper, counts overview candidates, and evaluates wrist crops. Live multi-cycle runs pause after each completed inspection until ENTER; Q+ENTER, EOF or Ctrl+C stops. The next cycle may move home/open the gripper, so clear hands first. Interactive terminal required for paused runs; fake/offline runs stay noninteractive by default. Use --pause-between-cycles / --no-pause-between-cycles to explicitly override. With `pick_after_inspection: true` and default `pick_target_source: pipeline`, wrist RGB/depth are captured at the same stationary pose and the original crop boxes directly select raw-depth ROIs, without requiring independent depth proposals or 12-pixel center matches. After batch VLM routing, each localized ACCEPT/REJECT object is attempted once, tallest measured height first, before pausing. External ROIs are reprojected and measured at close-up; original size/height/grasp and clearance limits still apply. GOOD and REJECT areas sit in the taped zones on the +y and -y sides (`good_xy`/`reject_xy` in `config/pick_bolt.yaml`, place grid in `config/pick_bolt_controller.yaml`); GOOD must keep the claws clear of the `wall-side` obstacle (y >= 450) in the machine config. Before each set-down the wrist camera looks at the area and the part takes the first free place (tape/cables ignored, room left for the opening jaws), so later runs do not stack on earlier ones; a full area puts the part back where it was picked. Objects already inside either area are not picked. Each batch target is re-scanned from straight above at scan height. A target whose top rises >= 40 mm is a standing bolt: aimed at its top from that scan (no close-up: at 340 mm its top is inside the depth camera's minimum range), approached from 25 mm above the top, gripped 27 mm above the table. Arm speed/acceleration 120 deg/s(^2) in `config/config.yaml`; gripper minimum part height 2 mm. Original verdicts are reused with no second inference pass or added whole-scene comparison. Unmatched/ambiguous targets and HOLD are left ungrasped. `--pick-target-source picker` retains the older independent picker selection/reclassification mode. Use `--inspect-only` to suppress that step. `--confirm-motion` remains mandatory for real runs. The optional standalone `--stage pick --confirm-motion` integrates the teammate's depth-based picker with our classifier on its freshly selected close-up target. ACCEPT uses the GOOD area, REJECT the reject area, and HOLD prevents grasping. It retains home verification and operator pauses. Batch execution has only been mock-tested; same-view associations were inspected in a read-only live audit, but physical placement is not validated; see `INSPECTED_PICKING.md`. Standalone picker motion is disabled to avoid its legacy accept-everything stub. Active imported controller settings are in `config/pick_bolt_controller.yaml`; bridge settings are in `config/pick_bolt.yaml`. Legacy sorter scaffolding has placeholder poses and is not validated. Each stage lives in its own module behind a small interface so it can be swapped or faked.

## Opt-in continuous picking

`main.py --continuous` adds an indefinite supplied-ROI loop without ENTER pauses or cycle limits; real runs still require fresh `--confirm-motion` authorization. It uses the wrist camera at verified home, not the separate camera's old overview-counting stage. After completed deliveries/returns (and HOLD/refusals), verify empty gripper before home/open, check existing channel events, and compare fresh stable home-view images. Only verified expected removal after successful GOOD/REJECT placement is exempted; full-area put-backs get no exemption. Unexpected/uncertain changes discard pending work for fresh inspection. Unchanged remaining targets retain original verdicts/geometry. Exhausted work idles instead of retrying rejects/HOLDs. Capacity, faults, motion/clearance checks, and failure propagation remain enforced. Optional `--voice` enables the Whisper audio producer and checkpoint-only start/stop/policy commands. It starts stopped; start resets pending inspection work, stop finishes delivery then stays stopped, and a new policy is logged pending until a later start confirms it. Confirmed custom policies replace Qwen's defect criteria and disable Stage 1 fast-pass. Hello replays the saved left_wave/right_wave pair twice; dance replays left_wave/up_wave/right_wave/down_wave once. Both use the saved home before/after a successful routine and leave picking stopped; see `GESTURES.md`. Real gestures require explicit motion authorization, idle teleop/arm, and an empty gripper. Stops cancel remaining waypoints at a completed-waypoint boundary; faults never auto-home/release. Buffered starts cannot cancel a just-acknowledged pause or resume picking after a gesture. No voice E-stop, arbitrary motion, fault/capacity reset, or side-camera adapter is added. See `VOICE_CONTROL.md`. This mode is fake/mock-tested, not physically validated. See `CONTINUOUS_PICKING.md`; coordinate shared controller deployment with producer owners.

## Optional background-event scaffold

`runtime_events` in `config/config.yaml` is disabled by default. `core/runtime_events.py` provides bounded thread-safe event delivery and producer lifecycle; `monitors/audio.py` is the Whisper listener; it is diagnostic-only unless continuous voice mode enables typed commands. `monitors/overview.py` remains an inert integration hook. The supplied-ROI batch loop checks events after planning remaining targets and after every completed attempt/delivery, never mid-grasp. Reset events discard pending boxes/verdicts for a fresh next cycle; GOOD placement counters, motion confirmation, live operator pauses and cycle limits remain intact. Event-enabled unpaused fake runs idle after exhausting a batch rather than repeatedly picking returned rejects. Queue overflow/producer failures fault the next checkpoint. This is not an immediate/safety stop. Audio readiness/command events are consumed only by the opt-in continuous voice controller; ordinary NOTICE remains informational. No model weights, live overview change detection, or optional second-view matching are included. See `RUNTIME_EVENTS.md` and `VOICE_CONTROL.md` for contracts and boundaries; real hardware behavior remains unvalidated.

## Hardware and runtime

- Target machine: a Linux mini PC (Intel NUC-style) with an Intel Core Ultra Series 3 processor, which has a CPU, an integrated GPU, and an NPU.
- The arm, gripper, and depth camera are controlled by `viam-server` running on that machine, configured at app.viam.com.
- **All model inference runs locally through OpenVINO on that machine.** No cloud inference APIs, no CUDA (there is no NVIDIA GPU).
- Development also happens on laptops without an NPU. Code must run on CPU anywhere and use GPU/NPU when available.

## Repo layout

```
anomaly/   train + export Anomalib model to OpenVINO, threshold calibration, inference wrapper
vlm/       VLM wrapper, prompt template, JSON parsing + validation
robot/     Viam connection, camera/arm/gripper wrappers, pose teaching tool, fakes.py
bench/     device benchmark: latency of each model on CPU / GPU / NPU
config/    config.yaml (item-specific settings), poses.json (taught arm poses)
main.py    capture → detect → decide → sort loop
data/      training/sample images (gitignored)
models/    exported OpenVINO models (gitignored)
logs/      per-item JSONL logs (gitignored)
```

## Commands

These are the intended entry points. Create them if they don't exist yet.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py --fake            # full loop with fake camera + fake arm
python main.py                   # real hardware (see Safety rules)
python anomaly/train.py --data data/<item> --model patchcore   # train, threshold, export
python bench/devices.py          # latency table for every model on every device
python robot/teach.py <name>     # save the arm's current pose to config/poses.json
```

## Conventions

- Python 3.10+. Keep dependencies minimal, and add every new one to `requirements.txt`.
- **Config, not constants.** `config/config.yaml` holds Viam resource names, ROI, model paths, the device for each model, the anomaly threshold, defect types, allowed actions, and the VLM prompt template.
- **Devices.** Never hardcode `"NPU"` or `"GPU"`. Read the device for each model from config and fall back to CPU if it's missing from `ov.Core().available_devices`, logging a warning.
- **Fake hardware.** `robot/fakes.py` provides `FakeCamera` (cycles through `data/samples/good/` and `data/samples/defect/`) and `FakeArm` / `FakeGripper` (log moves, no motion). The `--fake` flag selects them. Every piece of hardware-touching code needs a fake path.
- **Viam SDK** is async (`asyncio`). Read credentials from environment variables or `.env` (`VIAM_ADDRESS`, `VIAM_API_KEY`, `VIAM_API_KEY_ID`). Never commit them.
- **Logging.** For every item, append one JSON line to `logs/`: item id, timestamp, anomaly score, pass/fail, raw VLM output, parsed action, final bin, and per-stage latency in ms (capture, detect, decide, sort). These logs feed the accuracy and latency numbers in the design explanation.

## OpenVINO notes

- Set `CACHE_DIR` on `ov.Core()` so NPU and GPU models don't recompile on every start.
- The NPU needs static input shapes. Reshape models before compiling for NPU.
- Compile and warm up every model once at startup. Never reload a model per item.
- Use `PERFORMANCE_HINT: LATENCY` for per-item inference.
- VLM: keep `max_new_tokens` small, and prompt for JSON only.
- Before claiming a model runs on NPU or GPU, confirm it compiles on that device and that its outputs match CPU.

## Anomalib notes

- Use the Anomalib 2.x API at the version pinned in `requirements.txt`. Many online examples use the old 0.x/1.x workflow (`config.yaml` + `tools/` scripts), so don't copy those.
- Install CPU PyTorch. Never install CUDA builds.
- Use the `Folder` dataset format: `data/<item>/good/` and optionally `data/<item>/defect/`.
- Candidate models: Patchcore (default), Padim, EfficientAd.
- With few defect samples, set the threshold from the distribution of scores on good images (for example, the maximum good score plus a margin) and record how it was chosen.
- Anomaly models are sensitive to background, lighting, and position. Always crop to the ROI and use identical preprocessing for training and inference.

## Verifying work

- After changing pipeline or hardware code, run `python main.py --fake` and report the real output.
- Don't claim something works on NPU/GPU or on real hardware unless it was actually run there.

## Safety rules

- **Never run code that moves the real arm without explicit confirmation from the user in the current session.** Default to `--fake`.
- Keep arm speeds conservative and move through known waypoints. Always be able to return to the home pose.
- Ask before downloading large models (>1 GB) or installing system packages or drivers.
- Don't modify the `viam-server` config or organizer-installed system software.
- Never commit `data/`, `models/`, `logs/`, `.venv/`, or `.env`.
- Commit, or tell the user to commit, before large refactors.

## Current status

Scenario: scattered fasteners, with tape or marker marks as defects. Teammates are handling Anomalib and the vision-language model; this workstream handles localization and robot integration.

Station: armfarm22, hosted on intel-CEXD-INTRBL. Resource names: arm (xArm6), gripper (UFactory), overview-cam (separate webcam), cam (arm-mounted RealSense). Models/SDK adapters are still not fully hardware-verified; poses remain placeholders.

`tools/locate.py` and `robot/locate.py` implement pixel-only background subtraction from the overview camera. Settings are in `config/localization.yaml`. Outputs are frame-local IDs, full-image pixel centers, image-plane long-axis angles, bounding boxes, and individual crops. These are NOT calibrated robot targets or validated grasp points. Touching objects can merge; lighting changes, glare and shadows can create false detections. Real camera accuracy remains to be verified.

Organizer workshop files exist separately under ~/nyc_workshop and ~/workshop, with an Intel-XPU/Anomalib/OpenVINO environment. Their documented robot demo uses simulated SO-101 arms and blocks, not this physical xArm/fastener scenario. Do not overwrite their environment or checkpoints.
