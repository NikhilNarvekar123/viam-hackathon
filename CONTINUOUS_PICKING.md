# Continuous home-view validation loop (opt-in)

Implemented behind `main.py --continuous`. This is an incremental alternate
orchestration mode, NOT a new picker or a safety system. Fake/mock tested only;
physical home-view consistency, thresholds, and added movements need validation.
The existing default batch mode remains unchanged.

## Ownership and coordination

- Main/controller owns all motion. Audio runs in its own producer thread; the
  `run(context)` producer interface is retained. No arbitrary command dispatch.
- `reset_batch` and `scene_changed` invalidate pending work; `notice` remains
  informational. Optional `--voice` handles typed `audio_ready` / `voice_command`
  events, starts stopped, and requires start to resume/reset. Stop waits for the
  completed delivery/return. This is NOT an immediate or safety STOP.
- New policies are staged/logged, then a later start confirms them for Qwen and
  fresh inspection. Custom policies disable fast-pass. Wave/dance are disabled
  placeholders. See `VOICE_CONTROL.md` for semantics, tests, and constraints.
- Camera/audio settings must be verified before deployment. Hooks remain disabled
  by default. The standalone camera diagnostic does NOT publish to the controller.
- Continuous mode does not call the old overview-counting stage. It uses wrist
  camera `cam` at the saved home pose, so it does not assume the separate camera
  remains overhead. Side-camera height estimation/association belongs to its owner.

## Operation

1. Verify idle teleop/arm and empty gripper, replay the saved home pose (position 2),
   open, and capture stationary wrist RGB/depth using the existing ROI pipeline.
2. Run Anomalib and VLM routing once per new inspection epoch. Keep the original
   boxes/verdicts and existing tallest-first ordering. No second target selection.
3. Check channel requests/faults. Also compare two fresh stable home-view RGB
   captures before using the inspection output, to catch changes during inference.
4. Execute one attempt and complete delivery/return. Nothing consumes events or
   homes/opens between grasp and delivery completion.
5. Verify the gripper is empty BEFORE replaying home/open. At home, check channels,
   then capture two RGB-only views and validate the remaining scene. Check channels
   again after capture. HOLD/refusals and the last attempt also get this boundary.
6. Channel reset, unexpected change, uncertainty, or an expired pending batch:
   discard the remaining geometry/verdicts and start fresh home/capture/inspection.
   Otherwise reuse the original remaining work and proceed to the next target.
7. Exhausted pending work idles at home, checking channels and wrist images every
   configured interval. No repeated classification, returned-reject picking, or
   HOLD/refusal retry while the scene stays unchanged. It is NOT proof of an empty
   table. A later scene change/reset starts a new epoch.

No ENTER pauses and no cycle limit in this mode. `--n`, pause overrides,
`--inspect-only`, offline images, and independent-picker selection are rejected
when combined with `--continuous`, rather than silently ignored.

GOOD placement counters survive resets. The existing GOOD area capacity still
stops the run; indefinite operation does not authorize placing over occupied spots.
Faults/camera errors/holding-state uncertainty/delivery failure stop progression;
there is no automatic fault recovery or automatic home/open after delivery failure.
Ctrl+C uses existing context cleanup/stop attempts, not a guaranteed physical stop.
Keep machine Stop / physical E-stop available. `--confirm-motion` remains required
for live motion and is NOT bypassed by events or this new mode.

## Home-view CPU validation

Configuration: `config/continuous_pick.yaml`. Thresholds are uncalibrated starting
values. This does not restore the old brittle whole-scene object-count/center gate.

- Require idle arm and joint positions matching the original home capture.
- Require positive fresh camera capture timestamps; stale, repeated, missing or
  future timestamps fail closed. SDK/driver freshness support must be verified.
- Compare two nearby frames for stability; unstable images request fresh inspection.
- Compare against the original home-view image with bounded illumination tolerance.
  Significant changes outside any verified expected-removal patch invalidate the
  batch. No model calls or new geometry are used by this comparison.
- Following confirmed GOOD or REJECT placement, allow the original object's small image patch
  to disappear. Require actual disappearance, no residual segmented foreground,
  and resemblance to nearby original tabletop. Refuse patches overlapping another
  original object or containing other original foreground. Only that patch is
  updated; changes elsewhere are never silently accepted as a new baseline.
- REJECT to a destination area now counts as removal, but reject-return and full-area
  put-backs do not: the destination placement counter must have advanced on successful
  delivery. A displaced/reoriented return may trigger reinspection; we do not guess
  identity or silently accept it. HOLD, misses and refusals get no removal exemption.

This is conservative image consistency checking, not robust object tracking.
Shadows, glare, pose repeatability, low contrast, occlusion and segmentation errors
can cause false resets or missed changes. Two stable images cannot prove no hand
is present. A new object visually identical to tabletop, or changes beneath other
objects, may be invisible. Humans must remain outside the robot workspace. Normal
picker refinement, size/height/grasp and clearance checks stay in force.

## Commands and artifacts

```bash
# Noninteractive fake loop: three simulated deliveries, then idle until Ctrl+C.
python main.py --continuous --fake

# Live entry point, only with fresh operator authorization and workspace checks:
# python main.py --continuous --confirm-motion
```

With `--voice`, commands are consumed before any additional post-delivery home/open;
while stopped, only a fresh start resumes, never a scene-change event. Gestures do not
move at all. Pausing preserves placement counters and faults. See `VOICE_CONTROL.md`.

Models load once for the process. Changing a model path requires a new process to
ensure the new model is used; never start a second live controller alongside one.
No real motion was performed to validate this implementation.

Each inspection directory retains normal crop/classification artifacts,
`batch_pick.json` and `cycle.json`. `home_checks/` contains configuration and each
pair of validation images/results, including consumed events. These are private,
gitignored diagnostics, not files to publish. Unchanged idle samples overwrite
`home_checks/idle_latest/` rather than accumulating images indefinitely; a reset
preserves that sample under its numbered directory. New inspection epochs still
accumulate normal pipeline artifacts, so manage disk space for prolonged operation.
A disk-write failure propagates.

```bash
python -m pytest tests/test_home_scene.py tests/test_continuous_pick.py -q
```

Before merging/deploying shared controller changes, coordinate with the audio and
side-camera owners. Validate on saved wrist sequences, then supervise a small
physical run under separate explicit motion authorization.
