# Live overview change-detector diagnostic (no robot actions)

This standalone runner connects **only to the overview camera**, feeds sequential
images into `OverviewChangeDetector` on CPU, and reports hypothetical scene-change
events. It does not construct arm/gripper/motion resources, load inspection models,
publish to the runtime queue, or enable a producer. No Viam config is changed.

## Run on the NUC

Keep the arm stationary, ensure no other picking/teleop controller is operating,
and keep the camera fixed. A stationary base is part of the reference/background.
The initially visible tabletop can contain objects; it does not need to be empty.

```bash
cd /home/intel/viam
source .venv/bin/activate
python tools/watch_overview_changes.py --record --seconds 120
```

No keyboard input is required. The first capture becomes the reference, then the
runner polls **every 2 seconds** (0.5 Hz). Add, remove, or move an object, then
withdraw hands. With the default 3 persistent samples, confirmation takes roughly
4–6 seconds plus capture/processing time. These thresholds are uncalibrated, not
a timing guarantee. Use `--persistence 1` only for single-sample diagnostics (more
susceptible to noise and hands). Ctrl+C exits; the duration defaults to 120 seconds.

After each `SCENE_CHANGED`, automatic diagnostic mode accepts that **same captured
frame** as its new reference and continues watching. No later capture is silently
swallowed as a reference. This is a diagnostic acknowledgment only, NOT the
production controller reset/inspection handshake. The core detector and production
runtime are unchanged. Do not reuse this auto-acknowledgment in the live picker.

Optional `--manual` mode requires an interactive terminal and initial ENTER. It
provides these commands, each followed by ENTER:

- `r`: explicitly accept the next clear capture as a new reference and rearm.
  Use only once the workspace is stable and you have reviewed the change.
- `o`: mark the workspace obscured; suspend comparison **without erasing reference**.
- `c`: mark it clear again; compare against the old reference. A change made during
  an obscured interval can then trigger detection.
- `q` or Ctrl+C: exit. Duration defaults to 120 seconds.

In `--manual` mode only, one event is reported per reference epoch and a changed
scene remains `latched` until `r`. Clearing occlusion with `c` preserves the old
reference; it never automatically accepts changes made during the obscured interval.

## Masks and tuning

The processing image is downsampled to at most 640 pixels wide (aspect preserved).
The runner prints its exact size. `--roi X0 Y0 X1 Y1` limits the workspace, and
repeatable `--exclude X0 Y0 X1 Y1` masks the base or non-workspace. Coordinates are
in the **resized processing image**, not raw camera or wrist pixels. Right/bottom
edges are exclusive. Whole-image comparison is the diagnostic default, so changes
outside the tabletop can also trigger until a workspace ROI is set.

Other controls: `--width`, `--hz` (maximum 10), `--threshold` (RGB difference),
`--min-area` (connected changed pixels), `--persistence` (consecutive clear samples),
`--camera` (default `overview-cam`), `--timeout` (seconds). No settings are written
back to the pipeline config. Tune with saved evidence, not to suppress real changes.

## Output and replay

The printed `logs/overview_changes/watch_<timestamp>/` directory contains:

- `latest.png`: latest processing image with status and changed-region boxes.
- `reference_*.png`, `change_*.png`: accepted references / first confirmed changes.
- `workspace.png`, `excluded.png`: processing masks.
- `detections.jsonl`: status, bounds, hypothetical event, and timing per sample.
- With `--record`: every processing frame and `replay.json` for the offline tool.
  Automatic rearming adds an explicit reference operation using the same frame,
  so replay reproduces subsequent changes instead of remaining latched.

```bash
python tools/replay_overview_events.py logs/overview_changes/watch_<timestamp>/replay.json
```

Private images/logs stay local and gitignored. Recording consumes disk space with
duration/resolution/rate; keep runs short. `latest.png` is overwritten each sample;
this is not an OpenCV GUI window (the installed OpenCV may be headless).

## Fake smoke test

```bash
python tools/watch_overview_changes.py --fake --hz 2 --record --seconds 5
```

Synthetic frames show a fixed base and an object appearing after three captures.
Expect one diagnostic `SCENE_CHANGED`, automatic rearming, then `unchanged`.
Fake mode does not read credentials/config or access camera/model resources. It
validates wiring, not real camera accuracy.

## Limits

**Automatic arm/hand visibility detection is NOT implemented.** This runner is for
stationary-arm tests and manually labelled occlusion experiments. Unmarked arm,
hand, shadow, exposure, reflection, or background changes may trigger false events.
The `o/c` controls are annotations for testing, not motion or safety interlocks.
Never operate a robot near hands to test this script. Regions hidden behind the
base cannot be monitored. It does not associate expected picks with overview
objects or validate selected-target stability/path clearance.

Capture calls are sequential and bounded; a timeout/failure ends the run without
retry. Source timestamp/freshness validation is not implemented by the existing
camera-only wrapper, so repeated cached images cannot establish temporal accuracy.
The live producer still needs verified visibility, frame freshness, expected-pick
association, and the controller reference/reset handshake before integration.
