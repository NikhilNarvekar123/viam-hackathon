# Main pipeline: supplied boxes and batch picking

**Batch grasp/placement is not physically validated.** Fake tests exercise the
integration; the read-only live mapping audit found valid matches for some, not
all, objects. Stop competing controllers, clear the workspace/GOOD area and keep
the machine Stop / physical E-stop available. Keep the scene unchanged throughout
a batch, until the Enter pause.

## Run

```bash
.venv/bin/python main.py --fake --n 1
# ONE complete batch, not one object:
.venv/bin/python main.py --confirm-motion --n 1
# Complete batches with Enter between batches:
.venv/bin/python main.py --confirm-motion
# Inspection only:
.venv/bin/python main.py --confirm-motion --inspect-only
# Retained older picker-owned selection/reclassification, one target per cycle:
.venv/bin/python main.py --confirm-motion --pick-target-source picker
```

Default `pick_target_source: pipeline` in `config/config.yaml` supplies the
pipeline's boxes and original verdicts. `picker` retains the older independent
selection mode. The original picker detection code is NOT removed: it still
provides depth localization, close-up refinement, dimensions and clearance checks.
Standalone `pick_bolt.py` motion remains disabled to avoid its accept-all stub;
`--stage pick` retains the older independent inspected-picking shortcut.

## External-box mode

1. Read-only preflight checks teleop/arm state and whether an item is held before
   main homes/opens. Replay and verify organizer home (position 2, never save).
2. Capture/count the overview as before.
3. Capture wrist RGB and fused point clouds sequentially while stationary, with
   camera-to-world transform and joint-state checks. This is not a claim of
   hardware-synchronized RGB/depth. The exact captured RGB is passed to our existing
   cropper, Anomalib and VLM pipeline.
4. Measure raw valid depth within each supplied RGB foreground ROI and project
   its center/footprint using the fitted table plane and camera transform. No
   independent depth proposal, pickable kind, 5 mm raised-blob threshold or 12-pixel
   center match is required for initial localization. Missing depth still refuses
   localization rather than inventing a target.
5. After the ONE batch classification pass, process targets tallest first, using
   the configured height percentile (95th by default). Reproject each supplied ROI
   into the current scan and close-up views; measure its geometry directly there.
   External-ROI mode always performs close-up measurement before grasping. Original
   physical size/height eligibility, grasp geometry, clearance checks and delivery
   helpers remain. A low initial reading can approach for a better measurement,
   but a close-up still below the original gripper height limit is not grasped.
   Native picker selection remains optional and retains its original auto-close-up rule.
6. Pass the original stored ACCEPT/REJECT directly into its decision hook:
   **no second classification/cropping pass** and **no added whole-scene comparison**.
7. ACCEPT: pick and place in the existing GOOD area starting at (300,250) mm.
   REJECT: pick and return to the original pickup location. HOLD: do not grasp.
   Each bound crop is attempted once; no failed-object retries inside the batch.
8. Only after the complete batch does main pause for Enter. The placement count
   persists across batches; object IDs/bindings do not.

Original clearance refusals still apply. Measured height can be noisy; tallest-first
ordering does not guarantee clearance. Accepted tall objects move to GOOD, but
rejected tall objects return to their original spots and can still block later
picks. `pick_order: input` restores original crop order. No neural inference or
whole-scene consistency check is added during ROI refinement. Stale batches expire at the
configured age limit; the source verdict must not follow a rearranged scene.
This is not robust object tracking or a personnel-safety system.

## Configuration and artifacts

- `config/config.yaml`: `pick_after_inspection`, `pick_target_source`.
- `config/pick_bolt.yaml`: destinations, matching radius, batch age, GOOD capacity.
- `config/crop_target_mapping.yaml`: initial, uncalibrated association gates.
- `config/pick_bolt_controller.yaml`: active imported controller settings. Upstream
  constants remain as reference but YAML owns the active values.
- Existing wrist/VLM configs still own model threshold, preprocessing and prompts.
- Each wrist inspection directory contains `picker_bindings.json`, exact
  `frame.png`, live `picker_depth.npy`, classification artifacts, `batch_pick.json`
  and main's `cycle.json`. Reports distinguish fake actions from physical actions.

The inherited 35 mm GOOD placement pitch and destination are not independently
validated and do not guarantee clearance for long parts or an occupied area.
Clear/check the area between batches. Maximum capacity is 12 placements.
Cancellation/errors attempt `robot.stop_all()`; failed network stops require the
machine Stop or physical E-stop. No physical run is performed by software tests.
