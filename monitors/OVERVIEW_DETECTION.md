# CPU overview detector: offline first increment

## Status and ownership / coordination request

This detector increment adds `monitors/overview_detection.py`, this document,
`tools/replay_overview_events.py`, and their tests. Repository instructions and
runtime documentation are also clarified on the publication branch. The existing
NUC runtime scaffold from source `9fd2589` is published as a separate prerequisite
commit; its behavior is unchanged. The detector does not modify the audio adapter,
`monitors/overview.py`, runtime/controller/config behavior, packages, models, or
Viam settings. The production overview hook remains inert. Copying these five
new overview files to a deployment does not enable any listener. No live camera
detection or physical event behavior has been validated.

**Audio teammate:** please continue owning your audio implementation. Before
live overview integration, agree on any changes to `ProducerContext`, controller
lifecycle notifications, and shared configuration through `RUNTIME_EVENTS.md`.
This document is a proposed contract, not an agreement or an implemented shared
interface. No new shared runtime interface changes are made in this increment.

## Implemented CPU primitive

- NumPy/OpenCV CPU array operations; no model, GPU/NPU API, camera, credentials,
  or motion access. Existing dependencies only. Actual NUC latency is unmeasured.
- A configured boolean workspace mask and optional static exclusion mask (base,
  non-workspace, optionally GOOD). Fixed base pixels need not be monitored.
- Explicit clear reference, normally associated with a fresh inspection epoch.
  The reference is a populated workspace, not necessarily an empty-table image.
- Per-channel image difference with bounded median illumination compensation.
- Changed pixels must persist for a configured number of consecutive clear frames
  and form a component of sufficient area. Additions, removals, and moves all
  appear as changes; this is NOT object classification or identity tracking.
- Broad persistent differences request reinspection as uncertainty. They are not
  silently accepted as lighting changes or automatically learned as background.
- One `changed` transition; subsequent results latch until explicit new reference.
  A future adapter can publish `EventKind.SCENE_CHANGED` on that transition.
- `clear_view=False` prevents comparison and clears persistence, but preserves
  the old reference. After the arm clears, changes during occlusion are visible.
- `accept_expected_change` copies ONLY a supplied, independently verified patch
  into the reference. An unexpected change elsewhere is still detected. It
  cannot clear a latched event or accept an obscured frame.

Masks and boxes use **overview-image pixels**, with exclusive right/bottom box
edges. Wrist boxes cannot be used here without a validated mapping. Inputs must
be equal-sized RGB uint8 images and boolean masks. Replay masks are binary PNGs:
white means workspace/included, or excluded for an exclusion mask. No automatic
resizing: prepare images and masks at the same chosen processing resolution.
Parameters are configurable through `DetectionSettings` / the replay manifest;
defaults are starting values, not calibration for small reflective fasteners.
Frame persistence depends on sampling cadence, which must be bounded in the live
adapter. Stale frames and long gaps require invalidating persistence.

## What is deliberately NOT solved yet

1. **Arm visibility:** this core requires an explicit, trustworthy `clear_view`.
   It does not detect the arm or hands. A stationary arm can still obscure the
   table; an idle motion controller alone is NOT proof of a clear view. Initial
   integration should conservatively wait for verified unobscured views, retaining
   the old reference across movement. If the view never clears, it cannot monitor
   that region. Live monitoring must report prolonged loss of visibility rather
   than implying the table is unchanged. Neither fixed masks nor this detector
   solve permanently hidden regions behind the base.
2. **Expected picks:** no automatic overview/wrist identity association exists.
   The patch-update method is only a primitive for a future verified association.
   It cannot distinguish an unrelated change inside the same patch. Do not guess
   matches between similar fasteners, mask the entire workspace during an update,
   or rebaseline the whole image after every delivery. If the expected removal or
   reject return cannot be verified, request fresh inspection instead. Otherwise
   normal picks will correctly look like changes and cause unnecessary resets.
3. **Baseline/reset handshake:** references must be tied to the fresh inspection
   epoch, acknowledged by the sole controller, and not refreshed before it has
   consumed a reset. A latch alone is insufficient for live integration: changes
   after a latch but before fresh capture and changes during that capture must
   not get lost. This needs controller lifecycle coordination, not polling guesses.
4. **Capture/lifecycle:** the eventual producer owns a separate camera-only Viam
   client in its own thread/event loop, with bounded connection/capture waits,
   fresh-frame validation, low-rate polling, and cleanup. Camera failures should
   propagate to the existing runtime fault latch, not masquerade as no changes.
   Fake mode must never create a hardware client. No connection code exists here.

A conservative detector can report uncertain changes; it cannot establish human
safety, path clearance, or selected-target stability. Existing between-attempt
checkpoints remain the only consumers. Never interrupt a grasp/delivery with a
home or gripper-open command. Runtime listeners remain disabled pending physical
baseline validation and coordinated integration.

## Offline replay

Keep private frames/masks/manifests under ignored `logs/` or `data/`, never commit
them. Paths are relative to the manifest. Example `logs/overview_replay/run.json`:

```json
{
  "workspace_mask": "workspace.png",
  "excluded_mask": "base.png",
  "settings": {"difference_threshold": 25, "min_area_px": 12, "persistence_frames": 3},
  "samples": [
    {"image": "before.png", "operation": "reference", "clear_view": true},
    {"image": "arm_present.png", "clear_view": false},
    {"image": "after_1.png", "clear_view": true},
    {"image": "after_2.png", "clear_view": true},
    {"image": "after_3.png", "clear_view": true}
  ]
}
```

```bash
.venv/bin/python tools/replay_overview_events.py logs/overview_replay/run.json
.venv/bin/python -m pytest tests/test_overview_detection.py tests/test_overview_event_replay.py -q
```

The command only reads saved files and prints JSONL diagnostics with hypothetical
`would_publish` events. It does not connect to Viam or enqueue anything. Repeating
a single file can exercise persistence but is not evidence of temporal accuracy.
For a manually verified expected removal, a sample may specify
`"operation": "expected", "expected_mask": "verified_pick_patch.png"` with a clear
post-delivery image. Use a new `reference` sample only for a fresh inspection epoch.

Next evidence needed: a saved overview sequence with the fixed base, arm movement,
normal deliveries, and deliberate object additions/removals/moves, plus reviewed
workspace/base masks. Validate CPU latency, false resets, missed changes, camera
shake/exposure effects, and small low-contrast parts before enabling live events.
