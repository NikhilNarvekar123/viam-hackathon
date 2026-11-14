# Taught wave and dance paths

The saved Viam switch resources were inspected read-only on the NUC: `home-pose`,
`left_wave`, `right_wave`, `up_wave`, and `down_wave` each contained six finite
saved joint values. **No real gesture was executed or path clearance validated.**

`config/gestures.yaml` defines:

- **hello / wave:** home → left → right → left → right → home (two left/right cycles).
- **dance:** home → left → up → right → down → home (one cycle).

These replace the earlier invented offset/clapping routines. There are no gripper
open/close/grab commands in a gesture. The executor reads all required saved poses
before movement, converts the saved radians to SDK degrees, and freezes that
snapshot for the routine. It sends sequential joint-position moves and verifies
arrival at each waypoint. It never records/overwrites a saved pose or uses switch
position 1. Poses are referenced by resource name, not copied into the repository.

## Voice behavior

With `main.py --continuous --voice`, a validated `hello` or `dance` requests its
routine. If a pick is underway, its attempt and delivery/return must finish first.
The old batch is discarded. Once the arm is idle and the gripper is verified empty,
the controller executes the gesture and remains stopped afterward. Say a **fresh
start after the completion log** to inspect again. Buffered starts spoken before or
during the gesture cannot automatically resume picking.

A stop or new policy cancels a pending gesture. During a gesture it cancels the
remaining waypoints at the next completed-waypoint checkpoint; it does not interrupt
a moving waypoint. A new policy remains pending for later confirmation by start.
Scene resets/caller shutdown also cancel remaining waypoints. Voice remains a soft
checkpoint control, not an emergency stop.

Successful routines finish at the saved home. Interrupted routines stay at the last
completed waypoint. A movement failure, timeout, or cancellation attempts arm stop
and propagates the fault: **no automatic home or release on failure**. Only a later
authorized operation may move again.

## Preconditions and limits

- Real execution requires the existing explicit `--confirm-motion` authorization.
- Teleop must be stopped, arm idle, and gripper confirmed empty before each waypoint.
- A missing/invalid saved pose refuses the entire routine before any movement.
- Joint speed/acceleration use the configured home settings; restoring a config file
  does not update values already cached in a running controller.
- Saved endpoints are **not proof that the connecting joint-space paths clear the
  table, people, or obstacles**. The sequences are fake/mock tested only; review the
  taught poses and transitions before a separately authorized, supervised live run.
- Do not run the standalone gesture tool alongside another controller.

## Fake checks

```bash
python tools/gesture.py hello --fake
python tools/gesture.py dance --fake
python -m pytest tests/test_gestures.py tests/test_voice_control.py tests/test_continuous_voice.py -q
```

Fake mode uses synthetic joint positions and never connects to read machine poses.
The standalone tool also supports `--confirm-motion` for explicitly authorized live
use; it is not started by deployment. Configure `enabled: false` in
`config/gestures.yaml` to refuse gesture execution rather than silently performing
an untaught fallback.
