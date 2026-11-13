# Overview hover demo

```bash
cd ~/viam-hackathon
source .venv/bin/activate
python tools/overview_hover.py
```

Default: capture overview, find candidates, annotate estimated robot XY and print
simulated visits. No arm resource is constructed. `--image PATH` uses saved RGB.
`--fake` substitutes a synthetic camera (requires a matching reference to localize).

## Collision-checked planning, no movement

```bash
python tools/overview_hover.py --ids 4 --plan
```

Review the NEW run's `targets.png`: IDs can change between frames. The calibrated
camera must not move. Background subtraction can mistake hands, shadows, markers,
and clutter for objects. Merged objects are not valid individual targets.

The planner reads current arm pose, gripper bounding-box geometries, and frame
transforms. Holding the current downward orientation, it derives the endpoint Z
that puts the modeled lowest gripper point 101.6 mm above the assumed tabletop
Z=-23 mm. It accounts for wrist-camera mounting offset and optical-axis tilt to
center the wrist view over each target. Target XY comes from a provisional mapping,
not independently validated robot calibration. Targets outside calibration coverage
are rejected; coverage is not a reachability guarantee.

All stages are planned BEFORE any execution. Each target uses higher-altitude XY
travel then descent, with linear motion profile and configured obstacle collision
checking via Viam's builtin motion service. The entire route is rejected if any
stage fails, is partial, or moves another resource. Planned joints seed later stages.

## Supervised execution

```bash
python tools/overview_hover.py --ids 4 --execute --confirm-motion --accept-provisional-calibration
```

Execution captures and plans a NEW scene. Review that run's targets before typing
MOVE. Stop concurrent teleop/controllers, clear the workspace, and have the physical
emergency stop ready. A start-joint check rejects stale trajectories. Runtime driver
commands set 5 deg/s speed and 10 deg/s² acceleration. These settings are left in
place (not restored to the earlier faster setting). No cloud config is changed.

Robot collision models do not capture people or unconfigured obstacles, and modeled
gripper bounds are not a physical measurement. The assumed table and mounting
geometry remain unverified. Keep the test supervised; do not bypass planner failures.
There is no grasp, gripper actuation, or automatic return home. Failure during motion
attempts arm.stop; an emergency stop must remain available.

Results: frame, targets image/JSON, live geometry, and planned trajectories under
logs/overview_hover. Unit tests and one live two-stage plan were verified; physical
execution has not yet been tested.
