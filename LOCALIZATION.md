# Overview-camera localization (no robot movement)

On the NUC:

```bash
cd ~/viam-hackathon
source .venv/bin/activate
```

## 1. Empty-table reference

Stop teleop and keep the arm/camera stationary. Remove all fasteners/hands from
the view. Keep illumination and camera exposure stable. Capture once:

```bash
python tools/locate.py --capture --save-background data/overview_empty.png
```

Requires viam-sdk and the project's .env, just like the gripper demo. Only the
camera resource is used. Nothing moves and no Viam configuration is changed.
Existing reference files are never overwritten; use another filename if needed.

## 2. Locate objects

Put separated fasteners on the table; remove your hands from the view:

```bash
python tools/locate.py --capture --background data/overview_empty.png
```

Each run writes a new folder under `logs/localization/` containing:
- `frame.png`: original camera frame.
- `annotated.png`: green boundaries, yellow boxes/IDs, red centers, cyan long axes.
- `foreground_mask.png`: detected foreground, excluding small components.
- `crops/object_001.png`, etc.: original RGB crops, not annotated images.
- `objects.json`: object IDs, pixel centers, angles, boxes, crop paths, settings, timestamp.

Open the annotated image in remote VS Code. Review detections before feeding crops
into models. Re-run after objects/arm move; IDs do not track objects across frames.

## Tune

Edit `config/localization.yaml` (not the Viam machine configuration):
- `workspace_roi`: usable tabletop in FULL-image pixels [left, top, right, bottom].
- `difference_threshold`: raise to suppress small lighting changes; lower for weak contrast.
- `min_area_px`: reject noise without discarding the smallest fasteners.
- `closing_radius_px`: fills small gaps; can also merge nearby objects.
- `crop_padding_px`: keep consistent with the teammates' training data.

Pixel bounds use right/bottom-exclusive convention, matching PIL crop and NumPy slicing.
Angles are image-plane long-axis estimates, clockwise from the image's +u direction,
modulo 180 degrees. Round/symmetric objects return null orientation. These are NOT
gripper yaw commands. Centers can lie in a nut's hole; they are NOT grasp points.

## Offline use

```bash
python tools/locate.py --image data/table_frame.png --background data/overview_empty.png
```

## Limits

This is background subtraction, not a learned object detector. Touching objects can
merge; one shiny object can fragment. Shadows, reflections and camera/lighting motion
can produce false detections. A large foreground fraction aborts rather than reporting
misleading targets. Keep the background viewpoint identical. Synthetic tests pass;
accuracy on the actual camera/fasteners is not yet verified.

No robot coordinates, depth fusion, camera calibration, grasp planning or movement
are implemented by this locator. It is not yet wired into main.py.
