"""First pipeline stage: camera-only foreground candidate counting."""
import copy
from datetime import datetime, timezone
import json
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml
from robot.camera import capture_camera
from robot.locate import locate_objects, annotate
from robot.home import home_before_capture


def count_candidates(image, background, settings, maximum):
    result, mask = locate_objects(image, background, settings)
    if not isinstance(maximum, int) or maximum < settings['min_area_px']:
        raise ValueError('overview.max_area_px must be >= locator min_area_px')
    accepted, rejected = [], []
    for obj in result['objects']:
        reasons = list(obj['warnings'])
        if obj['area_px'] > maximum:
            reasons.append('exceeds_max_area')
        if reasons:
            obj['rejection_reasons'] = reasons
            rejected.append(obj)
        else:
            accepted.append(obj)
    for index, obj in enumerate(accepted, 1):
        obj['id'] = index
    result.update(objects=accepted, candidate_count=len(accepted), rejected_candidates=rejected)
    return result, mask


async def overview_stage(config, root, fake=False, image_path=None, confirm_motion=False):
    settings = config['overview']
    locator = yaml.safe_load((root / settings['locator_config']).read_text())
    start = time.perf_counter()
    home_status = 'skipped_offline' if image_path else None
    if fake:
        home_status = await home_before_capture(config, root, True, False)
        # Deterministic synthetic pair: test segmentation without camera, credentials or models.
        background = Image.new('RGB', (640,480), 'white')
        image = background.copy()
        draw = ImageDraw.Draw(image)
        for box in [(150,150,180,200),(290,210,330,250),(430,300,480,320)]:
            draw.rectangle(box, fill='black')
        source = 'synthetic: three separated shapes (not hardware validation)'
    else:
        with Image.open(root / settings['background']) as reference:
            background = reference.convert('RGB')
        if image_path:
            with Image.open(image_path) as saved:
                image = saved.convert('RGB')
            source = str(image_path)
        else:
            home_status = await home_before_capture(config, root, False, confirm_motion)
            camera_config = copy.deepcopy(config)
            camera_config['hardware']['viam']['camera'] = settings['camera_name']
            print(f"Capturing {settings['camera_name']} — camera only, no arm/gripper commands")
            async with capture_camera(camera_config, root, False) as camera:
                image = (await camera.get_frame()).image
            source = settings['camera_name']
    capture_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    result, mask = count_candidates(image, background, locator, settings['max_area_px'])
    detection_ms = (time.perf_counter()-start)*1000
    stamp = datetime.now(timezone.utc)
    folder = root / settings['output_dir'] / stamp.strftime('overview_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    result.update(timestamp=stamp.isoformat(), source=source, fake=fake, home_status=home_status, output_dir=str(folder),
                  capture_ms=capture_ms, detection_ms=detection_ms,
                  background=None if fake else settings['background'], settings=locator,
                  max_area_px=settings['max_area_px'])
    image.save(folder/'frame.png')
    annotated = annotate(image, result, mask)
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((0,0,330,24),fill='black')
    draw.text((5,5),f"Candidate objects: {result['candidate_count']} (review detections)",fill='white')
    annotated.save(folder/'detected_objects.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(folder/'foreground_mask.png')
    (folder/'objects.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f"Overview: {result['candidate_count']} candidate objects; {len(result['rejected_candidates'])} rejected regions")
    print(f"Detection time: {detection_ms:.1f} ms")
    print(f"Annotated image: {folder/'detected_objects.png'}")
    print(f"Object coordinates: {folder/'objects.json'}")
    print('Overview stage complete. No coordinate mapping or sorting movement.')
    print('Count is provisional: touching parts may merge; shadows/clutter may be counted.')
    return result
