"""Reusable wrist inspection stage; one compiled model per pipeline run."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
from anomaly.detector import OpenVINODetector
from anomaly.object_crops import automatic_crops, extract_candidate_crops
from anomaly.stub import StubDetector
from robot.locate import annotate
from tools.inspect_wrist import capture, inspect, annotate_scores, validate_rois


class WristInspector:
    def __init__(self, settings, root, fake=False, detector=None):
        self.settings = settings
        self.root = root
        self.fake = fake
        self.threshold = settings['model'].get('threshold')
        if self.threshold is not None:
            self.threshold = float(self.threshold)
            if not math.isfinite(self.threshold):
                raise ValueError('Wrist threshold must be finite or null')
        model_settings = dict(settings['model'])
        model_settings['threshold'] = self.threshold if self.threshold is not None else 0.0
        if detector is not None:
            self.detector = detector
        elif fake:
            self.detector = StubDetector(threshold=self.threshold if self.threshold is not None else 0.5,error_rate=0,seed=7)
        else:
            print(f"Loading wrist anomaly model once: {model_settings['path']}",flush=True)
            self.detector = OpenVINODetector(model_settings,root)

    def process(self, image):
        stamp = datetime.now(timezone.utc)
        folder = self.root / 'logs/wrist_inspection' / stamp.strftime('inspect_%Y%m%dT%H%M%S_%fZ')
        folder.mkdir(parents=True,exist_ok=False)
        image = image.convert('RGB')
        image.save(folder/'frame.png')
        rois = self.settings.get('rois')
        localization = None
        crops = None
        if not rois:
            rois,localization,mask = automatic_crops(image,self.settings['auto_crop'])
            crops = extract_candidate_crops(image,localization,mask)
            annotate(image,localization,mask).save(folder/'detected_objects.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/'foreground_mask.png')
            (folder/'candidates.json').write_text(json.dumps(localization,indent=2,allow_nan=False))
        records = []
        if rois:
            if crops is None:
                validate_rois(image,rois)
            records = inspect(image,rois,self.detector,self.threshold,folder,crops=crops,
                              objects=localization['objects'] if localization else None)
        else:
            print('Wrist camera: no candidate crops; no inference performed')
        scored = annotate_scores(image,records)
        scored.save(folder/'scored_objects.png')
        scored.save(folder/'detected_objects.png')
        report = dict(timestamp=stamp.isoformat(),camera_name=self.settings['camera_name'],fake=self.fake,
                      model=self.settings['model']['path'],device=getattr(self.detector,'device','stub'),
                      settings=self.settings,objects=records,localization=localization,
                      candidate_count=len(records),anomalous_count=sum(r['classification']=='anomalous' for r in records),
                      output_dir=str(folder),preprocessing_verified=False,
                      note='Wrist IDs are frame-local and NOT matched to overview IDs.')
        (folder/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f"Wrist: {len(records)} candidates, {report['anomalous_count']} flagged at threshold {self.threshold}")
        print(f"Wrist scores/overlays: {folder/'scored_objects.png'}")
        return report

    async def capture_and_inspect(self):
        print(f"Capturing {'FAKE' if self.fake else self.settings['camera_name']} for anomaly inspection")
        if self.fake:
            image = Image.new('RGB',(640,480),'gray')
            draw = ImageDraw.Draw(image)
            for box in [(150,150,175,185),(300,240,335,265),(450,330,480,365)]:
                draw.rectangle(box,fill='black')
        else:
            image = await capture(self.settings,False)
        # Local inference is synchronous. The camera connection is closed before inference.
        return self.process(image)
