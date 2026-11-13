"""Ground-truth-driven simulator, NOT an image anomaly model."""
import random

from core.types import Detection, Frame


class StubDetector:
    def __init__(self, threshold: float, error_rate: float, seed: int):
        if not 0 < threshold < 1 or not 0 <= error_rate <= 1:
            raise ValueError("threshold must be in (0, 1), error_rate in [0, 1]")
        self.threshold = threshold
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    def detect(self, frame: Frame) -> Detection:
        if frame.ground_truth is None:
            score = self.rng.random()
        else:
            anomalous = frame.ground_truth == "defect"
            if self.rng.random() < self.error_rate:
                anomalous = not anomalous
            value = self.rng.random()
            score = (self.threshold + value * (1 - self.threshold)
                     if anomalous else value * self.threshold)
        return Detection(score, score >= self.threshold)
