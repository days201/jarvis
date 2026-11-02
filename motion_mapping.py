"""Map image directions to this camera's measured ONVIF motor directions."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


CALIBRATION_PATH = Path(__file__).with_name("motion_calibration.json")


@dataclass(frozen=True)
class MotionMapping:
    pan_right: int = 1
    tilt_up: int = 1
    velocity_pan_right: int | None = None
    velocity_tilt_up: int | None = None
    velocity_single_axis: bool = False

    @property
    def continuous_calibrated(self):
        return self.velocity_pan_right in (-1, 1) and self.velocity_tilt_up in (-1, 1)

    @classmethod
    def load(cls, host, path=CALIBRATION_PATH):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if host and data["host"] == host and all(type(data[key]) is int and data[key] in (-1, 1) for key in ("pan_right", "tilt_up")):
                velocity = [data.get(key) for key in ("velocity_pan_right", "velocity_tilt_up")]
                if not all(type(value) is int and value in (-1, 1) for value in velocity):
                    velocity = [None, None]
                return cls(data["pan_right"], data["tilt_up"], *velocity,
                           velocity_single_axis=data.get("velocity_single_axis") is True)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return cls()


def image_shift(before, after):
    """Measure background displacement using matched features and RANSAC."""
    import cv2
    first = np.asarray(before.convert("L").resize((640, 360)))
    second = np.asarray(after.convert("L").resize((640, 360)))
    detector = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    key1, des1 = detector.detectAndCompute(first, None)
    key2, des2 = detector.detectAndCompute(second, None)
    if des1 is None or des2 is None:
        raise ValueError("The scene has too few features to measure camera direction.")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
    if len(matches) < 20:
        raise ValueError("The before and after views do not have enough reliable matches.")
    source = np.float32([key1[match.queryIdx].pt for match in matches])
    target = np.float32([key2[match.trainIdx].pt for match in matches])
    matrix, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                             ransacReprojThreshold=3, maxIters=3000, confidence=0.99)
    if matrix is None or int(mask.sum()) < 15 or float(mask.mean()) < 0.4:
        raise ValueError("Background movement could not be measured reliably.")
    center = np.array([320, 180, 1], dtype=np.float64)
    shift = matrix @ center - center[:2]
    return {"x": float(shift[0] / 640), "y": float(shift[1] / 360),
            "matches": len(matches), "inliers": int(mask.sum())}
