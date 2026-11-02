"""Recent camera frames and local presence detection. Nothing is recorded."""

import base64
from collections import deque
from dataclasses import dataclass
import io
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageFilter, ImageStat


class VisionError(Exception):
    """A camera-view error safe to display."""


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    kind: str = "face"

    @property
    def center(self):
        return self.x + self.width / 2, self.y + self.height * (0.5 if self.kind == "face" else 0.22)


class FrameStore:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.condition = threading.Condition()
        self.frames = deque(maxlen=8)
        self.serial = 0

    def push(self, image):
        now = self.clock()
        with self.condition:
            # Keep recent frames for tracking without recording the video.
            if self.frames and now - self.frames[-1][0] < 0.075:
                return
            self.serial += 1
            self.frames.append((now, self.serial, image.copy()))
            self.condition.notify_all()

    def latest(self, max_age=2.0):
        with self.condition:
            if not self.frames or self.clock() - self.frames[-1][0] > max_age:
                return None
            when, serial, image = self.frames[-1]
            return when, serial, image.copy()

    def clear(self):
        with self.condition:
            self.frames.clear()
            self.condition.notify_all()

    def snapshot(self, after=0.0, cancelled=None, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise VisionError("Camera look was cancelled.")
                now = self.clock()
                choices = [(when, image) for when, _, image in self.frames
                           if when >= after and now - when <= 0.6]
                if choices:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VisionError("There is no fresh camera image. Connect the camera and try again.")
                self.condition.wait(min(0.05, remaining))
        def sharpness(item):
            small = item[1].convert("L").resize((160, 90))
            return ImageStat.Stat(small.filter(ImageFilter.FIND_EDGES)).var[0]
        _, image = max(choices, key=sharpness)
        image = image.convert("RGB")
        image.thumbnail((960, 720), Image.Resampling.LANCZOS)
        data = io.BytesIO()
        image.save(data, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode("ascii")


class PersonDetector:
    """Detect faces without identifying them; use full-body HOG as a fallback."""

    def __init__(self):
        import cv2
        self.cv = cv2
        cv2.setNumThreads(2)
        path = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
        self.face = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.8, 0.3, 1000)
        self.body = cv2.HOGDescriptor()
        self.body.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image):
        image = image.copy()
        image.thumbnail((640, 480))
        pixels = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        height, width = pixels.shape[:2]
        self.face.setInputSize((width, height))
        _, faces = self.face.detect(pixels)
        if faces is not None:
            return [Detection(max(0, float(row[0]) / width), max(0, float(row[1]) / height),
                              float(row[2]) / width, float(row[3]) / height)
                    for row in faces if float(row[14]) >= 0.8]
        if width < 64 or height < 128:
            return []
        boxes, scores = self.body.detectMultiScale(pixels, winStride=(8, 8), padding=(8, 8), scale=1.08)
        return [Detection(x / width, y / height, w / width, h / height, "person")
                for (x, y, w, h), score in zip(boxes, scores) if float(score) >= 0.5]


class TargetTracker:
    def __init__(self):
        self.target = None
        self.last_seen = -100.0
        self.blocked_until = 0.0

    def update(self, detections, now):
        if not detections:
            if self.target is not None and now - self.last_seen > 0.8:
                self.target = None
                self.blocked_until = now + 1.5
            return None
        if now < self.blocked_until:
            return None
        if self.target is not None:
            x, y = self.target.center
            candidate = min(detections, key=lambda box: (box.center[0] - x) ** 2 + (box.center[1] - y) ** 2)
            if (candidate.center[0] - x) ** 2 + (candidate.center[1] - y) ** 2 > 0.3 ** 2:
                # Hold the current target instead of jumping to another person.
                if now - self.last_seen > 0.8:
                    self.target = None
                    self.blocked_until = now + 1.5
                return None
        else:
            candidate = max(detections, key=lambda box: box.width * box.height)
        self.target = candidate
        self.last_seen = now
        return candidate
