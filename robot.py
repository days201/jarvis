"""Bounded camera actions and local tracking shared by the UI and DeepSeek."""

from contextlib import contextmanager
import math
import re
import threading
import time

from camera import CameraError
from motion_mapping import MotionMapping
from tracking_motion import PositionController
from vision import FrameStore, PersonDetector, TargetTracker, VisionError


def function(name, description, properties=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": list(required), "additionalProperties": False}}}


TOOLS = [
    function("look", "Turn the camera in a direction, or return to its desk position. Then capture a fresh frame if needed.",
             {"direction": {"type": "string", "enum": ["left", "right", "up", "down", "desk"]},
              "degrees": {"type": "number", "minimum": 1, "maximum": 45}}, ["direction"]),
    function("look_at", "Center on an image point. Coordinates run from 0 to 1, left to right and top to bottom.",
             {"x": {"type": "number", "minimum": 0, "maximum": 1},
              "y": {"type": "number", "minimum": 0, "maximum": 1}}, ["x", "y"]),
    function("follow_person", "Enable or disable local person following. It does not identify people by name.",
             {"enabled": {"type": "boolean"}}, ["enabled"]),
    function("stop_motion", "Stop movement and pause automatic following until it is explicitly enabled again."),
    function("gesture", "Make one small nod or curious pan/tilt movement, then return. Use sparingly.",
             {"kind": {"type": "string", "enum": ["nod", "curious"]}}, ["kind"]),
    function("capture_frame", "Look at the current camera view. Use for visual questions or after moving; at most two frames per turn."),
]


def needs_view(text):
    return bool(re.search(r"\b(look|see|seeing|show|holding|wearing|read|color|colour|picture|camera|this|these|that|those|where|find|describe|recognize|recognise)\b", text, re.I))


class Cancelled:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event is not None and event.is_set() for event in self.events)


class Robot:
    def __init__(self, camera, position, emit=lambda *args: None, frames=None, detector=None, start=True):
        self.camera = camera
        self.motion = MotionMapping.load(getattr(camera, "host", None))
        self.position = tuple(position)
        self.desk_position = tuple(position)
        self.emit = emit
        self.frames = frames or FrameStore()
        self.closed = threading.Event()
        self.motion_cancel = threading.Event()
        self.motion_lock = threading.RLock()
        self.moving = threading.Event()
        self.following = True
        self.gestures = True
        self.greetings = True
        self.cloud_vision = True
        self.settled_after = 0.0
        self.pause_tracking_until = 0.0
        self.last_step = 0.0
        self.last_gesture = -100.0
        self.last_arrival = -1000.0
        self.visible_since = None
        self.absent_since = time.monotonic()
        self.arrival_eligible = True
        self.target = TargetTracker()
        self.smoothed = None
        self.detector = detector
        self.status = "Waiting for camera frames"
        self.responsive_tracking = callable(getattr(camera, "position", None))
        self.position_controller = PositionController(self.motion)
        self._observation_lock = threading.Lock()
        self._tracking_observation = (None, None)
        self._position_at = time.monotonic()
        self._last_control_at = 0.0
        self._intent_lock = threading.Lock()
        self._motion_epoch = 0
        if start:
            threading.Thread(target=self._watch, daemon=True, name="jarvis-presence").start()
            if self.responsive_tracking:
                threading.Thread(target=self._tracking_loop, daemon=True, name="jarvis-tracking-motors").start()

    def context(self):
        fresh = self.frames.latest() is not None
        return {"camera_online": fresh, "cloud_vision_enabled": self.cloud_vision,
                "following_enabled": self.following, "small_gestures_enabled": self.gestures,
                "presence_detected": fresh and self.target.target is not None,
                "motor_position": [round(p, 4) for p in self.position] if fresh else None,
                "tracking_status": self.status,
                "tracking_mode": "camera-default" if self.responsive_tracking else "stepped"}

    def configure(self, *, following=None, gestures=None, greetings=None, cloud_vision=None):
        if following is not None:
            changed = self.following != bool(following)
            self.following = bool(following)
            if not self.following:
                self.stop()
            elif changed:
                self.motion_cancel.clear()
                self.pause_tracking_until = 0.0
        if gestures is not None:
            self.gestures = bool(gestures)
        if greetings is not None:
            self.greetings = bool(greetings)
        if cloud_vision is not None:
            self.cloud_vision = bool(cloud_vision)

    def disconnected(self):
        self.frames.clear()
        with self._observation_lock:
            self._tracking_observation = (None, None)
        self.target = TargetTracker()
        self.smoothed = None
        self.visible_since = None
        self.absent_since = time.monotonic()
        self.status = "Camera disconnected"

    def stop(self):
        self.following = False
        with self._intent_lock:
            self._motion_epoch += 1
            self.motion_cancel.set()
        self.pause_tracking_until = float("inf")
        self.emit("robot_following", False)
        def worker():
            try:
                with self.motion_lock:
                    self._halt_tracking(force=True)
            except CameraError:
                self.emit("robot_error", "Stop requested, but the camera did not respond.")
        threading.Thread(target=worker, daemon=True, name="jarvis-motor-stop").start()

    @contextmanager
    def manual(self, resume_after=15, expected_epoch=None):
        epoch = self.motion_epoch if expected_epoch is None else expected_epoch
        self.pause_tracking_until = time.monotonic() + 15
        with self.motion_lock:
            self._halt_tracking(refresh=True)
            with self._intent_lock:
                if epoch != self._motion_epoch or self.closed.is_set():
                    raise CameraError("Movement was cancelled.")
                self.motion_cancel.clear()
            try:
                yield
            finally:
                self.settled_after = time.monotonic() + 0.65
                self.smoothed = None
                self.pause_tracking_until = time.monotonic() + resume_after

    @property
    def motion_epoch(self):
        with self._intent_lock:
            return self._motion_epoch

    def _halt_tracking(self, refresh=False, force=False):
        active = self.position_controller.goal is not None
        try:
            if active or force:
                # The motion lock orders Stop after any command already in flight.
                self.camera.stop()
        finally:
            self.position_controller.reset()
            if active:
                self.settled_after = time.monotonic() + 0.25
        if refresh and (active or self.responsive_tracking):
            self.position = tuple(self.camera.position())
            self._position_at = time.monotonic()
            self.emit("robot_position", self.position)

    def tracking_tick(self, now=None):
        """Run motor control independently of the detector and cloud requests."""
        live_clock = now is None
        now = time.monotonic() if live_clock else now
        if not self.responsive_tracking or now - self._last_control_at < 0.09:
            return
        if not self.motion_lock.acquire(blocking=False):
            return
        try:
            self._last_control_at = now
            if (not self.following or self.motion_cancel.is_set() or self.closed.is_set()
                    or now < self.pause_tracking_until or self.frames.latest(max_age=0.5) is None):
                self._halt_tracking()
                return
            if now - self._position_at >= (0.09 if self.position_controller.goal is not None else 0.3):
                self.position = tuple(self.camera.position())
                self._position_at = now
                self.emit("robot_position", self.position)
                if live_clock:
                    now = time.monotonic()
            with self._observation_lock:
                point, captured_at = self._tracking_observation
            command = self.position_controller.plan(point, captured_at, now, self.position)
            if command is None:
                return
            if command[0] == "stop":
                self._halt_tracking()
                return
            if (self.motion_cancel.is_set() or self.closed.is_set() or not self.following
                    or now < self.pause_tracking_until):
                self._halt_tracking()
                return
            # Omit Speed, as in the earlier quiet movement fix. The target is
            # finite even if the controller stops receiving camera frames.
            self.camera.move_to(*command[1])
            if self.motion_cancel.is_set() or self.closed.is_set():
                self._halt_tracking(force=True)
        finally:
            self.motion_lock.release()

    def _tracking_loop(self):
        while not self.closed.wait(0.04):
            try:
                self.tracking_tick()
            except (CameraError, TimeoutError) as error:
                self.stop()
                self.status = "Following paused after a camera error"
                self.emit("robot_error", str(error))
            except Exception:
                self.stop()
                self.status = "Following paused after a controller error"
                self.emit("robot_error", "Following paused after a controller error. Stop was requested.")

    def _move(self, target, cancelled=None):
        stop = Cancelled(self.closed, self.motion_cancel, cancelled)
        if stop.is_set():
            raise CameraError("Movement was cancelled.")
        requested = target
        target = tuple(max(-1.0, min(1.0, p)) for p in target)
        before = self.position
        limited = tuple(requested) != target
        if max(abs(before[i] - target[i]) for i in (0, 1)) <= 0.001:
            return {"ok": True, "moved": False, "limited": limited, "position": list(before)}
        self.moving.set()
        try:
            self.camera.move_to(*target)
            actual = self.camera.wait_position(target, timeout=8, cancelled=stop)
            if stop.is_set():
                raise CameraError("Movement was cancelled.")
            self.position = tuple(actual)
            self._position_at = time.monotonic()
            if max(abs(self.position[i] - target[i]) for i in (0, 1)) > 0.005:
                raise CameraError("The camera did not confirm the requested angle.")
            self.emit("robot_position", self.position)
            return {"ok": True, "moved": True, "limited": limited, "position": list(self.position),
                    "pan_degrees": round((self.position[0] - before[0]) * 170 * self.motion.pan_right, 2),
                    "tilt_degrees": round((self.position[1] - before[1]) * 35 * self.motion.tilt_up, 2)}
        finally:
            self.moving.clear()
            self.settled_after = time.monotonic() + 0.65
            self.smoothed = None
            self.camera.stop()

    def snapshot(self, cancelled=None):
        if not self.cloud_vision:
            raise VisionError("Cloud camera vision is switched off.")
        with self.motion_lock:
            self.pause_tracking_until = max(self.pause_tracking_until, time.monotonic() + 0.6)
            self._halt_tracking(refresh=True)
            return self.frames.snapshot(after=self.settled_after, cancelled=Cancelled(self.closed, cancelled))

    @staticmethod
    def _number(arguments, name, low, high):
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
        return float(value)

    def execute(self, name, arguments, cancelled):
        epoch = self.motion_epoch
        allowed = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS}
        if name not in allowed or not isinstance(arguments, dict):
            return {"ok": False, "error": "That camera action is not available."}
        schema = allowed[name]
        if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
            return {"ok": False, "error": "The camera action has invalid arguments."}
        if cancelled.is_set() or self.closed.is_set():
            return {"ok": False, "error": "Camera action was cancelled."}
        try:
            if name == "stop_motion":
                self.stop()
                return {"ok": True, "stop_requested": True, "following_enabled": False}
            if name == "capture_frame":
                return {"ok": True, "_image": self.snapshot(cancelled)}
            if self.frames.latest() is None:
                raise CameraError("The camera is disconnected. No movement was performed.")
            if name == "follow_person":
                if not isinstance(arguments["enabled"], bool):
                    raise ValueError("enabled must be true or false.")
                self.following = arguments["enabled"]
                if self.following:
                    self.motion_cancel.clear()
                    self.pause_tracking_until = 0.0
                else:
                    self.stop()
                self.emit("robot_following", self.following)
                return {"ok": True, "following_enabled": self.following}
            with self.motion_lock:
                self.pause_tracking_until = time.monotonic() + 15
                self._halt_tracking(refresh=True)
                with self._intent_lock:
                    if epoch != self._motion_epoch or cancelled.is_set() or self.closed.is_set():
                        raise CameraError("Movement was cancelled.")
                    self.motion_cancel.clear()
                if name == "look":
                    direction = arguments["direction"]
                    if not isinstance(direction, str) or direction not in {"left", "right", "up", "down", "desk"}:
                        raise ValueError("Choose left, right, up, down, or desk.")
                    degrees = self._number({"degrees": 15, **arguments}, "degrees", 1, 45)
                    pan, tilt = self.position
                    target = self.desk_position if direction == "desk" else (
                        pan + ({"left": -1, "right": 1}.get(direction, 0)) * degrees / 170 * self.motion.pan_right,
                        tilt + ({"up": 1, "down": -1}.get(direction, 0)) * degrees / 35 * self.motion.tilt_up)
                    return self._move(target, cancelled)
                if name == "look_at":
                    x = self._number(arguments, "x", 0, 1)
                    y = self._number(arguments, "y", 0, 1)
                    pan, tilt = self.position
                    return self._move((pan + max(-12, min(12, (x - 0.5) * 40)) / 170 * self.motion.pan_right,
                                       tilt + max(-8, min(8, (0.5 - y) * 30)) / 35 * self.motion.tilt_up), cancelled)
                if name == "gesture":
                    return self._gesture(arguments["kind"], cancelled)
        except (CameraError, VisionError, ValueError) as error:
            return {"ok": False, "error": str(error)}

    def _gesture(self, kind, cancelled):
        if not isinstance(kind, str) or kind not in {"nod", "curious"}:
            raise ValueError("Choose nod or curious.")
        if not self.gestures:
            return {"ok": False, "error": "Small gestures are switched off."}
        if time.monotonic() - self.last_gesture < 12:
            return {"ok": False, "error": "The camera is resting between gestures."}
        self.last_gesture = time.monotonic()
        start = self.position
        sign = -1 if start[1] > 0.8 else 1
        offset = (0, sign * 2.5 / 35) if kind == "nod" else (2 / 170, sign * 1.5 / 35)
        self._move((start[0] + offset[0], start[1] + offset[1]), cancelled)
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline and not Cancelled(self.closed, self.motion_cancel, cancelled).is_set():
            time.sleep(0.01)
        if Cancelled(self.closed, self.motion_cancel, cancelled).is_set():
            raise CameraError("Gesture was stopped.")
        result = self._move(start, cancelled)
        if self.following:
            self.pause_tracking_until = time.monotonic() + 0.45
        return result

    def observe(self, detections, now, observed_at=None):
        if self.moving.is_set() or (observed_at is not None and observed_at < self.settled_after):
            self.smoothed = None
            return
        target = self.target.update(detections, now)
        if detections:
            if self.visible_since is None:
                self.visible_since = now
                self.arrival_eligible = now - self.absent_since >= 15 or self.last_arrival < 0
            if self.greetings and self.arrival_eligible and now - self.visible_since >= 2 and now - self.last_arrival >= 300:
                self.last_arrival = now
                self.arrival_eligible = False
                self.emit("person_arrived", None)
        elif self.visible_since is not None:
            self.visible_since = None
            self.absent_since = now
        if target is None:
            self.smoothed = None
            self.status = "No person in view" if not detections else "Waiting for a stable target"
            return
        self.status = "Following a person" if self.following else "Person in view; following paused"
        if self.responsive_tracking and self.following:
            errors = (target.center[0]-0.5, 0.45-target.center[1])
            directions = (self.motion.pan_right, self.motion.tilt_up)
            if any(abs(self.position[i]) > 0.94 and abs(errors[i]) > 0.08
                   and self.position[i] * errors[i] * directions[i] > 0 for i in (0, 1)):
                self.status = "At turn limit; following where possible"
        self.emit("robot_target", target)
        point = target.center
        if self.responsive_tracking:
            with self._observation_lock:
                self._tracking_observation = (point, now if observed_at is None else observed_at)
            return
        self.smoothed = point if self.smoothed is None else tuple(0.65 * a + 0.35 * b for a, b in zip(self.smoothed, point))
        self.smoothed = tuple(current if (smooth - center) * (current - center) < 0 else smooth
                              for smooth, current, center in zip(self.smoothed, point, (0.5, 0.45)))
        if (not self.following or now < self.pause_tracking_until or now - self.last_step < 0.8
                or self.closed.is_set() or not self.motion_lock.acquire(blocking=False)):
            return
        try:
            if not self.following or self.motion_cancel.is_set():
                return
            x, y = self.smoothed
            dx = 0 if abs(x - 0.5) < 0.1 else max(-4, min(4, (x - 0.5) * 20))
            dy = 0 if abs(y - 0.45) < 0.12 else max(-2, min(2, (0.45 - y) * 12))
            if not dx and not dy:
                return
            self.last_step = now
            self.motion_cancel.clear()
            self._move((self.position[0] + dx / 170 * self.motion.pan_right,
                        self.position[1] + dy / 35 * self.motion.tilt_up))
        finally:
            self.motion_lock.release()

    def _watch(self):
        try:
            self.detector = self.detector or PersonDetector()
        except Exception:
            self.status = "Local person detection could not load"
            self.emit("robot_error", "Person detection could not load. Run setup_voice.py.")
            return
        last_serial = None
        while not self.closed.wait(0.08 if self.responsive_tracking else 0.25):
            frame = self.frames.latest(max_age=1.0)
            if frame is None or frame[1] == last_serial:
                if frame is None:
                    self.target = TargetTracker()
                    self.status = "Waiting for fresh camera frames"
                continue
            if self.moving.is_set() or frame[0] < self.settled_after:
                continue
            last_serial = frame[1]
            try:
                self.observe(self.detector.detect(frame[2]), time.monotonic(), observed_at=frame[0])
            except CameraError as error:
                self.following = False
                self.emit("robot_following", False)
                self.emit("robot_error", str(error))
            except Exception:
                self.emit("robot_error", "Person tracking paused after a detector error.")
                return

    def close(self):
        self.closed.set()
        self.stop()
        self.frames.clear()
