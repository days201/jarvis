"""Exercise image freshness, bounded actions, target continuity, and stopping."""

import base64
import io
import threading
import time
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from PIL import Image

from robot import Robot
from motion_mapping import MotionMapping, image_shift
from vision import Detection, FrameStore, TargetTracker, VisionError


class SimulatedCamera:
    def __init__(self):
        self.pose = (0.0, 0.0)
        self.moves = []
        self.stops = 0

    def move_to(self, x, y):
        self.pose = (x, y)
        self.moves.append(self.pose)

    def wait_position(self, target, **kwargs):
        return self.pose

    def stop(self):
        self.stops += 1


class RobotTests(unittest.TestCase):
    def setUp(self):
        self.camera = SimulatedCamera()
        self.robot = Robot(self.camera, (0, 0), start=False)
        self.robot.frames.push(Image.new("RGB", (1280, 720), "blue"))
        self.cancelled = threading.Event()

    def test_invalid_or_unavailable_actions_never_move(self):
        for name, arguments in [("look", {"direction": "right", "degrees": float("nan")}),
                                 ("look", {"direction": "right", "degrees": 360}),
                                 ("look_at", {"x": 2, "y": 0.5}),
                                 ("look", {"direction": ["right"]}),
                                 ("gesture", {"kind": ["nod"]}),
                                 ("follow_person", {"enabled": "yes"}),
                                 ("look", {"direction": "right", "url": "http://example.com"}),
                                 ("run_command", {"command": "anything"})]:
            with self.subTest(name=name, arguments=arguments):
                self.assertFalse(self.robot.execute(name, arguments, self.cancelled)["ok"])
        self.assertEqual(self.camera.moves, [])

    def test_look_clamps_physical_limits_and_requires_confirmation(self):
        self.robot.position = (0.99, 0.99)
        result = self.robot.execute("look", {"direction": "right", "degrees": 45}, self.cancelled)
        self.assertTrue(result["ok"])
        self.assertEqual(self.camera.moves[-1], (1.0, 0.99))
        self.assertEqual(self.camera.stops, 1)

    def test_cancelled_action_never_reaches_camera(self):
        self.cancelled.set()
        self.assertFalse(self.robot.execute("look", {"direction": "left"}, self.cancelled)["ok"])
        self.assertEqual(self.camera.moves, [])

    def test_tracking_does_not_repeat_commands_at_a_motor_limit(self):
        self.robot.position = (1, 0)
        self.camera.pose = (1, 0)
        now = time.monotonic()
        for step in range(3):
            self.robot.observe([Detection(0.85, 0.35, 0.1, 0.2)], now + step)
        self.assertEqual(self.camera.moves, [])

    def test_disconnected_camera_cannot_move_or_supply_an_old_photo(self):
        self.robot.disconnected()
        self.assertFalse(self.robot.execute("look", {"direction": "left"}, self.cancelled)["ok"])
        self.assertEqual(self.camera.moves, [])
        with self.assertRaises(VisionError):
            self.robot.frames.snapshot(timeout=0)

    def test_stopping_disables_tracking_until_explicitly_resumed(self):
        self.robot.stop()
        self.robot.observe([Detection(0.8, 0.3, 0.1, 0.2)], time.monotonic())
        self.assertFalse(self.robot.following)
        self.assertEqual(self.camera.moves, [])
        self.assertTrue(self.robot.execute("follow_person", {"enabled": True}, self.cancelled)["ok"])
        self.robot.observe([Detection(0.8, 0.3, 0.1, 0.2)], time.monotonic())
        self.assertEqual(len(self.camera.moves), 1)
        self.assertLessEqual(abs(self.camera.moves[0][0]), 4 / 170)

    def test_centered_person_and_detector_jitter_do_not_move_motors(self):
        now = time.monotonic()
        for offset in (0, 0.01, -0.02, 0.02):
            self.robot.observe([Detection(0.45 + offset, 0.35, 0.1, 0.2)], now)
            now += 1
        self.assertEqual(self.camera.moves, [])

    def test_tracking_is_rate_limited(self):
        now = time.monotonic()
        box = Detection(0.8, 0.3, 0.1, 0.2)
        self.robot.observe([box], now)
        self.robot.observe([box], now + 0.2)
        self.robot.observe([box], now + 0.5)
        self.assertEqual(len(self.camera.moves), 1)

    def test_calibrated_horizontal_tracking_reduces_error_on_both_sides(self):
        # This camera moves scene features right when raw ONVIF pan increases.
        # Simulate that response and verify the closed loop reduces image error.
        self.robot.motion = MotionMapping(pan_right=-1, tilt_up=1)
        now = time.monotonic()
        for initial_x in (0.2, 0.8):
            with self.subTest(initial_x=initial_x):
                self.robot.position = (0, 0)
                self.robot.target = TargetTracker()
                self.robot.smoothed = None
                self.robot.last_step = 0
                self.camera.pose = (0, 0)
                point = initial_x
                previous_error = abs(point - 0.5)
                for step in range(5):
                    before = self.robot.position
                    self.robot.observe([Detection(point - 0.05, 0.35, 0.1, 0.2)], now + step)
                    point += (self.robot.position[0] - before[0]) * 3
                    self.assertLessEqual(abs(point - 0.5), previous_error)
                    previous_error = abs(point - 0.5)
                self.assertLess(abs(point - 0.5), abs(initial_x - 0.5))

    def test_calibrated_look_and_centering_use_the_same_direction(self):
        self.robot.motion = MotionMapping(pan_right=-1, tilt_up=1)
        result = self.robot.execute("look", {"direction": "right", "degrees": 10}, self.cancelled)
        self.assertTrue(result["ok"])
        self.assertLess(self.camera.moves[-1][0], 0)
        self.assertEqual(result["pan_degrees"], 10)
        start = self.robot.position
        self.robot.execute("look_at", {"x": 0.8, "y": 0.5}, self.cancelled)
        self.assertLess(self.camera.moves[-1][0], start[0])

    def test_pre_move_detection_cannot_trigger_another_correction(self):
        self.robot.settled_after = 100
        self.robot.observe([Detection(0.8, 0.35, 0.1, 0.2)], 101, observed_at=99.9)
        self.assertEqual(self.camera.moves, [])

    def test_crossing_the_center_does_not_keep_chasing_the_old_direction(self):
        self.robot.motion = MotionMapping(pan_right=-1, tilt_up=1)
        self.robot.smoothed = (0.8, 0.45)
        self.robot.target.target = Detection(0.3, 0.35, 0.1, 0.2)
        self.robot.target.last_seen = time.monotonic()
        self.robot.observe([Detection(0.3, 0.35, 0.1, 0.2)], time.monotonic())
        self.assertGreater(self.camera.moves[-1][0], 0)

    def test_scene_shift_measurement_preserves_direction(self):
        import numpy as np
        import cv2
        rng = np.random.default_rng(12)
        pixels = rng.integers(0, 256, size=(360, 640), dtype=np.uint8)
        shifted = cv2.warpAffine(pixels, np.float32([[1, 0, 24], [0, 1, -12]]), (640, 360))
        shift = image_shift(Image.fromarray(pixels), Image.fromarray(shifted))
        self.assertAlmostEqual(shift["x"], 24 / 640, delta=0.003)
        self.assertAlmostEqual(shift["y"], -12 / 360, delta=0.003)

    def test_mapping_loads_only_valid_calibration_for_this_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text(json.dumps({"host": "192.168.2.88", "pan_right": -1, "tilt_up": 1}), encoding="utf-8")
            self.assertEqual(MotionMapping.load("192.168.2.88", path).pan_right, -1)
            self.assertEqual(MotionMapping.load("192.168.2.89", path), MotionMapping())
            path.write_text(json.dumps({"host": "192.168.2.88", "pan_right": 12, "tilt_up": 1}), encoding="utf-8")
            self.assertEqual(MotionMapping.load("192.168.2.88", path), MotionMapping())

    def test_delayed_ui_position_cannot_overwrite_newer_motor_position(self):
        from desktop import Desktop
        self.robot.position = (0.4, 0.3)
        display = SimpleNamespace(robot=self.robot, desk_position=None)
        Desktop.set_position(display, (0.1, 0.2))
        self.assertEqual(self.robot.position, (0.4, 0.3))

    def test_gesture_returns_to_start_and_has_a_rest_period(self):
        result = self.robot.execute("gesture", {"kind": "nod"}, self.cancelled)
        self.assertTrue(result["ok"])
        self.assertEqual(self.camera.moves[-1], (0.0, 0.0))
        self.assertFalse(self.robot.execute("gesture", {"kind": "curious"}, self.cancelled)["ok"])
        self.assertEqual(len(self.camera.moves), 2)

    def test_tracking_keeps_current_person_when_someone_larger_appears(self):
        tracker = TargetTracker()
        first = Detection(0.1, 0.2, 0.12, 0.3)
        other = Detection(0.7, 0.1, 0.25, 0.7)
        self.assertEqual(tracker.update([first], 10), first)
        self.assertEqual(tracker.update([first, other], 10.2), first)
        self.assertIsNone(tracker.update([other], 10.4))

    def test_arrival_greetings_require_stable_presence_and_a_long_cooldown(self):
        events = []
        self.robot.emit = lambda kind, value: events.append(kind)
        self.robot.following = False
        now = time.monotonic()
        face = [Detection(0.4, 0.3, 0.2, 0.2)]
        self.robot.observe(face, now)
        self.robot.observe(face, now + 1)
        self.assertNotIn("person_arrived", events)
        self.robot.observe(face, now + 2.1)
        self.robot.observe([], now + 3)
        self.robot.observe(face, now + 25)
        self.robot.observe(face, now + 28)
        self.assertEqual(events.count("person_arrived"), 1)

    def test_snapshots_are_bounded_jpeg_data_and_privacy_switch_blocks_them(self):
        uri = self.robot.snapshot(self.cancelled)
        image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(image.format, "JPEG")
        self.assertLessEqual(image.width, 960)
        self.robot.cloud_vision = False
        self.assertFalse(self.robot.execute("capture_frame", {}, self.cancelled)["ok"])

    def test_snapshot_rejects_frames_older_than_the_move(self):
        with self.assertRaises(VisionError):
            self.robot.frames.snapshot(after=time.monotonic() + 1, timeout=0)

    def test_frame_store_does_not_reuse_stale_images(self):
        current = [10.0]
        frames = FrameStore(clock=lambda: current[0])
        frames.push(Image.new("RGB", (20, 20)))
        current[0] = 13.0
        self.assertIsNone(frames.latest())
        with self.assertRaises(VisionError):
            frames.snapshot(timeout=0)


if __name__ == "__main__":
    unittest.main()
