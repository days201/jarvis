"""Exercise default-speed tracking, stale detections, and command ownership."""

import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from robot.camera import CameraError
from robot.motion_mapping import MotionMapping
from robot.robot import Robot
from robot.vision import Detection


class TrackingCamera:
    host = '192.168.2.88'
    motion_defaults = {'pan_speed': 1.0, 'tilt_speed': 0.89}

    def __init__(self):
        self.pose = (0.0, 0.0)
        self.velocities = []
        self.moves = []
        self.stops = 0
        self.speed_overrides = []

    def continuous_move(self, x, y):
        self.velocities.append((x, y))

    def position(self):
        return self.pose

    def stop(self):
        self.stops += 1

    def move_to(self, x, y, speed=None):
        self.pose = (x, y)
        self.moves.append(self.pose)
        self.speed_overrides.append(speed)

    def wait_position(self, target, **kwargs):
        return self.pose


class TrackingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.camera=TrackingCamera()
        with patch('robot.robot.MotionMapping.load',return_value=MotionMapping(-1,1)):
            self.robot=Robot(self.camera,(0,0),start=False)
        self.robot.frames.push(Image.new('RGB',(100,100)))
        self.now=time.monotonic()
        self.face=Detection(.75,.35,.1,.2)

    def observe_and_tick(self, now):
        self.robot.observe([self.face],now,observed_at=now)
        self.robot.tracking_tick(now)

    def test_detection_keeps_running_without_stop_or_small_step_commands(self):
        self.observe_and_tick(self.now)
        count=len(self.camera.moves)
        self.robot.observe([self.face],self.now+.1,observed_at=self.now+.1)
        self.assertEqual(len(self.camera.moves),count)
        self.robot.tracking_tick(self.now+.1)
        self.assertEqual(len(self.camera.moves),count)
        self.observe_and_tick(self.now+.3)
        self.assertGreater(len(self.camera.moves),count)
        self.assertEqual(self.camera.velocities,[])
        self.assertEqual(self.camera.stops,0)
        self.assertGreater(abs(self.camera.moves[0][0])*170,4)

    def test_tracking_never_sends_a_speed_override_or_continuous_velocity(self):
        for i in range(8):
            now=self.now+i*.12
            self.robot.observe([Detection(.75,.1,.1,.1)],now,observed_at=now)
            self.robot.tracking_tick(now)
        self.assertTrue(self.camera.moves)
        self.assertTrue(all(speed is None for speed in self.camera.speed_overrides))
        self.assertEqual(self.camera.velocities,[])

    def test_lost_detections_expire_even_while_video_remains_fresh(self):
        self.observe_and_tick(self.now)
        self.robot.tracking_tick(self.now+.4)
        self.assertEqual(self.camera.stops,1)
        self.assertIsNone(self.robot.position_controller.goal)

    def test_manual_action_stops_tracking_and_uses_current_position(self):
        self.observe_and_tick(self.now)
        self.camera.pose=(.3,.2)
        with self.robot.manual():
            self.assertEqual(self.robot.position,(.3,.2))
            count=len(self.camera.moves)
            self.robot.tracking_tick(self.now+.2)
            self.assertEqual(len(self.camera.moves),count)
        self.assertEqual(self.camera.stops,1)

    def test_stop_prevents_a_later_detection_from_restarting_motion(self):
        self.observe_and_tick(self.now)
        self.robot.stop()
        with self.robot.motion_lock:
            pass
        count=len(self.camera.moves)
        self.observe_and_tick(self.now+.2)
        self.assertEqual(len(self.camera.moves),count)
        self.assertFalse(self.robot.following)

    def test_device_without_position_feedback_keeps_existing_fallback(self):
        from .support import SimulatedCamera
        camera=SimulatedCamera()
        with patch('robot.robot.MotionMapping.load',return_value=MotionMapping()):
            robot=Robot(camera,(0,0),start=False)
        robot.frames.push(Image.new('RGB',(100,100)))
        self.assertFalse(robot.responsive_tracking)
        robot.observe([self.face],self.now)
        self.assertTrue(camera.moves)

    def test_stop_cancels_a_manual_job_that_was_queued_before_it(self):
        epoch=self.robot.motion_epoch
        self.robot.stop()
        with self.assertRaisesRegex(CameraError,'cancelled'):
            with self.robot.manual(expected_epoch=epoch):
                self.camera.move_to(.1,0)
        self.assertEqual(self.camera.moves,[])


if __name__=='__main__':
    unittest.main()
