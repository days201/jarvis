"""Check quiet tracking goals, completion, image freshness, and limits."""

import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from robot.camera import Camera, NS, PTZ
from robot.motion_mapping import MotionMapping
from robot.robot import Robot
from robot.tracking_motion import PositionController
from robot.vision import Detection


class PositionTrackingTests(unittest.TestCase):
    def setUp(self):
        self.control = PositionController(MotionMapping(-1,1))

    def test_large_correction_is_a_finite_destination_in_the_calibrated_direction(self):
        kind,goal=self.control.plan((.8,.2),10,10,(0,0))
        self.assertEqual(kind,'move')
        self.assertAlmostEqual(goal[0],-18/170)
        self.assertAlmostEqual(goal[1],8.75/35)

    def test_active_move_does_not_reaccumulate_correction_from_each_frame(self):
        first=self.control.plan((.8,.2),10,10,(0,0))
        for i in range(1,8):
            self.assertIsNone(self.control.plan((.8,.2),10+i*.1,10+i*.1,(0,0)))
        self.assertEqual(self.control.goal,first[1])

    def test_completion_uses_a_fresh_frame_without_a_fixed_long_settle(self):
        goal=self.control.plan((.8,.45),10,10,(0,0))[1]
        self.assertIsNone(self.control.plan((.6,.45),10.2,10.2,goal))
        self.assertIsNone(self.control.plan((.6,.45),10.21,10.4,goal))
        self.assertEqual(self.control.plan((.6,.45),10.4,10.4,goal)[0],'move')

    def test_missing_or_stale_target_stops_an_active_move(self):
        self.control.plan((.8,.45),10,10,(0,0))
        self.assertEqual(self.control.plan((.8,.45),10,10.4,(0,0)),('stop',None))
        self.assertEqual(self.control.plan(None,None,10.4,(0,0)),('stop',None))

    def test_finite_goal_has_a_completion_deadline(self):
        self.control.plan((.8,.45),10,10,(0,0))
        with self.assertRaises(TimeoutError):
            self.control.plan((.8,.45),13.1,13.1,(0,0))

    def test_centered_jitter_and_repeated_frames_do_not_move(self):
        for i,point in enumerate([(.5,.45),(.53,.48),(.47,.43)]):
            self.assertIsNone(self.control.plan(point,10+i*.1,10+i*.1,(0,0)))
        self.assertIsNone(self.control.plan((.8,.45),10.2,10.3,(0,0)))

    def test_limits_never_create_a_move_opposite_to_the_request(self):
        self.assertIsNone(self.control.plan((.1,.45),10,10,(.99,0)))
        self.assertEqual(self.control.plan((.9,.45),10.1,10.1,(.99,0))[0],'move')
        self.assertLess(self.control.goal[0],.99)

    def test_target_loss_then_reset_cannot_resume_the_old_move(self):
        self.control.plan((.8,.45),10,10,(0,0))
        self.control.reset()
        self.assertIsNone(self.control.goal)
        self.assertIsNone(self.control.plan(None,None,11,(0,0)))

    def test_real_tracking_request_omits_the_optional_speed_field(self):
        camera=Camera('192.168.2.88','test','test');camera.profile='profile_1'
        robot=Robot(camera,(0,0),start=False)
        robot.frames.push(Image.new('RGB',(20,20)))
        now=time.monotonic()
        robot.observe([Detection(.75,.1,.1,.1)],now,observed_at=now)
        with patch.object(camera,'_request') as request:
            robot.tracking_tick(now)
        args=request.call_args.args
        self.assertEqual(args[2],'AbsoluteMove')
        command=ET.Element(f'{{{PTZ}}}{args[2]}');args[3](command)
        self.assertIsNone(command.find('p:Speed',NS))


if __name__=='__main__':
    unittest.main()
