"""Exercise default-speed tracking, protocol expiry, and command ownership."""

import threading
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from camera import Camera, CameraError, NS, PTZ, SCHEMA
from motion_mapping import MotionMapping
from robot import Robot
from vision import Detection


class ContinuousCamera:
    supports_continuous = True
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
        self.camera=ContinuousCamera()
        with patch('robot.MotionMapping.load',return_value=MotionMapping(-1,1,1,1)):
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
        from test_robot import SimulatedCamera
        camera=SimulatedCamera()
        with patch('robot.MotionMapping.load',return_value=MotionMapping()):
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


class ContinuousRequestTests(unittest.TestCase):
    def test_capability_discovery_requires_velocity_space_and_short_expiry(self):
        camera=Camera('192.168.2.88','test','test')
        config=ET.Element('configuration',token='ptz-config')
        xml='''<Options xmlns:tt="http://www.onvif.org/ver10/schema">
          <tt:ContinuousPanTiltVelocitySpace>
            <tt:URI>http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace</tt:URI>
            <tt:XRange><tt:Min>-1</tt:Min><tt:Max>1</tt:Max></tt:XRange>
            <tt:YRange><tt:Min>-1</tt:Min><tt:Max>1</tt:Max></tt:YRange>
          </tt:ContinuousPanTiltVelocitySpace>
          <tt:PTZTimeout><tt:Min>PT1S</tt:Min><tt:Max>PT600S</tt:Max></tt:PTZTimeout>
        </Options>'''
        with patch.object(camera,'_request',return_value=ET.fromstring(xml)):
            camera._read_continuous_options(config)
        self.assertTrue(camera.supports_continuous)
        with patch.object(camera,'_request',return_value=ET.fromstring(xml.replace('PT1S','PT5S'))):
            camera._read_continuous_options(config)
        self.assertFalse(camera.supports_continuous)

    def test_missing_or_unreadable_capabilities_keep_absolute_fallback(self):
        camera=Camera('192.168.2.88','test','test')
        with patch.object(camera,'_request',side_effect=CameraError('Unavailable')):
            camera._read_continuous_options(ET.Element('configuration',token='config'))
        self.assertFalse(camera.supports_continuous)

    def test_velocity_request_has_signed_axes_and_device_expiry(self):
        camera=Camera('192.168.2.88','test','test')
        camera.profile='profile_1';camera.supports_continuous=True
        with patch.object(camera,'_request') as request:
            camera.continuous_move(-.6,.4)
        args=request.call_args.args
        command=ET.Element(f'{{{PTZ}}}{args[2]}');args[3](command)
        self.assertEqual(command.find('p:Velocity/tt:PanTilt',NS).get('x'),'-0.600')
        self.assertEqual(command.find('p:Velocity/tt:PanTilt',NS).get('y'),'0.400')
        self.assertEqual(command.findtext('p:Timeout',namespaces=NS),'PT1S')
        self.assertEqual(request.call_args.kwargs['request_timeout'],(.5,1.25))

    def test_invalid_or_unadvertised_velocity_never_reaches_hardware(self):
        camera=Camera('192.168.2.88','test','test')
        with patch.object(camera,'_request') as request:
            with self.assertRaises(CameraError): camera.continuous_move(.5,0)
            camera.supports_continuous=True
            for x,y,timeout in [(2,0,1),(0,float('nan'),1),(True,0,1),(.5,0,20),(.5,0,.2)]:
                with self.assertRaises(CameraError): camera.continuous_move(x,y,timeout)
            request.assert_not_called()


if __name__=='__main__':
    unittest.main()
