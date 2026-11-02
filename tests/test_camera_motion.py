"""Check the movement requests sent to the camera without moving hardware."""

import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from robot.camera import Camera, CameraError, NS, PTZ


class MovementRequestTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera("192.168.2.88", "test-user", "test-password")
        self.camera.profile = "profile_1"

    def capture(self, operation, fill):
        command = ET.Element(f"{{{PTZ}}}{operation}")
        fill(command)
        return command

    def test_default_move_does_not_override_camera_speed(self):
        with patch.object(self.camera, "_request") as request:
            self.camera.move_to(0.1, -0.2)
        args = request.call_args.args
        command = self.capture(args[2], args[3])
        self.assertIsNone(command.find("p:Speed", NS))
        self.assertEqual(command.find("p:Position/tt:PanTilt", NS).attrib, {"x": "0.1", "y": "-0.2"})

    def test_explicit_supported_speed_is_preserved(self):
        with patch.object(self.camera, "_request") as request:
            self.camera.move_to(0.1, -0.2, speed=1.0)
        args = request.call_args.args
        command = self.capture(args[2], args[3])
        self.assertEqual(command.find("p:Speed/tt:PanTilt", NS).attrib, {"x": "1.0", "y": "1.0"})

    def test_invalid_movements_never_reach_the_camera(self):
        for x, y, speed in [(1.01, 0, None), (0, -1.01, None), (float("nan"), 0, None), (0, 0, 0), (0, 0, 1.01), (0, 0, float("inf"))]:
            with self.subTest(x=x, y=y, speed=speed), patch.object(self.camera, "_request") as request:
                with self.assertRaises(CameraError):
                    self.camera.move_to(x, y, speed)
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
