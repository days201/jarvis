"""Measure both camera directions locally with small probes, then restore pose."""

import json
import math
import sys
import threading
import time

import av

from robot.camera import Camera, CameraError
from robot.motion_mapping import CALIBRATION_PATH, image_shift
from robot.vision import FrameStore


def main():
    login = json.loads(sys.stdin.read())
    camera = Camera(login["host"], login["username"], login["password"])
    del login
    camera.connect()
    camera.stop()
    start = camera.position()
    frames = FrameStore()
    done = threading.Event()
    def feed():
        with camera.open_video(low_quality=True) as stream:
            for packet in stream.demux():
                if done.is_set():
                    return
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        frames.push(frame.to_image())
    threading.Thread(target=feed, daemon=True).start()
    def fresh(after):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            frame = frames.latest()
            if frame and frame[0] >= after:
                return frame[2]
            time.sleep(0.04)
        raise ValueError("No fresh frame arrived for direction measurement.")
    def move(target):
        camera.move_to(*target)
        camera.wait_position(target, timeout=8)
        camera.stop()
        return camera.position()
    report = {"passed": False, "start": start, "probes": {}}
    try:
        for axis, key, scale in [(0, "pan", 170), (1, "tilt", 35)]:
            before = fresh(time.monotonic() + 0.65)
            delta = (6 if axis == 0 else 3) / scale * (-1 if start[axis] > 0.9 else 1)
            target = list(start)
            target[axis] = max(-1, min(1, target[axis] + delta))
            actual = move(target)
            after = fresh(time.monotonic() + 0.65)
            shift = image_shift(before, after)
            report["probes"][key] = {"motor_delta": actual[axis] - start[axis], "scene_shift": shift}
            move(start)
        pan = report["probes"]["pan"]
        tilt = report["probes"]["tilt"]
        if abs(pan["scene_shift"]["x"]) < 0.005 or abs(tilt["scene_shift"]["y"]) < 0.005:
            raise ValueError("The scene displacement was too small to calibrate direction.")
        mapping = {"host": camera.host,
                   "pan_right": -int(math.copysign(1, pan["scene_shift"]["x"] / pan["motor_delta"])),
                   "tilt_up": int(math.copysign(1, tilt["scene_shift"]["y"] / tilt["motor_delta"]))}
        CALIBRATION_PATH.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        report.update({"passed": True, "mapping": mapping})
    except (CameraError, ValueError) as error:
        report["error"] = str(error)
    finally:
        try:
            actual = move(start)
            report["restored"] = max(abs(actual[i] - start[i]) for i in (0, 1)) < 0.002
        except CameraError as error:
            report["restored"] = False
            report["restore_error"] = str(error)
        done.set()
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] and report["restored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
