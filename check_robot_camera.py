"""Verify cloud vision and bounded movement on the camera, then restore its pose."""

import json
import argparse
import sys
import threading
import time

import av

from assistant import DeepSeekClient
from camera import Camera, CameraError
from robot import Robot
from vision import PersonDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-only", action="store_true")
    args = parser.parse_args()
    login = json.loads(sys.stdin.read())
    camera = Camera(login["host"], login["username"], login["password"])
    client = DeepSeekClient(login["api_key"])
    del login
    camera.connect()
    start = camera.position()
    robot = Robot(camera, start, start=False)
    robot.following = False
    robot.greetings = False
    done = threading.Event()
    ready = threading.Event()
    counts = {"video": 0, "audio": 0}
    errors = []
    def feed():
        try:
            with camera.open_video(low_quality=True) as stream:
                for packet in stream.demux():
                    if done.is_set():
                        break
                    for frame in packet.decode():
                        if isinstance(frame, av.VideoFrame):
                            robot.frames.push(frame.to_image())
                            counts["video"] += 1
                            ready.set()
                        elif isinstance(frame, av.AudioFrame):
                            counts["audio"] += 1
        except Exception:
            errors.append("Camera stream ended during the test.")
    threading.Thread(target=feed, daemon=True).start()
    report = {"passed": False, "physical_camera": True}
    try:
        if not ready.wait(12):
            raise CameraError("No live frame arrived during the test.")
        answer = "" if args.tracking_only else "".join(client.reply(
            "Use look to turn right by exactly 5 degrees once. Then briefly describe what you actually see in the camera view. Do not make any other movements.",
            threading.Event(), robot=robot, new_conversation=True))
        after_look = camera.position()
        direction_ok = args.tracking_only or (after_look[0] - start[0]) * robot.motion.pan_right > 0.01
        detector = PersonDetector()
        detections = []
        tracking_moves = 0
        centers = []
        robot.following = True
        robot.pause_tracking_until = 0
        target_updates = []
        if robot.responsive_tracking:
            move_to = camera.move_to
            def record_target(x, y, speed=None):
                target_updates.append((x, y))
                return move_to(x, y, speed)
            camera.move_to = record_target
            threading.Thread(target=robot._tracking_loop, daemon=True).start()
        steps = (45 if args.tracking_only else 20) if robot.responsive_tracking else (7 if args.tracking_only else 3)
        for _ in range(steps):
            frame = robot.frames.latest()
            if frame:
                detections = detector.detect(frame[2])
                centers.append([box.center for box in detections])
                before = robot.position
                robot.observe(detections, time.monotonic(), observed_at=frame[0])
                tracking_moves += robot.position != before
            done.wait(0.12 if robot.responsive_tracking else 0.85)
        report.update({"passed": direction_ok and counts["video"] > 5 and counts["audio"] > 0
                       and (not args.tracking_only or any(centers)),
                       "reply": answer, "cloud_metrics": client.metrics,
                       "start_position": start, "after_look": after_look,
                       "people_or_faces_detected": len(detections), "tracking_corrections": tracking_moves,
                       "tracking_mode": robot.context()["tracking_mode"], "tracking_commands": len(target_updates),
                       "detected_centers": centers,
                       "frames": counts, "stream_errors": errors})
    except Exception as error:
        from assistant import AssistantError
        report["error"] = str(error) if isinstance(error, (CameraError, AssistantError)) else "Physical camera test could not finish."
    finally:
        robot.following = False
        robot.closed.set()
        robot.motion_cancel.set()
        try:
            with robot.motion_lock:
                robot._halt_tracking(force=True)
                camera.move_to(*start)
                camera.wait_position(start, timeout=10)
                camera.stop()
                restored = camera.position()
            report["restored_position"] = restored
            report["restored"] = max(abs(restored[i] - start[i]) for i in (0, 1)) <= 0.002
        except CameraError as error:
            report["restored"] = False
            report["restore_error"] = str(error)
        done.set()
        robot.close()
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] and report.get("restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
