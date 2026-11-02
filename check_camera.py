"""Check authenticated video and optionally a small, reversible pan movement."""

import argparse
import getpass
import json
from pathlib import Path
import sys
import time

from camera import Camera, CameraError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin-login", action="store_true", help="Read credentials from an in-memory parent process.")
    parser.add_argument("--motion", action="store_true", help="Move pan a few degrees and restore the starting position.")
    args = parser.parse_args()
    if args.stdin_login:
        login = json.loads(sys.stdin.read())
    else:
        login = {"host": input("Camera IP [192.168.2.88]: ").strip() or "192.168.2.88",
                 "username": input("Camera Account username: ").strip(),
                 "password": getpass.getpass("Camera Account password: ")}
    camera = Camera(login["host"], login["username"], login["password"])
    del login
    report = {"host": camera.host, "success": False}
    try:
        report["device"] = camera.connect()
        report["starting_position"] = camera.position()
        started = time.monotonic()
        frames = 0
        with camera.open_video(low_quality=False) as video:
            for frame in video.decode(video=0):
                frames += 1
                report["decoded_video"] = {"width": frame.width, "height": frame.height, "frames": frames}
                if frames >= 15:
                    break
                if time.monotonic() - started > 12:
                    raise CameraError("Timed out while checking live video.")
        if frames < 15:
            raise CameraError("The camera did not supply enough video frames.")
        report["video_test_seconds"] = round(time.monotonic() - started, 2)
        if args.motion:
            start = camera.position()
            target = (start[0] + (0.02 if start[0] < 0.95 else -0.02), start[1])
            report["motion"] = {"start": start, "target": target, "restored": False}
            try:
                camera.move_to(*target)
                report["motion"]["reached"] = camera.wait_position(target)
            finally:
                try:
                    camera.stop()
                finally:
                    camera.move_to(*start)
                    report["motion"]["final"] = camera.wait_position(start)
                    camera.stop()
                    report["motion"]["restored"] = True
        report["success"] = True
    except CameraError as error:
        report["error"] = str(error)
    except Exception:
        report["error"] = "The camera check failed. No raw network error was logged."
    folder = Path(__file__).resolve().parent / "diagnostics"
    folder.mkdir(exist_ok=True)
    (folder / "connection-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
