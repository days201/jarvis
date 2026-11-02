"""Check cloud vision, optional cloud speech, and simulated camera movement."""

import argparse
import json
import queue
import re
import sys
import threading
import time
import tkinter as tk

from PIL import Image, ImageDraw, ImageFont

from check_cloud_robot import SimulatedRobot
from check_handsfree import synthesize_pcm
from desktop import Desktop
from speech import load_tts
from test_robot import SimulatedCamera
from tapo_talk import TalkError


class SilentSpeaker:
    def open(self):
        return {"authenticated": True}

    def close(self):
        pass

    def play(self, samples, rate, stopped):
        time.sleep(0.15)
        return not stopped()

    def play_stream(self, chunks, stopped):
        played = False
        for chunk in chunks:
            if stopped():
                return False
            played = played or bool(chunk.audio_float_array.size)
        return played and not stopped()


class DesktopCamera(SimulatedCamera):
    def create_speaker(self, password=None):
        return SilentSpeaker()


class SpeakerUnavailableCamera(DesktopCamera):
    def create_speaker(self, password=None):
        raise TalkError("Camera speaker login was rejected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker-unavailable', action='store_true')
    args = parser.parse_args()
    login = json.loads(sys.stdin.read())
    pcm = synthesize_pcm(load_tts(), "What is nine multiplied by nine?")
    root = tk.Tk()
    app = Desktop(root, api_key=login.pop("api_key"), deepgram_key=login.pop("deepgram_key", None))
    del login
    camera = SpeakerUnavailableCamera() if args.speaker_unavailable else DesktopCamera()
    if args.speaker_unavailable:
        # Consume computer-output audio silently while exercising real cloud
        # synthesis, output completion, and follow-up timing.
        app.conversation.speech._play_chunks = SilentSpeaker().play_stream
    robot = SimulatedRobot(camera, (0, 0), start=False)
    robot.following = False
    robot.greetings = False
    app.camera = camera
    app.robot = robot
    app.desk_position = (0, 0)
    app.login_panel.grid_remove()
    app.details.set("Simulation. No physical camera connection.")
    app.conversation.set_robot(robot)
    app.conversation.set_camera(camera)
    app.conversation.set_microphone(True)
    app.enable_controls(True)
    image = Image.new("RGB", (800, 450), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 70, 330, 370), fill="blue")
    draw.text((430, 140), "42", font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 110), fill="black")
    started = time.monotonic()
    done = threading.Event()
    state = {"phase": 0, "pcm": b"", "followup_at": None, "followup_started": False, "wakes": 0}
    lock = threading.Lock()
    emit = app.conversation.speech.emit
    def observe(kind, value):
        if kind == "followup_started":
            state["followup_started"] = True
        elif kind == "wake_detected":
            state["wakes"] += 1
        emit(kind, value)
    app.conversation.speech.emit = observe
    def feed():
        while not done.is_set():
            robot.frames.push(image)
            try:
                app.frames.put_nowait(image.copy())
            except queue.Full:
                pass
            with lock:
                audio = state["pcm"][:1280] if state["pcm"] else b"\0" * 1280
                state["pcm"] = state["pcm"][1280:]
            app.conversation.feed_audio(audio)
            done.wait(0.04)
    threading.Thread(target=feed, daemon=True).start()
    result = {"passed": False, "simulation": True}
    def finish():
        layouts = {}
        for geometry in ("1280x880", "1100x800"):
            root.geometry(geometry)
            for panel in (app.controls_panel, app.conversation):
                app.side_tabs.select(panel)
                root.update_idletasks()
                layouts[geometry + ("/movement" if panel is app.controls_panel else "/conversation")] = all(
                    child.winfo_y() + child.winfo_height() <= panel.winfo_height()
                    for child in panel.winfo_children() if child.winfo_ismapped())
        history = app.conversation.history.get("1.0", "end").strip()
        last_answer = app.conversation.client._history[-1]["content"] if app.conversation.client._history else ""
        result.update({"passed": state["phase"] == 2 and state["followup_started"] and state["wakes"] == 0
                       and bool(re.search(r"\b(eighty[- ]one|81)\b", last_answer, re.I)) and all(layouts.values())
                       and (not args.speaker_unavailable or (app.conversation.speech._camera_speaker is None
                            and app.conversation.speech.hands_free.is_set()
                            and app.conversation.speech.output == 'computer')),
                       "transcript": history, "followup_without_wake": state["followup_started"],
                       "simulated_moves": camera.moves, "spoken_chunks": app.conversation.speech.played_chunks,
                       "voice_state": app.conversation.speech.voice_state, "layouts": layouts,
                       "transcription_provider": app.conversation.speech.recognition_provider,
                       "tts_provider": app.conversation.speech.tts_provider,
                       "speaker_unavailable_scenario": args.speaker_unavailable,
                       "hands_free": app.conversation.speech.hands_free.is_set(),
                       "voice_output": app.conversation.speech.output,
                       "elapsed_seconds": round(time.monotonic() - started, 3)})
        print(json.dumps(result), flush=True)
        done.set()
        app.close()
    def poll():
        speech = app.conversation.speech
        if state["phase"] == 0 and speech.hands_free.is_set() and speech.voice_state == "wake":
            state["phase"] = 1
            app.conversation.prompt.set("This is a simulation test. Use the look tool to turn right 10 degrees once. Then tell me the rectangle color and the printed number in one short sentence.")
            app.conversation.send()
        elif state["phase"] == 1 and speech.voice_state == "followup":
            if state["followup_at"] is None:
                state["followup_at"] = time.monotonic() + 0.75
            if time.monotonic() >= state["followup_at"]:
                state["phase"] = 2
                with lock:
                    state["pcm"] = pcm + b"\0" * 32000
        elif state["phase"] == 2 and app.conversation.turn >= 2 and speech.voice_state == "followup" and not app.conversation.busy:
            finish()
            return
        if time.monotonic() - started > 35:
            finish()
            return
        root.after(50, poll)
    root.after(100, poll)
    root.mainloop()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
