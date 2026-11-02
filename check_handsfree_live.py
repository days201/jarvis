"""Test camera wake input and spoken replies with session credentials on stdin."""

import argparse
import json
import sys
import threading
import time
import tkinter as tk

import numpy as np
import sounddevice as sd

from check_handsfree import synthesize_pcm
from desktop import Desktop
from speech import load_tts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject", action="store_true", help="Feed test PCM without playing the PC speaker.")
    parser.add_argument("--wake-only", action="store_true", help="Say Jarvis, wait for acknowledgement, then ask a question.")
    args = parser.parse_args()
    login = json.loads(sys.stdin.read())
    tts = load_tts()
    pcm = synthesize_pcm(tts, "Jarvis" if args.wake_only else "Hey Jarvis, what is two plus two?")
    followup = synthesize_pcm(tts, "What is six times seven?") if args.wake_only else None
    root = tk.Tk()
    app = Desktop(root, login=login, api_key=login.get("api_key"))
    del login
    speech = app.conversation.speech
    started = time.monotonic()
    report = {"test_input": "injected PCM" if args.inject else "computer speaker heard by camera microphone",
              "events": [], "result": "timeout"}
    original_emit = speech.emit
    original_feed = app.conversation.feed_audio
    state = {"started": False, "injecting": False, "input_end": None, "reply_audio": None,
             "reply_complete": None, "wakes": 0, "acknowledged": False,
             "followup_at": None, "followup_started": False}

    def observe(kind, value):
        now = time.monotonic()
        if kind in {"wake_candidate", "wake_detected", "recognized", "acknowledge", "voice_state", "tts_error", "recognition_error"}:
            report["events"].append((round(now - started, 3), kind, value))
        if kind == "wake_detected":
            state["wakes"] += 1
        if kind == "acknowledge":
            state["acknowledged"] = True
        if kind == "output_idle" and args.wake_only and state["acknowledged"] and not state["followup_started"]:
            state["followup_at"] = now + 0.7
        if kind == "speaking" and state["started"] and state["reply_audio"] is None and (not args.wake_only or state["followup_started"]):
            state["reply_audio"] = now
        original_emit(kind, value)

    def camera_feed(data):
        if not state["injecting"]:
            original_feed(data)

    speech.emit = observe
    app.conversation.feed_audio = camera_feed

    def test_input(data):
        if args.inject:
            state["injecting"] = True
            audio = b"\0" * 16000 + data + b"\0" * 48000
            next_frame = time.monotonic()
            for offset in range(0, len(audio), 1280):
                speech.feed(audio[offset:offset + 1280])
                next_frame += 0.04
                time.sleep(max(0, next_frame - time.monotonic()))
                if offset >= 16000 + len(data) and state["input_end"] is None:
                    state["input_end"] = time.monotonic()
            state["injecting"] = False
        else:
            sd.play(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768 * 0.8, 16000)
            sd.wait()
            state["input_end"] = time.monotonic()

    def finish():
        report.update({"elapsed_seconds": round(time.monotonic() - started, 3),
                       "voice_output": app.conversation.output.get(),
                       "played_chunks": speech.played_chunks,
                       "camera_audio_frames": app.audio_frames_seen,
                       "video_frames": app.frame_count,
                       "wake_count": state["wakes"],
                       "acknowledged": state["acknowledged"],
                       "transcript": app.conversation.history.get("1.0", "end").strip()})
        if state["reply_audio"] is not None and state["input_end"] is not None:
            report["question_end_to_speech_seconds"] = round(state["reply_audio"] - state["input_end"], 3)
        print(json.dumps(report), flush=True)
        app.close()

    def poll():
        if app.closing:
            return
        now = time.monotonic()
        if not state["started"] and speech.voice_state == "wake" and app.frame_count > 5:
            state["started"] = True
            threading.Thread(target=test_input, args=(pcm,), daemon=True).start()
        if args.wake_only and state["followup_at"] is not None and now >= state["followup_at"] and not state["followup_started"]:
            state["followup_started"] = True
            state["input_end"] = None
            threading.Thread(target=test_input, args=(followup,), daemon=True).start()
        transcript = app.conversation.history.get("1.0", "end").lower()
        if state["started"] and speech.played_chunks and not app.conversation.busy and speech.voice_state == "wake":
            if state["reply_complete"] is None:
                state["reply_complete"] = now
            if now - state["reply_complete"] >= 1.2:
                answer = ("forty-two" in transcript or "42" in transcript or "forty two" in transcript) if args.wake_only else ("four" in transcript or "4" in transcript)
                report["result"] = "passed" if answer and state["wakes"] == 1 else "unexpected reply or extra wake"
                finish()
                return
        if now - started > 26:
            finish()
            return
        root.after(50, poll)

    root.after(100, poll)
    root.mainloop()
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
