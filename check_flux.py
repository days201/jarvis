"""Check real Flux recognition using generated speech and a key supplied on stdin."""

import itertools
import json
import re
import sys
import time

import av
import numpy as np

from check_handsfree import synthesize_pcm
from flux import FluxRecognizer
from handsfree import SileroVAD, VoiceEngine, make_wake_recognizer
from speech import MODELS, load_recognizer_model, load_tts
from tapo_talk import encode_alaw


def camera_bandwidth(pcm):
    """Approximate the camera's 8 kHz A-law source without using its microphone."""
    encoded = encode_alaw(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768, 16000)
    decoder = av.CodecContext.create("pcm_alaw", "r")
    decoder.sample_rate = 8000
    decoder.layout = "mono"
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    result = bytearray()
    for frame in decoder.decode(av.Packet(encoded)) + decoder.decode(None):
        for converted in resampler.resample(frame):
            result.extend(converted.to_ndarray().tobytes())
    for converted in resampler.resample(None):
        result.extend(converted.to_ndarray().tobytes())
    return bytes(result)


def main():
    login = json.loads(sys.stdin.read())
    key = login.pop("deepgram_key")
    del login
    tts, model = load_tts(), load_recognizer_model()
    reports = []
    cases = [
        ("inline wake", "Hey Jarvis, what is two plus two?", None, "two plus two"),
        ("follow-up with a pause", "Could you please tell me", "what is six times seven?", "six times seven"),
    ]
    for name, first, second, expected in cases:
        events = []
        identifiers = itertools.count(1)
        recognizer = FluxRecognizer(key)
        started = time.monotonic()
        engine = VoiceEngine(make_wake_recognizer(model), recognizer, SileroVAD(MODELS / "silero_vad.onnx"),
                             lambda kind, value: events.append((kind, value, time.monotonic() - started)),
                             lambda: next(identifiers))
        engine.set_online(True)
        engine.enable(True)
        if second:
            engine.open_followup(15, time.monotonic())
        pcm = b"\0" * 16000 + camera_bandwidth(synthesize_pcm(tts, first))
        if second:
            pcm += b"\0" * 28800 + camera_bandwidth(synthesize_pcm(tts, second))
        speech_end = len(pcm) / 32000
        pcm += b"\0" * 192000
        deadline = time.monotonic()
        try:
            for offset in range(0, len(pcm), 1280):
                now = time.monotonic()
                engine.process(pcm[offset:offset + 1280], now, now)
                engine.tick(now)
                if any(kind in {"recognized", "recognition_error"} for kind, _, _ in events):
                    break
                deadline += 0.04
                time.sleep(max(0, deadline - time.monotonic()))
            recognized = [(value[1], elapsed) for kind, value, elapsed in events if kind == "recognized"]
            transcript = recognized[-1][0] if recognized else ""
            normalized = re.sub(r"[^a-z0-9 ]", "", transcript.lower())
            reports.append({"case": name, "transcript": transcript, "passed": expected in normalized,
                            "question_end_to_final_seconds": round(recognized[-1][1] - speech_end, 3) if recognized else None,
                            "errors": [value for kind, value, _ in events if kind == "recognition_error"]})
        finally:
            engine.cancel()
    print(json.dumps({"generated_speech": True, "camera_bandwidth_simulation": True,
                      "no_camera_recording": True, "results": reports}), flush=True)
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
