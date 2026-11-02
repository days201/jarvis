"""Exercise real local wake and question models with speech generated in memory."""

import itertools
import json
import time

import av
from vosk import KaldiRecognizer

from handsfree import SileroVAD, VoiceEngine, make_wake_recognizer
from speech import MODELS, load_recognizer_model, load_tts


def synthesize_pcm(tts, text):
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    result = bytearray()
    for chunk in tts.synthesize(text):
        frame = av.AudioFrame.from_ndarray(chunk.audio_float_array[None, :], format="flt", layout="mono")
        frame.sample_rate = chunk.sample_rate
        for converted in resampler.resample(frame):
            result.extend(converted.to_ndarray().tobytes())
    for converted in resampler.resample(None):
        result.extend(converted.to_ndarray().tobytes())
    return bytes(result)


def main():
    model, tts = load_recognizer_model(), load_tts()
    reports = []
    for phrase in ("Hey Jarvis, what is two plus two?", "Jarvis, what is the capital of France?",
                   "What is two plus two?", "Hey Jarvis", "Jarvis"):
        events = []
        identifiers = itertools.count(1)
        wake = make_wake_recognizer(model)
        engine = VoiceEngine(wake, KaldiRecognizer(model, 16000), SileroVAD(MODELS / "silero_vad.onnx"),
                             lambda kind, value: events.append((kind, value, round(engine.clock, 3))),
                             lambda: next(identifiers))
        engine.set_online(True)
        engine.enable(True)
        pcm = b"\0" * 16000 + synthesize_pcm(tts, phrase) + b"\0" * 64000
        started = time.perf_counter()
        now = 100.0
        for offset in range(0, len(pcm), 1280):
            now += 0.04
            engine.process(pcm[offset:offset + 1280], now, now)
        reports.append({"phrase": phrase, "processing_seconds": round(time.perf_counter() - started, 3),
                        "audio_seconds": round(len(pcm) / 32000, 3),
                        "events": [event for event in events if event[0] != "partial"]})
    print(json.dumps(reports, indent=2))
    return 0 if (any(e[0] == "recognized" for e in reports[0]["events"])
                 and any(e[0] == "recognized" for e in reports[1]["events"])
                 and not any(e[0] == "wake_detected" for e in reports[2]["events"])
                 and any(e[0] == "acknowledge" for e in reports[3]["events"])
                 and any(e[0] == "acknowledge" for e in reports[4]["events"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
