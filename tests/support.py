"""Shared simulated devices and in-memory speech for tests."""

import json
import av

from robot.robot import Robot


class SimulatedCamera:
    def __init__(self):
        self.pose = (0.0, 0.0)
        self.moves = []
        self.stops = 0

    def move_to(self, x, y):
        self.pose = (x, y)
        self.moves.append(self.pose)

    def wait_position(self, target, **kwargs):
        return self.pose

    def stop(self):
        self.stops += 1

class Decoder:
    def __init__(self):
        self.text = ""
        self.calls = 0

    def Reset(self):
        self.text = ""

    def AcceptWaveform(self, data):
        self.calls += 1
        return False

    def PartialResult(self):
        return json.dumps({"partial": self.text})

    def FinalResult(self):
        return json.dumps({"text": self.text})


class Activity:
    voiced = True

    def reset(self):
        pass

    def is_speech(self, data):
        return self.voiced

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

class SimulatedRobot(Robot):
    def context(self):
        return {**super().context(), "simulation": True, "note": "Motors and image source are simulated for this test."}
