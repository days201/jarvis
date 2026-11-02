"""Local wake detection and bounded voice turns for the camera microphone."""

from collections import deque
from dataclasses import dataclass
import json
import re
import time

import numpy as np


@dataclass(frozen=True)
class VoiceTiming:
    end_silence: float = 0.64
    acknowledge_after: float = 0.85
    question_timeout: float = 7.0
    maximum_question: float = 25.0
    echo_tail: float = 0.55
    stream_timeout: float = 3.0
    maximum_backlog: float = 0.75
    preroll: float = 2.4


def wake_prefix(text):
    return re.match(r"^(?:hey\s+)?jarvis\b", text.strip().lower()) is not None


def remove_wake(text):
    # The full decoder can spell the name differently from the wake decoder.
    return re.sub(r"^(?:(?:hey|a|hi|okay|ok)\s+)?(?:jarvis|jervis|travis)\b[\s,.!?]*",
                  "", text.strip(), count=1, flags=re.I).strip()


def make_wake_recognizer(model):
    from vosk import KaldiRecognizer
    recognizer = KaldiRecognizer(model, 16000, json.dumps(["hey jarvis", "jarvis", "[unk]"]))
    recognizer.SetWords(True)
    return recognizer


class SileroVAD:
    """Run the publisher's recurrent ONNX model with one CPU thread."""

    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.pending = b""
        self.last_probability = 0.0

    def is_speech(self, pcm):
        self.pending += pcm
        probabilities = []
        while len(self.pending) >= 1024:
            raw, self.pending = self.pending[:1024], self.pending[1024:]
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)[None, :] / 32768
            samples = np.concatenate((self.context, samples), axis=1)
            probability, self.state = self.session.run(None, {
                "input": samples, "state": self.state, "sr": np.array(16000, dtype=np.int64)})
            self.context = samples[:, -64:]
            self.last_probability = float(probability.reshape(-1)[0])
            probabilities.append(self.last_probability)
        return max(probabilities, default=self.last_probability) >= 0.45


class VoiceEngine:
    """Single-thread state machine. Only the recognition worker calls methods."""

    def __init__(self, wake, recognizer, vad, emit, next_identifier, timing=None):
        self.wake = wake
        self.recognizer = recognizer
        self.vad = vad
        self.emit = emit
        self.next_identifier = next_identifier
        self.timing = timing or VoiceTiming()
        self.enabled = False
        self.online = False
        self.busy = False
        self.output_active = False
        self.resume_after = 0.0
        self.last_packet = None
        self.clock = 0.0
        self.mode = "off"
        self.identifier = None
        self.buffer = deque()
        self.buffer_seconds = 0.0
        self.segments = []
        self.last_partial = ""
        self.wake_candidate = None
        self.last_voice = -100.0
        self.started = 0.0
        self.acknowledged = False
        self._state = None
        self.wake_samples = 0
        self.command_has_wake_audio = True
        self.followup_until = 0.0
        self.followup_voice = 0.0
        self.from_followup = False
        self.turn_has_speech = False

    @property
    def remote_turns(self):
        return bool(getattr(self.recognizer, "remote_turns", False))

    def state(self, value):
        if value != self._state:
            self._state = value
            self.emit("voice_state", value)

    def reset_audio(self):
        self.wake.Reset()
        self.recognizer.Reset()
        self.vad.reset()
        self.buffer.clear()
        self.buffer_seconds = 0.0
        self.segments = []
        self.last_partial = ""
        self.wake_candidate = None
        self.last_voice = -100.0
        self.followup_voice = 0.0
        self.turn_has_speech = False
        self.wake_samples = 0

    def idle(self):
        self.identifier = None
        self.reset_audio()
        self.mode = "wake" if self.enabled and self.online and not self.busy else "off"
        if self.mode == "wake" and time.monotonic() < self.followup_until:
            self.mode = "followup"
        self.show_state()

    def show_state(self):
        if not self.online:
            self.state("offline")
        elif self.output_active:
            self.state("speaking")
        elif self.busy or self.mode == "waiting":
            self.state("thinking")
        elif self.mode in {"command", "manual"}:
            self.state("listening")
        elif self.mode == "followup":
            self.state("followup")
        elif self.enabled:
            self.state("wake")
        else:
            self.state("muted")

    def enable(self, value):
        value = bool(value)
        if value != self.enabled:
            self.enabled = value
            self.followup_until = 0.0
            self.idle()

    def set_online(self, value):
        value = bool(value)
        if value != self.online:
            if self.identifier is not None:
                self.emit("listen_cancelled", self.identifier)
            self.online = value
            self.followup_until = 0.0
            self.last_packet = None
            self.idle()

    def set_busy(self, value):
        self.busy = bool(value)
        if self.busy:
            self.followup_until = 0.0
        self.idle()

    def open_followup(self, seconds, now):
        if not self.enabled or not self.online or self.busy:
            return
        self.followup_until = max(now, self.resume_after) + min(15, max(0, seconds))
        self.reset_audio()
        self.mode = "followup"
        self.show_state()

    def set_output(self, value, now):
        self.output_active = bool(value)
        self.reset_audio()
        if not self.output_active:
            self.resume_after = now + self.timing.echo_tail
        self.show_state()

    def start_manual(self, identifier):
        self.reset_audio()
        self.identifier = identifier
        self.started = self.clock
        self.mode = "manual"
        self.from_followup = False
        self.acknowledged = True
        self.show_state()

    def cancel(self):
        self.followup_until = 0.0
        self.idle()

    def discontinuity(self):
        self.followup_until = 0.0
        if self.identifier is not None:
            self.emit("listen_cancelled", self.identifier)
        self.idle()

    def tick(self, now):
        if self.online and self.last_packet is not None and now - self.last_packet > self.timing.stream_timeout:
            self.set_online(False)
            return
        self._poll_recognizer()
        if self.mode == "followup" and now >= self.followup_until:
            self.followup_until = 0.0
            self.idle()

    def _remember(self, data, duration):
        self.buffer.append((data, duration))
        self.buffer_seconds += duration
        while self.buffer_seconds > self.timing.preroll and self.buffer:
            _, removed = self.buffer.popleft()
            self.buffer_seconds -= removed

    def _decode(self, data):
        if self.remote_turns:
            self.recognizer.feed(data)
            self._poll_recognizer()
            return self.last_partial
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result()).get("text", "").strip()
            if result:
                self.segments.append(result)
            partial = ""
        else:
            partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
        text = " ".join(self.segments + ([partial] if partial else []))
        return remove_wake(text) if (self.mode == "command" and self.command_has_wake_audio) or self.from_followup else text

    def _poll_recognizer(self):
        if not self.remote_turns:
            return
        for kind, text in self.recognizer.poll():
            if self.mode not in {"command", "manual"}:
                return
            if kind == "error":
                identifier = self.identifier
                self.followup_until = 0.0
                self.idle()
                self.emit("listen_cancelled", identifier)
                self.emit("recognition_error", text)
                return
            if kind == "speech_started":
                self.turn_has_speech = True
                continue
            if self.mode == "command" or self.from_followup:
                text = remove_wake(text)
            if kind == "final":
                if not text and self.mode == "command" and not self.acknowledged:
                    self.recognizer.Reset()
                    self.turn_has_speech = False
                    continue
                self._complete(text)
                return
            if text and text != self.last_partial:
                self.last_partial = text
                self.emit("partial", (self.identifier, text))

    def _activate(self, alignment):
        words = [word for word in alignment.get("result", []) if word.get("word") == "jarvis"]
        audio = b"".join(data for data, _ in self.buffer)
        self.command_has_wake_audio = not words
        if words:
            # Cut the wake audio by its acoustic boundary, independent of how
            # the question decoder would spell the assistant's name.
            tail_seconds = self.wake_samples / 16000 - float(words[-1]["end"]) - 0.02
            keep_bytes = max(0, round(tail_seconds * 16000)) * 2
            audio = b"\0" * 4800 + (audio[-keep_bytes:] if keep_bytes else b"")
        self.identifier = self.next_identifier()
        self.mode = "command"
        self.from_followup = False
        self.started = self.clock
        self.acknowledged = False
        self.segments = []
        self.recognizer.Reset()
        if audio:
            self._decode(audio)
        self.buffer.clear()
        self.buffer_seconds = 0.0
        self.emit("wake_detected", self.identifier)
        self.show_state()

    def finish(self):
        if self.mode not in {"command", "manual"}:
            return
        if self.remote_turns:
            if self.recognizer.finish():
                self.state("transcribing")
            else:
                self._complete("")
            return
        result = json.loads(self.recognizer.FinalResult()).get("text", "").strip()
        text = " ".join(self.segments + ([result] if result else []))
        if (self.mode == "command" and self.command_has_wake_audio) or self.from_followup:
            text = remove_wake(text)
        self._complete(text)

    def _complete(self, text):
        identifier = self.identifier
        self.mode = "waiting"
        self.reset_audio()
        self.state("thinking")
        self.emit("recognized", (identifier, text))

    def process(self, pcm, received, now):
        self.last_packet = now
        if not self.online:
            self.set_online(True)
            self.last_packet = now
        if now - received > self.timing.maximum_backlog:
            self.discontinuity()
            return
        if self.mode == "followup" and now >= self.followup_until:
            self.followup_until = 0.0
            self.idle()
        if self.output_active or received < self.resume_after or self.busy:
            return
        self._poll_recognizer()
        duration = len(pcm) / 32000
        self.clock += duration
        if self.mode not in {"wake", "command", "manual", "followup"}:
            return
        voiced = self.vad.is_speech(pcm)
        if voiced:
            self.last_voice = self.clock
        if self.mode == "followup":
            self._remember(pcm, duration)
            self.followup_voice = self.followup_voice + duration if voiced else 0.0
            if self.followup_voice >= 0.12:
                audio = b"".join(data for data, _ in self.buffer)
                identifier = self.next_identifier()
                self.start_manual(identifier)
                self.from_followup = True
                self.followup_until = 0.0
                self.last_voice = self.clock
                self._decode(audio)
                self.emit("followup_started", identifier)
            return
        if self.mode == "wake":
            self._remember(pcm, duration)
            self.wake_samples += len(pcm) // 2
            final = bool(self.wake.AcceptWaveform(pcm))
            result = json.loads(self.wake.Result() if final else self.wake.PartialResult())
            text = result.get("text" if final else "partial", "").strip()
            if wake_prefix(text) and self.clock - self.last_voice < 0.8:
                if self.wake_candidate is None:
                    self.wake_candidate = self.clock
                    self.emit("wake_candidate", None)
                if final or self.clock - self.wake_candidate >= 0.28:
                    alignment = result if final else json.loads(self.wake.FinalResult())
                    if wake_prefix(alignment.get("text", "")):
                        self._activate(alignment)
                    else:
                        self.wake_candidate = None
            else:
                self.wake_candidate = None
            if final and self.mode == "wake":
                self.buffer.clear()
                self.buffer_seconds = 0.0
            return
        text = self._decode(pcm)
        if self.mode not in {"command", "manual"}:
            return
        if text and text != self.last_partial:
            self.last_partial = text
            self.emit("partial", (self.identifier, text))
        elapsed = self.clock - self.started
        silence = self.clock - self.last_voice
        if self.remote_turns:
            if self.recognizer.finishing:
                return
            if elapsed >= self.timing.maximum_question:
                self.finish()
            elif not text and not self.turn_has_speech and elapsed >= self.timing.question_timeout:
                identifier = self.identifier
                self.idle()
                self.emit("listen_timeout", identifier)
            elif self.mode == "command" and not text and not self.turn_has_speech and not self.acknowledged and silence >= self.timing.acknowledge_after:
                self.acknowledged = True
                self.emit("acknowledge", self.identifier)
        elif text and silence >= self.timing.end_silence:
            self.finish()
        elif elapsed >= self.timing.maximum_question:
            self.finish()
        elif not text and elapsed >= self.timing.question_timeout:
            identifier = self.identifier
            self.idle()
            self.emit("listen_timeout", identifier)
        elif self.mode == "command" and not text and not self.acknowledged and silence >= self.timing.acknowledge_after:
            self.acknowledged = True
            self.emit("acknowledge", self.identifier)
