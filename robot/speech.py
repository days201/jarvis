"""Cloud or local speech output with local wake detection and camera audio."""

import json
from .paths import MODELS
import queue
import re
import threading
import time

from .speech_stream import ReplyText, buffered_audio





def load_tts():
    from piper import PiperVoice
    return PiperVoice.load(str(MODELS / "en_GB-alan-medium.onnx"), use_cuda=False)


def load_recognizer_model():
    from vosk import Model, SetLogLevel
    SetLogLevel(-1)
    return Model(str(MODELS / "vosk-model-small-en-us-0.15"))


def clean_speech(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return text.strip()


class Speech:
    def __init__(self, emit, deepgram_key=None):
        self.emit = emit
        self.closed = threading.Event()
        self.listening = threading.Event()
        self.hands_free = threading.Event()
        self.recognition_ready = threading.Event()
        self.tts_ready = threading.Event()
        self.audio = queue.Queue(maxsize=48)
        self.controls = queue.Queue()
        self.sentences = queue.Queue()
        self._epoch = 0
        self._epoch_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._provider_lock = threading.Lock()
        self._cloud_voice = None
        self._computer_stream = None
        self._camera_speaker = None
        self.output = "computer"
        self._listen_id = 0
        self._listen_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_speech = 0
        self._turn_busy = False
        self._followup_pending = 0
        self.voice_state = "offline"
        self.recognition_provider = "Vosk local"
        self.tts_provider = "Piper Alan"
        self.played_chunks = 0
        if deepgram_key:
            self.set_transcription_key(deepgram_key)
        threading.Thread(target=self._synthesis_loop, daemon=True, name="jarvis-speech").start()
        threading.Thread(target=self._recognition_loop, daemon=True, name="jarvis-listening").start()

    def speak(self, text, cancelled=None):
        text = clean_speech(text)
        if text and not self.closed.is_set() and not (cancelled and cancelled.is_set()):
            reply = self.begin_reply(cancelled)
            reply.write(text)
            reply.finish()

    def begin_reply(self, cancelled=None):
        with self._epoch_lock:
            epoch = self._epoch
        reply = ReplyText(lambda: not self._current(epoch) or bool(cancelled and cancelled.is_set()))
        with self._pending_lock:
            self._pending_speech += 1
            if self._pending_speech == 1:
                self.controls.put(("output", True))
        self.sentences.put((epoch, reply, cancelled))
        return reply

    @property
    def microphone_active(self):
        return self.hands_free.is_set() or self.listening.is_set()

    def set_hands_free(self, enabled):
        if bool(enabled) == self.hands_free.is_set():
            return
        if enabled:
            self.hands_free.set()
        else:
            self.hands_free.clear()
            self.listening.clear()
        self.controls.put(("enable", bool(enabled)))

    def set_microphone(self, available):
        self.controls.put(("online", bool(available)))

    def set_transcription_key(self, key):
        from .flux import FluxRecognizer
        from .cloud_voice import FluxVoice, VOICE_LABEL
        key = key.strip()
        if key:
            FluxRecognizer(key)
        voice = FluxVoice(key) if key else None
        self.stop_speaking()
        with self._provider_lock:
            previous, self._cloud_voice = self._cloud_voice, voice
            self.tts_provider = VOICE_LABEL if voice else "Piper Alan"
        if previous is not None:
            previous.close()
        self.emit("tts_provider_changed", self.tts_provider)
        if voice is not None:
            self.tts_ready.set()
        self.controls.put(("transcription_key", key))

    def set_turn_busy(self, busy):
        self._turn_busy = bool(busy)
        self.controls.put(("busy", self._turn_busy))

    def reset_voice_context(self):
        with self._provider_lock:
            voice = self._cloud_voice
        if voice is not None:
            voice.close()

    def open_followup(self, seconds=15):
        with self._pending_lock:
            if self._pending_speech:
                self._followup_pending = seconds
            else:
                self.controls.put(("followup", seconds))

    def _next_identifier(self):
        with self._listen_lock:
            self._listen_id += 1
            return self._listen_id

    def _voice_event(self, kind, value):
        if kind in {"wake_detected", "followup_started"}:
            self.listening.set()
        elif kind in {"recognized", "listen_timeout", "listen_cancelled"}:
            self.listening.clear()
        elif kind == "voice_state":
            self.voice_state = value
        elif kind == "acknowledge":
            self.speak("Yes?")
        self.emit(kind, value)

    def stop_speaking(self):
        with self._pending_lock:
            self._followup_pending = 0
        self.controls.put(("close_followup", None))
        with self._epoch_lock:
            self._epoch += 1
        try:
            import sounddevice as sd
            with self._playback_lock:
                sd.stop()
                if self._computer_stream is not None:
                    self._computer_stream.abort()
        except Exception:
            pass
        speaker = self._camera_speaker
        if speaker is not None:
            speaker.close()
        with self._provider_lock:
            voice = self._cloud_voice
        if voice is not None:
            voice.cancel_active()

    def set_camera_speaker(self, speaker, select=True):
        self.stop_speaking()
        with self._playback_lock:
            self._camera_speaker = speaker
            if select:
                self.output = "camera"

    def select_output(self, destination):
        from .tapo_talk import TalkError
        if destination not in {"camera", "computer"}:
            raise TalkError("Choose camera speaker or computer speakers.")
        if destination == "camera" and self._camera_speaker is None:
            raise TalkError("Camera speaker is not connected. Use Keys to connect it.")
        self.stop_speaking()
        with self._playback_lock:
            self.output = destination

    def _current(self, epoch):
        with self._epoch_lock:
            return epoch == self._epoch and not self.closed.is_set()

    def _synthesis_loop(self):
        local_tts = None
        try:
            with self._provider_lock:
                cloud = self._cloud_voice
            if cloud is None:
                local_tts = load_tts()
                list(local_tts.synthesize("Ready."))
            self.tts_ready.set()
            self.emit("tts_ready", None)
        except Exception:
            self.emit("tts_error", "Local speech could not start. Run python -m tools.setup_models and check the audio output device.")
        while not self.closed.is_set():
            try:
                epoch, reply, cancelled = self.sentences.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if not self._current(epoch) or (cancelled and cancelled.is_set()):
                    continue
                stopped = lambda: not self._current(epoch) or bool(cancelled and cancelled.is_set())
                with self._provider_lock:
                    cloud = self._cloud_voice
                if cloud is not None:
                    chunks = buffered_audio(lambda halted: cloud.synthesize_stream(reply, halted), stopped)
                else:
                    if local_tts is None:
                        local_tts = load_tts()
                    def local_chunks():
                        for text in reply(stopped):
                            yield from local_tts.synthesize(text)
                    chunks = local_chunks()
                if self._play_chunks(chunks, stopped) and self._current(epoch):
                    self.played_chunks += 1
                if self._current(epoch):
                    self.emit("speech_finished", None)
            except Exception as error:
                if self._current(epoch):
                    from .tapo_talk import TalkError
                    from .cloud_voice import VoiceError
                    self.emit("tts_error", str(error) if isinstance(error, (TalkError, VoiceError)) else "Could not play speech through the selected audio output.")
            finally:
                reply.abort()
                with self._pending_lock:
                    self._pending_speech -= 1
                    if self._pending_speech == 0:
                        self.controls.put(("output", False))
                        if self._followup_pending:
                            self.controls.put(("followup", self._followup_pending))
                            self._followup_pending = 0
                        self.emit("output_idle", None)
        with self._provider_lock:
            voice = self._cloud_voice
        if voice is not None:
            voice.close()

    def _play_chunks(self, chunks, stopped):
        from .tapo_talk import TalkError
        with self._playback_lock:
            destination, speaker = self.output, self._camera_speaker
        if stopped():
            return False
        self.emit("speaking", None)
        if destination == "camera":
            if speaker is None:
                raise TalkError("Camera speaker is not connected. Use Keys to connect it.")
            return speaker.play_stream(chunks, stopped)
        import sounddevice as sd
        stream = None
        source = iter(chunks)
        try:
            for chunk in source:
                if stopped():
                    return False
                if stream is None:
                    stream = sd.OutputStream(samplerate=chunk.sample_rate, channels=1,
                                             dtype="float32", blocksize=480)
                    with self._playback_lock:
                        if stopped():
                            return False
                        self._computer_stream = stream
                        stream.start()
                for offset in range(0, len(chunk.audio_float_array), 480):
                    if stopped():
                        return False
                    stream.write(chunk.audio_float_array[offset:offset + 480])
            if stream is not None and not stopped():
                stream.stop()
            return stream is not None and not stopped()
        finally:
            with self._playback_lock:
                self._computer_stream = None
            if stream is not None:
                stream.close()
            if hasattr(source, "close"):
                source.close()

    def start_listening(self):
        if not self.recognition_ready.is_set() or self.closed.is_set():
            return False
        self.stop_speaking()
        identifier = self._next_identifier()
        self.controls.put(("start", identifier))
        self.listening.set()
        return identifier

    def feed(self, pcm):
        if self.microphone_active and not self.closed.is_set():
            try:
                self.audio.put_nowait((time.monotonic(), pcm))
            except queue.Full:
                self._clear_audio()
                self.controls.put(("discontinuity", None))
                self.listening.clear()

    def finish_listening(self):
        self.listening.clear()
        self.controls.put(("finish", None))

    def cancel_listening(self):
        self.listening.clear()
        self._clear_audio()
        self.controls.put(("cancel", None))

    def _clear_audio(self):
        while True:
            try:
                self.audio.get_nowait()
            except queue.Empty:
                return

    def _recognition_loop(self):
        try:
            from vosk import KaldiRecognizer
            from .handsfree import SileroVAD, VoiceEngine, make_wake_recognizer
            model = load_recognizer_model()
            wake = make_wake_recognizer(model)
            recognizer = KaldiRecognizer(model, 16000)
            engine = VoiceEngine(wake, recognizer, SileroVAD(MODELS / "silero_vad.onnx"),
                                 self._voice_event, self._next_identifier)
            self.recognition_ready.set()
            self.emit("recognition_ready", None)
        except Exception:
            self.emit("recognition_error", "Local recognition could not start. Run python -m tools.setup_models to install its model.")
            return
        while not self.closed.is_set():
            while True:
                try:
                    kind, value = self.controls.get_nowait()
                except queue.Empty:
                    break
                if kind == "enable":
                    engine.enable(value)
                elif kind == "online":
                    engine.set_online(value)
                elif kind == "transcription_key":
                    from .flux import FluxRecognizer
                    if engine.identifier is not None:
                        self._voice_event("listen_cancelled", engine.identifier)
                    engine.cancel()
                    engine.recognizer.Reset()
                    engine.recognizer = FluxRecognizer(value) if value else KaldiRecognizer(model, 16000)
                    self.recognition_provider = "Deepgram Flux" if value else "Vosk local"
                    self.emit("recognizer_changed", self.recognition_provider)
                    value = None
                elif kind == "busy":
                    engine.set_busy(value)
                elif kind == "output":
                    engine.set_output(value, time.monotonic())
                elif kind == "followup":
                    engine.open_followup(value, time.monotonic())
                elif kind == "close_followup":
                    engine.followup_until = 0.0
                    if engine.mode == "followup":
                        engine.idle()
                elif kind == "start":
                    engine.start_manual(value)
                elif kind == "finish":
                    engine.finish()
                elif kind in {"cancel", "discontinuity"}:
                    engine.discontinuity() if kind == "discontinuity" else engine.cancel()
            try:
                received, pcm = self.audio.get(timeout=0.04)
            except queue.Empty:
                engine.tick(time.monotonic())
                continue
            try:
                engine.process(pcm, received, time.monotonic())
            except Exception:
                engine.discontinuity()
                self.emit("recognition_error", "Microphone recognition restarted. Please say Jarvis and try again.")
        engine.recognizer.Reset()

    def close(self):
        self.closed.set()
        self.listening.clear()
        self.hands_free.clear()
        self.stop_speaking()
        with self._provider_lock:
            voice = self._cloud_voice
        if voice is not None:
            voice.close()
