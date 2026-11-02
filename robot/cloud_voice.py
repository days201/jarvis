"""Deepgram Flux speech synthesis with the British male Colin voice."""

from dataclasses import dataclass
import json
import logging
import threading
import time
from urllib.parse import urlencode

import numpy as np


VOICE_MODEL = "flux-colin-en"
SAMPLE_RATE = 24000
VOICE_LABEL = "Deepgram Colin"
VOICE_URL = "wss://api.deepgram.com/v2/speak?" + urlencode({
    "model": VOICE_MODEL, "encoding": "linear16", "sample_rate": SAMPLE_RATE,
})


class VoiceError(RuntimeError):
    pass


@dataclass
class AudioChunk:
    audio_float_array: np.ndarray
    sample_rate: int = SAMPLE_RATE


def voice_error(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in {401, 403}:
        return "Deepgram voice access was rejected. Check the key under Keys."
    if status == 402:
        return "Deepgram needs account credit to generate speech."
    if status == 429:
        return "Deepgram voice is rate limited. Please wait and try again."
    return "Deepgram voice disconnected. Please try the question again."


class FluxVoice:
    """The speech worker owns synthesis. Other threads may cancel or close it."""

    def __init__(self, api_key, connector=None):
        if not isinstance(api_key, str) or len(api_key.strip()) < 20 or any(c.isspace() for c in api_key.strip()):
            raise ValueError("Enter a valid Deepgram API key.")
        self._key = api_key.strip()
        self._connector = connector
        self._lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._connection = None
        self._active = threading.Event()
        self._generation = 0

    def close(self):
        with self._lock:
            self._generation += 1
            connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def cancel_active(self):
        if self._active.is_set():
            self.close()

    def _connect(self, stopped):
        with self._lock:
            connection = self._connection
            generation = self._generation
        if connection is not None:
            return connection
        if self._connector is None:
            from websockets.sync.client import connect
            connector = connect
        else:
            connector = self._connector
        logger = logging.getLogger("jarvis.voice.transport")
        logger.disabled = True
        connection = connector(VOICE_URL, additional_headers={"Authorization": "Token " + self._key},
                               open_timeout=5, close_timeout=0.2, ping_interval=20, ping_timeout=5,
                               compression=None, max_queue=16, max_size=262144, logger=logger)
        with self._lock:
            cancelled = stopped() or generation != self._generation
            if not cancelled:
                self._connection = connection
        if cancelled:
            connection.close()
            return None
        return connection

    def synthesize(self, text, stopped=lambda: False):
        yield from self.synthesize_stream((text,), stopped)

    def synthesize_stream(self, texts, stopped=lambda: False):
        """Send arriving text while receiving audio, then flush once per reply.

        A callable text source receives the cancellation predicate so it can
        stop waiting for the model when playback ends.
        """
        while not stopped():
            if self._synthesis_lock.acquire(timeout=0.05):
                break
        else:
            return
        try:
            yield from self._synthesize_stream(texts, stopped)
        finally:
            self._synthesis_lock.release()

    def _synthesize_stream(self, texts, stopped):
        abort = threading.Event()
        sending_done = threading.Event()
        flushed = threading.Event()
        errors = []
        sender = None
        halted = lambda: abort.is_set() or stopped()
        source = iter(texts(halted) if callable(texts) else texts)
        text = next(source, "")
        if stopped() or not text.strip():
            return
        self._active.set()
        completed = False
        try:
            connection = self._connect(stopped)
            if connection is None or stopped():
                return

            def send_text():
                total = 0
                try:
                    from itertools import chain
                    for part in chain((text,), source):
                        if halted():
                            return
                        if not part.strip():
                            continue
                        total += len(part)
                        if total > 12000 or len(part) > 3000:
                            raise VoiceError("This speech reply is too long. Please ask for a shorter reply.")
                        connection.send(json.dumps({"type": "Speak", "text": part.strip() + " "}))
                    if not halted():
                        flushed.set()
                        connection.send(json.dumps({"type": "Flush"}))
                except Exception as error:
                    if not halted():
                        errors.append(error)
                finally:
                    sending_done.set()

            sender = threading.Thread(target=send_text, daemon=True, name="jarvis-voice-text")
            sender.start()
            deadline = time.monotonic() + 8
            pending = b""
            while not halted():
                if errors:
                    raise errors[0]
                # Tools may still be moving the camera or waiting for a fresh
                # frame. The short audio timeout starts after all text is sent.
                if sending_done.is_set() and time.monotonic() > deadline:
                    raise VoiceError("Deepgram voice took too long to respond. Please try again.")
                if not sending_done.is_set():
                    deadline = time.monotonic() + 8
                try:
                    message = connection.recv(timeout=0.1)
                except TimeoutError:
                    continue
                deadline = time.monotonic() + 8
                if isinstance(message, bytes):
                    pending += message
                    aligned = len(pending) // 2 * 2
                    if aligned:
                        audio = np.frombuffer(pending[:aligned], dtype="<i2").astype(np.float32) / 32768
                        pending = pending[aligned:]
                        yield AudioChunk(audio)
                    continue
                event = json.loads(message)
                if event.get("type") == "Error":
                    raise VoiceError("Deepgram could not generate this speech. Please try again.")
                if event.get("type") == "SpeechMetadata":
                    if not flushed.is_set():
                        raise VoiceError("Deepgram ended speech before the reply was complete.")
                    if pending:
                        raise VoiceError("Deepgram returned an incomplete audio sample.")
                    completed = True
                    return
        except VoiceError:
            raise
        except Exception as error:
            if not stopped():
                raise VoiceError(voice_error(error)) from None
        finally:
            abort.set()
            if not completed:
                self.close()
            if sender is not None:
                sender.join(timeout=0.3)
            self._active.clear()
