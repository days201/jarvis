"""Bounded text and audio queues for a single spoken reply."""

import queue
import threading

import numpy as np

from cloud_voice import AudioChunk, SAMPLE_RATE


class ReplyText:
    def __init__(self, stopped):
        self._stopped = stopped
        self._queue = queue.Queue(maxsize=32)
        self._finished = threading.Event()
        self._aborted = threading.Event()

    def write(self, text):
        if not text or self._finished.is_set():
            return
        while not self._stopped() and not self._aborted.is_set():
            try:
                self._queue.put(text, timeout=0.05)
                return
            except queue.Full:
                continue

    def finish(self):
        self._finished.set()

    def abort(self):
        self._aborted.set()

    def __call__(self, stopped):
        while not stopped() and not self._stopped() and not self._aborted.is_set():
            try:
                text = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._finished.is_set():
                    return
                continue
            yield text


def buffered_audio(make_chunks, stopped):
    """Receive up to four seconds ahead of playback in memory.

    During a tool wait, silence keeps the camera's presentation clock running.
    Cancellation stops the receiver even if the output stops consuming chunks.
    """
    audio = queue.Queue(maxsize=40)
    abort = threading.Event()
    done = threading.Event()
    errors = []
    halted = lambda: abort.is_set() or stopped()

    def receive():
        source = None
        try:
            source = iter(make_chunks(halted))
            for chunk in source:
                size = max(1, chunk.sample_rate // 10)
                for offset in range(0, len(chunk.audio_float_array), size):
                    part = AudioChunk(chunk.audio_float_array[offset:offset + size], chunk.sample_rate)
                    while not halted():
                        try:
                            audio.put(part, timeout=0.05)
                            break
                        except queue.Full:
                            continue
                    if halted():
                        return
        except Exception as error:
            if not halted():
                errors.append(error)
        finally:
            try:
                if hasattr(source, "close"):
                    source.close()
            finally:
                done.set()

    receiver = threading.Thread(target=receive, daemon=True, name="jarvis-voice-audio")
    receiver.start()
    started = False
    try:
        while not halted():
            if errors:
                raise errors[0]
            try:
                chunk = audio.get(timeout=0.02 if started else 0.05)
            except queue.Empty:
                if done.is_set():
                    break
                if started:
                    yield AudioChunk(np.zeros(SAMPLE_RATE // 50, dtype=np.float32))
                continue
            started = True
            yield chunk
    finally:
        abort.set()
        receiver.join(timeout=0.5)
