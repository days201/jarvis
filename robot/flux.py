"""Stream active voice turns to Deepgram Flux without blocking camera audio."""

from dataclasses import dataclass, field
import json
import logging
import queue
import threading
import time
from urllib.parse import urlencode


SAMPLE_RATE = 16000
CHUNK_BYTES = 2560  # 80 ms of signed 16-bit mono audio.
URL = "wss://api.deepgram.com/v2/listen?" + urlencode({
    "model": "flux-general-en", "encoding": "linear16", "sample_rate": SAMPLE_RATE,
    "eot_threshold": 0.85, "eot_timeout_ms": 5000,
})


def connection_error(error):
    """Expose useful status codes without copying headers or server bodies."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in {401, 403}:
        return "Deepgram rejected the key. Open Keys and check its transcription access."
    if status == 402:
        return "Deepgram needs account credit before it can transcribe speech."
    if status == 429:
        return "Deepgram is busy or rate limited. Please wait and try again."
    return "Cloud transcription disconnected. Check the connection and say Jarvis to try again."


@dataclass
class _Turn:
    outgoing: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=100))
    incoming: queue.Queue = field(default_factory=queue.Queue)
    stopped: threading.Event = field(default_factory=threading.Event)
    pending: bytearray = field(default_factory=bytearray)
    finishing: bool = False
    finish_deadline: float = 0.0


class FluxRecognizer:
    """Open one connection per question. Reset discards that turn and its results."""

    remote_turns = True

    def __init__(self, api_key, connector=None):
        if not isinstance(api_key, str) or len(api_key.strip()) < 20 or any(c.isspace() for c in api_key.strip()):
            raise ValueError("Enter a valid Deepgram API key.")
        self._api_key = api_key.strip()
        self._connector = connector
        self._turn = None

    def Reset(self):
        turn, self._turn = self._turn, None
        if turn is not None:
            turn.stopped.set()
            turn.pending.clear()
            while True:
                try:
                    turn.outgoing.get_nowait()
                except queue.Empty:
                    break

    @property
    def finishing(self):
        return bool(self._turn and self._turn.finishing)

    def _put(self, turn, payload):
        try:
            turn.outgoing.put_nowait(payload)
        except queue.Full:
            turn.stopped.set()
            turn.incoming.put(("error", "Cloud transcription fell behind. Please say Jarvis and repeat the question."))

    def feed(self, pcm):
        if not pcm:
            return
        if self._turn is None:
            self._turn = _Turn()
            threading.Thread(target=self._run, args=(self._turn,), daemon=True,
                             name="jarvis-flux").start()
        turn = self._turn
        if turn.stopped.is_set() or turn.finishing:
            return
        turn.pending.extend(pcm)
        while len(turn.pending) >= CHUNK_BYTES:
            chunk = bytes(turn.pending[:CHUNK_BYTES])
            del turn.pending[:CHUNK_BYTES]
            self._put(turn, chunk)

    def finish(self):
        turn = self._turn
        if turn is None:
            return False
        if not turn.finishing:
            if turn.pending:
                self._put(turn, bytes(turn.pending).ljust(CHUNK_BYTES, b"\0"))
                turn.pending.clear()
            self._put(turn, json.dumps({"type": "ForceEndTurn"}))
            turn.finishing = True
            turn.finish_deadline = time.monotonic() + 5
        return True

    def poll(self):
        turn = self._turn
        if turn is None:
            return []
        events = []
        while True:
            try:
                events.append(turn.incoming.get_nowait())
            except queue.Empty:
                break
        if turn.finishing and time.monotonic() >= turn.finish_deadline and not any(kind in {"final", "error"} for kind, _ in events):
            turn.stopped.set()
            events.append(("error", "Cloud transcription took too long. Please say Jarvis and try again."))
        return events

    def _run(self, turn):
        logger = logging.getLogger("jarvis.flux.transport")
        logger.disabled = True
        try:
            if self._connector is None:
                from websockets.sync.client import connect
                connector = connect
            else:
                connector = self._connector
            with connector(URL, additional_headers={"Authorization": "Token " + self._api_key},
                           open_timeout=5, close_timeout=0.2, ping_interval=10, ping_timeout=5,
                           compression=None, max_size=131072, logger=logger) as connection:
                while not turn.stopped.is_set():
                    for _ in range(4):
                        if turn.stopped.is_set():
                            break
                        try:
                            payload = turn.outgoing.get_nowait()
                        except queue.Empty:
                            break
                        connection.send(payload)
                    if turn.stopped.is_set():
                        break
                    try:
                        raw = connection.recv(timeout=0.02)
                    except TimeoutError:
                        continue
                    message = json.loads(raw)
                    if message.get("type") == "Error":
                        turn.incoming.put(("error", "Deepgram could not transcribe this question. Please try again."))
                        break
                    if message.get("type") != "TurnInfo":
                        continue
                    event = message.get("event")
                    text = str(message.get("transcript") or "").strip()
                    if event == "StartOfTurn":
                        turn.incoming.put(("speech_started", ""))
                    if event == "EndOfTurn":
                        turn.incoming.put(("final", text))
                        break
                    if text:
                        turn.incoming.put(("partial", text))
                if not turn.stopped.is_set():
                    connection.send(json.dumps({"type": "CloseStream"}))
        except Exception as error:
            if not turn.stopped.is_set():
                turn.incoming.put(("error", connection_error(error)))
        finally:
            turn.stopped.set()
