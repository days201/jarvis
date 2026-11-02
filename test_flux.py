"""Check cloud turn boundaries, cancellation, and the local wake privacy gate."""

import itertools
import json
import queue
import threading
import time
from types import SimpleNamespace
import unittest

from flux import CHUNK_BYTES, FluxRecognizer, connection_error
from handsfree import VoiceEngine
from test_handsfree import Activity, Decoder


class Socket:
    def __init__(self):
        self.sent = []
        self.received = queue.Queue()
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed.set()

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout):
        try:
            return self.received.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None


def wait_for(predicate):
    deadline = time.monotonic() + 1
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    if not predicate():
        raise AssertionError("Cloud worker did not reach the expected state.")


class FluxTests(unittest.TestCase):
    def setUp(self):
        self.socket = Socket()
        self.connections = []

        def connect(url, **kwargs):
            self.connections.append((url, kwargs))
            return self.socket

        self.recognizer = FluxRecognizer("test-key-for-offline-checks", connector=connect)
        self.addCleanup(self.recognizer.Reset)
        self.events = []
        self.wake, self.vad = Decoder(), Activity()
        identifiers = itertools.count(1)
        self.engine = VoiceEngine(self.wake, self.recognizer, self.vad,
                                  lambda kind, value: self.events.append((kind, value)),
                                  lambda: next(identifiers))
        self.engine.enable(True)
        self.engine.set_online(True)
        self.now = time.monotonic()

    def feed(self, seconds=0.1):
        for _ in range(round(seconds / 0.1)):
            self.now += 0.1
            self.engine.process(b"\0" * 3200, self.now, self.now)

    def message(self, event, text=""):
        self.socket.received.put(json.dumps({"type": "TurnInfo", "event": event, "transcript": text}))

    def test_idle_audio_and_speaker_echo_never_open_cloud_connection(self):
        self.wake.text = "[unk]"
        self.feed(3)
        self.engine.set_output(True, self.now)
        self.wake.text = "hey jarvis"
        self.feed(2)
        self.assertFalse(self.connections)

    def test_flux_finishes_question_after_model_event_not_local_silence(self):
        self.wake.text = "hey jarvis"
        self.feed(0.5)
        wait_for(lambda: bool(self.connections))
        self.message("Update", "Jarvis, could you explain")
        wait_for(lambda: self.recognizer._turn.incoming.qsize() > 0)
        self.engine.tick(self.now)
        self.vad.voiced = False
        self.feed(1.1)
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))
        self.assertEqual(self.engine.mode, "command")
        self.message("EndOfTurn", "Jarvis, could you explain how rain forms?")
        wait_for(lambda: self.recognizer._turn.incoming.qsize() > 0)
        self.engine.tick(self.now)
        self.assertIn(("recognized", (1, "could you explain how rain forms?")), self.events)
        self.assertEqual(self.engine.mode, "waiting")

    def test_reset_discards_late_cloud_results_and_closes_connection(self):
        self.engine.start_manual(12)
        self.feed(0.2)
        wait_for(lambda: bool(self.connections))
        self.engine.enable(False)
        self.message("EndOfTurn", "this cancelled question must not be sent")
        wait_for(self.socket.closed.is_set)
        self.engine.tick(self.now)
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))
        self.assertIsNone(self.recognizer._turn)

    def test_manual_finish_flushes_audio_before_force_end_and_waits_for_final(self):
        self.engine.start_manual(9)
        self.feed(0.1)
        self.engine.finish()
        wait_for(lambda: any(isinstance(item, str) and "ForceEndTurn" in item for item in self.socket.sent))
        audio = [item for item in self.socket.sent if isinstance(item, bytes)]
        self.assertEqual(len(audio), 2)
        self.assertTrue(all(len(item) == CHUNK_BYTES for item in audio))
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))
        self.message("EndOfTurn", "What time is it?")
        wait_for(lambda: self.recognizer._turn.incoming.qsize() > 0)
        self.engine.tick(self.now)
        self.assertIn(("recognized", (9, "What time is it?")), self.events)

    def test_muting_followup_cancels_cloud_and_returns_to_local_wake(self):
        self.engine.open_followup(15, self.now)
        self.feed(0.3)
        wait_for(lambda: bool(self.connections))
        self.engine.enable(False)
        wait_for(self.socket.closed.is_set)
        self.assertEqual(self.engine.mode, "off")
        self.assertIsNone(self.recognizer._turn)

    def test_network_error_cancels_turn_and_recovers_without_guessing_text(self):
        self.engine.start_manual(3)
        self.feed(0.2)
        self.socket.received.put(json.dumps({"type": "Error", "description": "private server details"}))
        wait_for(lambda: self.recognizer._turn.incoming.qsize() > 0)
        self.engine.tick(self.now)
        self.assertIn(("listen_cancelled", 3), self.events)
        self.assertTrue(any(kind == "recognition_error" for kind, _ in self.events))
        self.assertNotIn("private server details", str(self.events))
        self.assertEqual(self.engine.mode, "wake")

    def test_exception_details_never_expose_credentials(self):
        error = RuntimeError("secret key in an upstream header")
        error.response = SimpleNamespace(status_code=401)
        self.assertIn("rejected the key", connection_error(error))
        self.assertNotIn("secret", connection_error(error))
        self.assertNotIn("secret", connection_error(RuntimeError("secret")))

    def test_standalone_wake_acknowledges_and_preserves_question_identifier(self):
        self.wake.text = "jarvis"
        self.feed(0.5)
        identifier = self.engine.identifier
        self.vad.voiced = False
        self.feed(1.2)
        self.assertEqual(sum(kind == "acknowledge" for kind, _ in self.events), 1)
        self.engine.set_output(True, self.now)
        self.engine.set_output(False, self.now)
        self.assertEqual(self.engine.identifier, identifier)
        self.assertEqual(self.engine.mode, "command")
        self.assertIsNone(self.recognizer._turn)

    def test_audio_queue_overflow_fails_without_uploading_the_partial_question(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def delayed_connection(*args, **kwargs):
            release.wait(1)
            return self.socket

        self.recognizer._connector = delayed_connection
        self.recognizer.feed(b"\0" * (CHUNK_BYTES * 102))
        self.assertTrue(any(kind == "error" for kind, _ in self.recognizer.poll()))
        release.set()
        wait_for(self.socket.closed.is_set)
        self.assertFalse(self.socket.sent)

    def test_disconnect_discards_pending_final_before_delivering_it(self):
        self.engine.start_manual(7)
        self.feed(0.1)
        self.message("EndOfTurn", "this must not arrive after a camera disconnect")
        wait_for(lambda: self.recognizer._turn.incoming.qsize() > 0)
        self.engine.tick(self.now + 4)
        self.assertIn(("listen_cancelled", 7), self.events)
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))
        self.assertEqual(self.engine.mode, "off")


if __name__ == "__main__":
    unittest.main()
