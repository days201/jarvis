"""Check streamed speech framing, turn completion, cancellation, and safe errors."""

import json
import queue
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from robot.cloud_voice import FluxVoice, VoiceError, VOICE_MODEL
from robot.speech_stream import ReplyText


class VoiceSocket:
    def __init__(self):
        self.messages = []
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self, timeout):
        if self.messages:
            return self.messages.pop(0)
        raise TimeoutError

    def close(self):
        self.closed = True


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.socket = VoiceSocket()
        self.connections = []
        def connect(url, **kwargs):
            self.connections.append((url, kwargs))
            return self.socket
        self.voice = FluxVoice("test-voice-key-not-a-real-secret", connect)
        self.addCleanup(self.voice.close)

    def complete(self, audio):
        self.socket.messages.extend([audio, json.dumps({"type": "SpeechMetadata"})])

    def test_binary_frames_keep_sample_alignment_and_use_british_voice(self):
        pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        self.socket.messages.extend([json.dumps({"type": "Connected"}), pcm[:1], pcm[1:5], pcm[5:],
                                     json.dumps({"type": "Flushed"}), json.dumps({"type": "SpeechMetadata"})])
        chunks = list(self.voice.synthesize("Hello."))
        np.testing.assert_allclose(np.concatenate([chunk.audio_float_array for chunk in chunks]),
                                   np.array([-1, 0, 32767 / 32768]))
        self.assertTrue(all(chunk.sample_rate == 24000 for chunk in chunks))
        self.assertIn(VOICE_MODEL, self.connections[0][0])
        self.assertEqual([m["type"] for m in self.socket.sent], ["Speak", "Flush"])
        self.assertEqual(self.socket.sent[0]["text"], "Hello. ")

    def test_completed_turn_keeps_connection_for_the_next_phrase(self):
        for text in ["Hello.", "How can I help?"]:
            self.complete(b"\0" * 480)
            self.assertTrue(list(self.voice.synthesize(text)))
        self.assertEqual(len(self.connections), 1)
        self.assertFalse(self.socket.closed)

    def test_cancellation_stops_audio_and_discards_the_connection(self):
        self.socket.messages.extend([b"\0" * 480, b"\0" * 480, json.dumps({"type": "SpeechMetadata"})])
        stopped = threading.Event()
        audio = self.voice.synthesize("Do not finish this sentence.", stopped.is_set)
        next(audio)
        stopped.set()
        self.assertEqual(list(audio), [])
        self.assertTrue(self.socket.closed)

    def test_cancellation_before_start_does_not_send_text(self):
        self.assertEqual(list(self.voice.synthesize("Private text", lambda: True)), [])
        self.assertFalse(self.connections)

    def test_closing_playback_generator_cancels_synthesis(self):
        self.socket.messages.append(b"\0" * 480)
        audio = self.voice.synthesize("Hello.")
        next(audio)
        audio.close()
        self.assertTrue(self.socket.closed)
        self.assertFalse(self.voice._active.is_set())

    def test_server_error_does_not_copy_private_details(self):
        self.socket.messages.append(json.dumps({"type": "Error", "description": "private key and upstream details"}))
        with self.assertRaises(VoiceError) as raised:
            list(self.voice.synthesize("Hello."))
        self.assertNotIn("private", str(raised.exception))
        self.assertTrue(self.socket.closed)

    def test_authentication_error_hides_headers_and_requests_key_check(self):
        def fail(*args, **kwargs):
            error = RuntimeError("private Authorization header")
            error.response = SimpleNamespace(status_code=403)
            raise error
        self.voice._connector = fail
        with self.assertRaisesRegex(VoiceError, "key under Keys") as raised:
            list(self.voice.synthesize("Hello."))
        self.assertNotIn("Authorization", str(raised.exception))

    def test_incomplete_sample_fails_before_reusing_connection(self):
        self.complete(b"\0")
        with self.assertRaisesRegex(VoiceError, "incomplete audio sample"):
            list(self.voice.synthesize("Hello."))
        self.assertTrue(self.socket.closed)

    def test_live_text_keeps_one_turn_and_can_arrive_during_playback(self):
        incoming = queue.Queue()
        first_sent, next_sent = threading.Event(), threading.Event()
        def send(message):
            event = json.loads(message)
            self.socket.sent.append(event)
            if event['type'] == 'Speak':
                incoming.put(b'\0' * 480)
                (first_sent if len(self.socket.sent) == 1 else next_sent).set()
            elif event['type'] == 'Flush':
                incoming.put(json.dumps({'type': 'SpeechMetadata'}))
        self.socket.send = send
        def receive(timeout):
            try:
                return incoming.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError from None
        self.socket.recv = receive
        text = ReplyText(lambda: False)
        text.write('First sentence.')
        audio = self.voice.synthesize_stream(text)
        self.addCleanup(audio.close)
        self.assertTrue(next(audio).audio_float_array.size)
        self.assertTrue(first_sent.is_set())
        self.assertNotIn('Flush', [m['type'] for m in self.socket.sent])
        # Playback has not requested more audio, but the next sentence must
        # already reach synthesis rather than waiting for playback to finish.
        text.write('Second sentence.')
        self.assertTrue(next_sent.wait(1))
        text.finish()
        self.assertTrue(list(audio))
        self.assertEqual([m['type'] for m in self.socket.sent], ['Speak', 'Speak', 'Flush'])
        self.assertEqual(''.join(m.get('text', '') for m in self.socket.sent),
                         'First sentence. Second sentence. ')

    def test_closing_audio_cancels_a_sender_waiting_for_more_text(self):
        text = ReplyText(lambda: False)
        text.write('Hello.')
        self.socket.messages.append(b'\0' * 480)
        audio = self.voice.synthesize_stream(text)
        next(audio)
        audio.close()
        self.assertFalse(self.voice._active.is_set())
        self.assertTrue(self.socket.closed)
        self.assertNotIn('Flush', [m['type'] for m in self.socket.sent])


if __name__ == "__main__":
    unittest.main()
