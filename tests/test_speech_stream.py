"""Check speech continuity, bounded buffering, and playback cancellation."""

import queue
import threading
import unittest
from unittest.mock import patch

import numpy as np

from robot.cloud_voice import AudioChunk, VoiceError
from robot.speech import Speech
from robot.speech_stream import ReplyText, buffered_audio


class StreamingTests(unittest.TestCase):
    def test_audio_receiver_runs_ahead_and_stops_when_playback_closes(self):
        advanced, stopped = threading.Event(), threading.Event()
        def source(halted):
            try:
                yield AudioChunk(np.ones(2400, dtype=np.float32))
                advanced.set()
                while not halted():
                    threading.Event().wait(0.01)
            finally:
                stopped.set()
        stream = buffered_audio(source, lambda: False)
        next(stream)
        self.assertTrue(advanced.wait(1))
        stream.close()
        self.assertTrue(stopped.wait(1))

    def test_tool_wait_keeps_camera_clock_running_with_silence(self):
        release = threading.Event()
        def source(halted):
            yield AudioChunk(np.ones(2400, dtype=np.float32))
            while not release.wait(0.01) and not halted():
                pass
            if not halted():
                yield AudioChunk(np.ones(2400, dtype=np.float32) * 0.5)
        stream = buffered_audio(source, lambda: False)
        self.addCleanup(stream.close)
        self.assertTrue(np.all(next(stream).audio_float_array == 1))
        self.assertTrue(np.all(next(stream).audio_float_array == 0))
        release.set()
        self.assertTrue(any(np.any(chunk.audio_float_array == 0.5) for chunk in stream))

    def test_cancel_releases_a_writer_with_a_full_text_queue(self):
        cancelled, done = threading.Event(), threading.Event()
        text = ReplyText(cancelled.is_set)
        def fill():
            for _ in range(100):
                text.write('A sentence.')
            done.set()
        writer = threading.Thread(target=fill, daemon=True)
        writer.start()
        cancelled.set()
        self.assertTrue(done.wait(1))
        self.assertEqual(list(text(cancelled.is_set)), [])

    def test_voice_error_reaches_playback_without_waiting_for_all_audio(self):
        def source(halted):
            raise VoiceError('Voice failed.')
            yield
        with self.assertRaisesRegex(VoiceError, 'Voice failed'):
            list(buffered_audio(source, lambda: False))

    def test_voice_start_failure_does_not_leave_playback_waiting(self):
        def fail(halted):
            raise VoiceError('Voice could not start.')
        with self.assertRaisesRegex(VoiceError, 'Voice could not start'):
            list(buffered_audio(fail, lambda: False))

    def test_reply_uses_one_speaker_session_and_followup_waits_for_playback(self):
        held, release, second_received, idle = (threading.Event() for _ in range(4))
        calls = {'voice': 0, 'speaker': 0}
        events = []
        class Cloud:
            def synthesize_stream(self, text, stopped):
                calls['voice'] += 1
                for phrase in text(stopped):
                    if phrase == 'Second sentence.':
                        second_received.set()
                    yield AudioChunk(np.ones(2400, dtype=np.float32))
            def close(self):
                pass
            def cancel_active(self):
                pass
        class Speaker:
            def play_stream(self, chunks, stopped):
                calls['speaker'] += 1
                for chunk in chunks:
                    held.set()
                    while not release.wait(0.01) and not stopped():
                        pass
                    if stopped():
                        return False
                return True
            def close(self):
                release.set()
        def emit(kind, value):
            events.append(kind)
            if kind == 'output_idle':
                idle.set()
        with patch('robot.speech.load_tts'), patch.object(Speech, '_recognition_loop'):
            speech = Speech(emit)
            self.addCleanup(speech.close)
            self.assertTrue(speech.tts_ready.wait(1))
            speech._cloud_voice = Cloud()
            speech._camera_speaker = Speaker()
            speech.output = 'camera'
            reply = speech.begin_reply()
            reply.write('First sentence.')
            self.assertTrue(held.wait(1))
            reply.write('Second sentence.')
            self.assertTrue(second_received.wait(1))
            reply.finish()
            speech.open_followup(15)
            self.assertEqual(speech._pending_speech, 1)
            controls = []
            while not speech.controls.empty():
                controls.append(speech.controls.get_nowait())
            self.assertNotIn(('followup', 15), controls)
            release.set()
            self.assertTrue(idle.wait(1))
            self.assertEqual(calls, {'voice': 1, 'speaker': 1})
            self.assertEqual(speech._pending_speech, 0)
            self.assertEqual(speech.controls.get_nowait(), ('output', False))
            self.assertEqual(speech.controls.get_nowait(), ('followup', 15))


if __name__ == '__main__':
    unittest.main()
