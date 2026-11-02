"""Check hands-free readiness independently of speaker authentication."""

import threading
from types import SimpleNamespace
import unittest

from robot.conversation import Conversation
from robot.handsfree import VoiceEngine
from .support import Activity, Decoder


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class HandsFreeReadinessTests(unittest.TestCase):
    def make_conversation(self):
        speech = SimpleNamespace(recognition_ready=threading.Event(), tts_ready=threading.Event(),
                                 _camera_speaker=None, output='computer', enabled=False)
        speech.recognition_ready.set()
        speech.tts_ready.set()
        speech.set_hands_free = lambda enabled: setattr(speech, 'enabled', enabled)
        return SimpleNamespace(client=object(), microphone_available=True, speech=speech,
                               output_chosen=False, hands_free_enabled=Value(True), voice_status=Value(''))

    def test_working_audio_enables_wake_words_without_camera_speaker_login(self):
        conversation = self.make_conversation()
        Conversation.sync_hands_free(conversation)
        self.assertTrue(conversation.speech.enabled)
        self.assertIsNone(conversation.speech._camera_speaker)
        self.assertFalse(conversation.output_chosen)

    def test_missing_microphone_key_or_models_does_not_enable_listening(self):
        for missing in ('microphone', 'key', 'recognition', 'tts'):
            with self.subTest(missing=missing):
                conversation = self.make_conversation()
                if missing == 'microphone': conversation.microphone_available = False
                elif missing == 'key': conversation.client = None
                elif missing == 'recognition': conversation.speech.recognition_ready.clear()
                else: conversation.speech.tts_ready.clear()
                Conversation.sync_hands_free(conversation)
                self.assertFalse(conversation.speech.enabled)

    def test_user_can_still_switch_off_wake_listening(self):
        conversation = self.make_conversation()
        conversation.hands_free_enabled.set(False)
        Conversation.sync_hands_free(conversation)
        self.assertFalse(conversation.speech.enabled)
        self.assertEqual(conversation.voice_status.get(), 'Wake word listening is off')

    def test_followup_remains_available_without_speaker_authentication(self):
        conversation = self.make_conversation()
        events = []
        engine = VoiceEngine(Decoder(), Decoder(), Activity(),
                             lambda kind,value: events.append((kind,value)), lambda: 1)
        engine.set_online(True)
        conversation.speech.set_hands_free = engine.enable
        Conversation.sync_hands_free(conversation)
        engine.open_followup(15, 100)
        self.assertEqual(engine.mode, 'followup')
        for now in (100.1, 100.2):
            engine.process(b'\0' * 3200, now, now)
        self.assertIn(('followup_started', 1), events)
        self.assertIsNone(conversation.speech._camera_speaker)


if __name__ == '__main__':
    unittest.main()
