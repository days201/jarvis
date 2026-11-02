"""Check voice turn boundaries, privacy, echo suppression, and recovery."""

import itertools
import json
import unittest

from handsfree import VoiceEngine, remove_wake, wake_prefix


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


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.wake, self.decoder, self.activity = Decoder(), Decoder(), Activity()
        self.events = []
        identifiers = itertools.count(1)
        self.engine = VoiceEngine(self.wake, self.decoder, self.activity,
                                  lambda kind, value: self.events.append((kind, value)),
                                  lambda: next(identifiers))
        self.now = 100.0
        self.engine.set_online(True)
        self.engine.enable(True)

    def feed(self, seconds=0.1):
        for _ in range(round(seconds / 0.1)):
            self.now += 0.1
            self.engine.process(b"\0" * 3200, self.now, self.now)

    def activate(self, name="hey jarvis"):
        self.wake.text = name
        self.feed(0.5)
        self.assertEqual(self.engine.mode, "command")

    def test_both_wake_forms_require_complete_words(self):
        for text in ["jarvis", "hey jarvis", "Hey Jarvis what time is it"]:
            self.assertTrue(wake_prefix(text))
        for text in ["jarvison", "the jarvis", "hey", "travis", "[unk] jarvis"]:
            self.assertFalse(wake_prefix(text))
        self.activate("jarvis")
        self.assertEqual(sum(kind == "wake_detected" for kind, _ in self.events), 1)

    def test_inline_question_keeps_first_words_and_waits_for_silence(self):
        self.activate()
        self.decoder.text = "hey jarvis what is the capital of france"
        self.feed(0.4)
        self.activity.voiced = False
        self.feed(0.4)
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))
        self.activity.voiced = True
        self.feed(0.2)
        self.activity.voiced = False
        self.feed(0.7)
        self.assertIn(("recognized", (1, "what is the capital of france")), self.events)
        self.assertEqual(self.engine.mode, "waiting")
        self.assertFalse(any(kind == "acknowledge" for kind, _ in self.events))

    def test_wake_alone_acknowledges_once_then_times_out_without_a_question(self):
        self.activate()
        self.decoder.text = "hey jarvis"
        self.activity.voiced = False
        self.feed(2)
        self.assertEqual(sum(kind == "acknowledge" for kind, _ in self.events), 1)
        self.feed(5.5)
        self.assertIn(("listen_timeout", 1), self.events)
        self.assertEqual(self.engine.mode, "wake")
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))

    def test_ambient_questions_and_silence_do_not_activate(self):
        self.wake.text = "[unk]"
        self.feed(8)
        self.activity.voiced = False
        self.feed(0.6)
        self.wake.text = "jarvis"
        self.feed(2)
        self.assertFalse(any(kind == "wake_detected" for kind, _ in self.events))

    def test_speaker_audio_and_echo_tail_cannot_wake_the_assistant(self):
        self.engine.set_output(True, self.now)
        self.wake.text = "hey jarvis"
        self.feed(2)
        self.assertEqual(self.wake.calls, 0)
        self.engine.set_output(False, self.now)
        self.wake.text = "jarvis"
        self.feed(0.5)
        self.assertEqual(self.wake.calls, 0)
        self.feed(0.6)
        self.assertEqual(self.engine.mode, "command")

    def test_acknowledgement_preserves_question_session(self):
        self.activate()
        identifier = self.engine.identifier
        self.engine.set_output(True, self.now)
        self.feed(0.8)
        self.engine.set_output(False, self.now)
        self.feed(0.6)
        self.decoder.text = "what is two plus two"
        self.feed(0.4)
        self.activity.voiced = False
        self.feed(0.7)
        self.assertIn(("recognized", (identifier, "what is two plus two")), self.events)

    def test_privacy_switch_discards_an_unfinished_question(self):
        self.activate()
        self.decoder.text = "do not send this"
        self.feed(0.3)
        self.engine.enable(False)
        self.wake.text = "jarvis"
        self.feed(2)
        self.assertEqual(self.engine.mode, "off")
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))

    def test_stale_audio_cancels_the_question_instead_of_sending_it(self):
        self.activate()
        self.decoder.text = "a stale question"
        self.engine.process(b"\0" * 3200, self.now - 2, self.now)
        self.assertIn(("listen_cancelled", 1), self.events)
        self.assertEqual(self.engine.mode, "wake")
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))

    def test_disconnect_cancels_question_and_new_audio_rearms_wake_detection(self):
        self.activate()
        self.engine.tick(self.now + 4)
        self.assertEqual(self.engine.mode, "off")
        self.assertIn(("listen_cancelled", 1), self.events)
        self.now += 4
        self.feed()
        self.assertEqual(self.engine.mode, "wake")
        self.assertTrue(self.engine.online)

    def test_a_completed_question_waits_until_reply_is_ready(self):
        self.activate()
        self.decoder.text = "hey jarvis hello"
        self.engine.finish()
        self.wake.text = "jarvis"
        self.feed(1)
        self.assertEqual(sum(kind == "wake_detected" for kind, _ in self.events), 1)
        self.engine.set_busy(True)
        self.feed(1)
        self.engine.set_busy(False)
        self.assertEqual(self.engine.mode, "wake")

    def test_wake_removal_does_not_remove_the_name_inside_a_question(self):
        self.assertEqual(remove_wake("Hey Jarvis, tell me about Jarvis"), "tell me about Jarvis")
        self.assertEqual(remove_wake("what can Jarvis do"), "what can Jarvis do")

    def test_followup_accepts_speech_without_a_wake_word(self):
        self.engine.open_followup(15, self.now)
        self.feed(0.3)
        self.assertTrue(any(kind == "followup_started" for kind, _ in self.events))
        self.decoder.text = "and what about tomorrow"
        self.feed(0.3)
        self.activity.voiced = False
        self.feed(0.7)
        self.assertIn(("recognized", (1, "and what about tomorrow")), self.events)

    def test_followup_window_expires_without_sending_ambient_speech(self):
        self.engine.open_followup(15, self.now)
        self.activity.voiced = False
        self.feed(15.2)
        self.assertEqual(self.engine.mode, "wake")
        self.assertFalse(any(kind == "recognized" for kind, _ in self.events))

    def test_followup_waits_out_echo_tail_before_listening(self):
        self.engine.set_output(False, self.now)
        self.engine.open_followup(15, self.now)
        self.feed(0.5)
        self.assertFalse(any(kind == "followup_started" for kind, _ in self.events))
        self.feed(0.4)
        self.assertTrue(any(kind == "followup_started" for kind, _ in self.events))

    def test_muting_or_disconnect_removes_followup_window(self):
        self.engine.open_followup(15, self.now)
        self.engine.enable(False)
        self.assertEqual(self.engine.followup_until, 0)
        self.engine.enable(True)
        self.engine.open_followup(15, self.now)
        self.engine.set_online(False)
        self.assertEqual(self.engine.followup_until, 0)


if __name__ == "__main__":
    unittest.main()
