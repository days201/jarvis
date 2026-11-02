"""Check the speaker wire format and cancellation without contacting a camera."""

import struct
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tapo_talk import AudioTransport, TalkError, TapoTalk, digest_authorization, encode_alaw, encode_alaw_stream, mpeg_crc


def transport_payload(packet):
    offset = 4
    if packet[3] & 0x20:
        offset += 1 + packet[4]
    return packet[offset:]


def presentation_timestamp(pes):
    b = pes[9:14]
    return ((b[0] & 14) << 29) | (b[1] << 22) | ((b[2] & 254) << 14) | (b[3] << 7) | (b[4] >> 1)


class TalkTests(unittest.TestCase):
    def test_crc_matches_mpeg2_reference_vector(self):
        self.assertEqual(mpeg_crc(b"123456789"), 0x0376E6E7)

    def test_tables_advertise_tapo_alaw_with_valid_checksums(self):
        data = AudioTransport().header()
        self.assertEqual(len(data), 376)
        for offset, expected_pid in [(0, 0), (188, 0x1000)]:
            packet = data[offset:offset + 188]
            self.assertEqual(packet[0], 0x47)
            self.assertEqual(((packet[1] & 31) << 8) | packet[2], expected_pid)
            payload = transport_payload(packet)
            section = payload[1:]
            length = ((section[1] & 15) << 8) | section[2]
            self.assertEqual(mpeg_crc(section[:length + 3]), 0)
        self.assertEqual(transport_payload(data[188:])[13], 0x90)

    def test_audio_packets_preserve_samples_and_advance_timestamp(self):
        transport = AudioTransport()
        audio = bytes(range(256)) + bytes(range(64))
        first = transport.payload(audio)
        packets = [first[i:i + 188] for i in range(0, len(first), 188)]
        self.assertTrue(all(len(packet) == 188 and packet[0] == 0x47 for packet in packets))
        self.assertEqual([packet[3] & 15 for packet in packets], [0, 1])
        pes = b"".join(transport_payload(packet) for packet in packets)
        self.assertEqual(pes[:4], b"\x00\x00\x01\xc0")
        self.assertEqual(struct.unpack(">H", pes[4:6])[0], len(audio) + 8)
        self.assertEqual(pes[14:], audio)
        self.assertEqual(presentation_timestamp(pes), 0)
        second = transport.payload(audio)
        second_pes = b"".join(transport_payload(second[i:i + 188]) for i in range(0, len(second), 188))
        self.assertEqual(presentation_timestamp(second_pes), 3600)

    def test_conversion_resamples_to_eight_khz_alaw(self):
        encoded = encode_alaw(np.zeros(1600, dtype=np.float32), 16000)
        self.assertEqual(len(encoded), 800)
        self.assertEqual(set(encoded), {0xD5})
        with self.assertRaises(TalkError):
            encode_alaw(np.array([float("nan")], dtype=np.float32), 16000)

    def test_cancelled_audio_does_not_open_a_connection(self):
        speaker = TapoTalk("192.168.2.88", "test-password")
        with patch.object(speaker, "open") as opening:
            self.assertFalse(speaker.play(np.zeros(100), 16000, lambda: True))
            opening.assert_not_called()

    def test_login_selects_camera_hash_and_hides_plaintext(self):
        challenge = 'Digest realm="camera", nonce="nonce", qop="auth", algorithm=MD5'
        legacy = digest_authorization(challenge, "test-password", cnonce="client")
        newer = digest_authorization(challenge + ', encrypt_type="3"', "test-password", cnonce="client")
        self.assertIn('username="admin"', legacy)
        self.assertNotIn("test-password", legacy)
        self.assertNotEqual(legacy, newer)
        self.assertNotIn("test-password", repr(TapoTalk("192.168.2.88", "test-password")))

    def test_stop_during_playback_closes_after_current_audio_packet(self):
        speaker = TapoTalk("192.168.2.88", "test-password")
        stopped = threading.Event()
        sent = []
        def capture(body):
            sent.append(body)
            if len(sent) == 2:
                stopped.set()
        with patch("tapo_talk.encode_alaw", return_value=b"\xd5" * 8000), patch.object(speaker, "open"), patch.object(speaker, "close") as closing, patch.object(speaker, "_audio_part", side_effect=capture):
            self.assertFalse(speaker.play(np.zeros(16000), 16000, stopped.is_set))
            self.assertEqual(len(sent), 2)
            closing.assert_called_once()

    def test_stream_conversion_matches_continuous_audio_across_uneven_chunks(self):
        samples = np.sin(np.arange(4800, dtype=np.float32) * 0.07) * 0.3
        chunks = [SimpleNamespace(audio_float_array=part, sample_rate=24000)
                  for part in np.array_split(samples, [439, 1601, 2700])]
        streamed = b"".join(encode_alaw_stream(chunks, lambda: False))
        self.assertEqual(streamed, encode_alaw(samples, 24000))

    def test_streamed_chunks_share_one_camera_session_and_timestamp_sequence(self):
        speaker = TapoTalk("192.168.2.88", "test-password")
        chunks = [SimpleNamespace(audio_float_array=np.zeros(1200, dtype=np.float32), sample_rate=24000)
                  for _ in range(2)]
        sent = []
        with patch.object(speaker, "open") as opening, patch.object(speaker, "close") as closing, patch.object(speaker, "_audio_part", side_effect=sent.append):
            self.assertTrue(speaker.play_stream(chunks, lambda: False))
            opening.assert_called_once()
            closing.assert_called_once()
        timestamps = []
        audio = bytearray()
        for body in sent[1:]:
            pes = b"".join(transport_payload(body[i:i + 188]) for i in range(0, len(body), 188))
            timestamps.append(presentation_timestamp(pes))
            audio.extend(pes[14:])
        self.assertEqual(len(audio), 800 + 1600)
        self.assertEqual(timestamps[:3], [0, 3600, 7200])

    def test_cancelled_stream_does_not_pull_audio_or_open_camera(self):
        speaker = TapoTalk("192.168.2.88", "test-password")
        def forbidden_audio():
            raise AssertionError("Cancelled playback pulled cloud audio.")
            yield
        with patch.object(speaker, "open") as opening:
            self.assertFalse(speaker.play_stream(forbidden_audio(), lambda: True))
            opening.assert_not_called()

    def test_stream_stop_closes_generator_and_discards_later_audio(self):
        speaker = TapoTalk("192.168.2.88", "test-password")
        stopped = threading.Event()
        discarded = threading.Event()
        def audio():
            try:
                yield SimpleNamespace(audio_float_array=np.zeros(4800, dtype=np.float32), sample_rate=24000)
                raise AssertionError("Playback requested audio after cancellation.")
            finally:
                discarded.set()
        sent = []
        def capture(body):
            sent.append(body)
            if len(sent) == 2:
                stopped.set()
        with patch.object(speaker, "open"), patch.object(speaker, "close"), patch.object(speaker, "_audio_part", side_effect=capture):
            self.assertFalse(speaker.play_stream(audio(), stopped.is_set))
        self.assertTrue(discarded.is_set())
        self.assertEqual(len(sent), 2)


if __name__ == "__main__":
    unittest.main()
