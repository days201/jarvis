"""Authenticated local Tapo talk using multipart HTTP and A-law MPEG-TS audio.

Protocol references:
https://github.com/AlexxIT/go2rtc/tree/master/pkg/tapo
https://github.com/AlexxIT/go2rtc/tree/master/pkg/mpegts
"""

import hashlib
import ipaddress
import json
import re
import secrets
import socket
import struct
import threading
import time


CLIENT_BOUNDARY = b"----client-stream-boundary--"
DEVICE_BOUNDARY = b"----device-stream-boundary--"


class TalkError(Exception):
    """A talk error without credentials or raw protocol data."""


def digest_authorization(challenge, password, cnonce=None):
    if not challenge.startswith("Digest "):
        raise TalkError("The camera requested an unsupported talk login method.")
    values = {m.group(1).lower(): m.group(2) if m.group(2) is not None else m.group(3)
              for m in re.finditer(r'(\w+)\s*=\s*(?:"([^"\r\n]*)"|([^,\s]+))', challenge[7:])}
    if not values.get("realm") or not values.get("nonce"):
        raise TalkError("The camera returned an incomplete talk login challenge.")
    if values.get("algorithm", "MD5").upper() != "MD5":
        raise TalkError("The camera requested an unsupported digest algorithm.")
    hash_method = hashlib.sha256 if values.get("encrypt_type") == "3" else hashlib.md5
    hashed_password = hash_method(password.encode()).hexdigest().upper()
    md5 = lambda value: hashlib.md5(value.encode()).hexdigest()
    ha1 = md5(f"admin:{values['realm']}:{hashed_password}")
    ha2 = md5("POST:/stream")
    cnonce = cnonce or secrets.token_hex(16)
    header = f'Digest username="admin", realm="{values["realm"]}", nonce="{values["nonce"]}", uri="/stream", algorithm=MD5'
    if values.get("qop"):
        if "auth" not in {item.strip() for item in values["qop"].split(",")}:
            raise TalkError("The camera requested an unsupported digest quality of protection.")
        response = md5(f"{ha1}:{values['nonce']}:00000001:{cnonce}:auth:{ha2}")
        header += f', qop=auth, nc=00000001, cnonce="{cnonce}"'
    else:
        response = md5(f"{ha1}:{values['nonce']}:{ha2}")
    header += f', response="{response}"'
    if values.get("opaque"):
        header += f', opaque="{values["opaque"]}"'
    return header


def mpeg_crc(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)) & 0xFFFFFFFF
    return crc


def pts_bytes(timestamp):
    timestamp &= (1 << 33) - 1
    return bytes([0x21 | ((timestamp >> 29) & 0x0E), (timestamp >> 22) & 0xFF,
                  1 | ((timestamp >> 14) & 0xFE), (timestamp >> 7) & 0xFF, 1 | ((timestamp << 1) & 0xFE)])


class AudioTransport:
    """A single Tapo A-law audio track with a 90 kHz presentation clock."""

    def __init__(self):
        self.sequence = 0
        self.samples = 0

    @staticmethod
    def _psi(pid, section):
        section += struct.pack(">I", mpeg_crc(section))
        packet = bytes([0x47, 0x40 | (pid >> 8), pid & 0xFF, 0x10, 0]) + section
        return packet.ljust(188, b"\xff")

    def header(self):
        pat = bytes.fromhex("00b00d0001c100000001f000")
        pmt = bytes.fromhex("02b0120001c10000fffff00090e100f000")
        return self._psi(0, pat) + self._psi(0x1000, pmt)

    def payload(self, alaw):
        timestamp = self.samples * 90000 // 8000
        self.samples += len(alaw)
        pes = b"\x00\x00\x01\xc0" + struct.pack(">H", len(alaw) + 8) + b"\x80\x80\x05" + pts_bytes(timestamp) + alaw
        output = bytearray()
        first = True
        while pes:
            part, pes = pes[:184], pes[184:]
            pid_high = 0x01 | (0x40 if first else 0)
            counter = self.sequence & 15
            self.sequence += 1
            if len(part) == 184:
                packet = bytes([0x47, pid_high, 0, 0x10 | counter]) + part
            else:
                adaptation_size = 183 - len(part)
                adaptation = b"\x00" + b"\xff" * (adaptation_size - 1) if adaptation_size else b""
                packet = bytes([0x47, pid_high, 0, 0x30 | counter, adaptation_size]) + adaptation + part
            output.extend(packet)
            first = False
        return bytes(output)


def encode_alaw(samples, sample_rate):
    import av
    import numpy as np
    audio = np.asarray(samples, dtype=np.float32).reshape(1, -1)
    if not np.isfinite(audio).all():
        raise TalkError("Speech audio contains invalid samples.")
    frame = av.AudioFrame.from_ndarray(np.clip(audio, -1, 1), format="flt", layout="mono")
    frame.sample_rate = sample_rate
    resampler = av.AudioResampler(format="s16", layout="mono", rate=8000)
    encoder = av.CodecContext.create("pcm_alaw", "w")
    encoder.sample_rate = 8000
    encoder.layout = "mono"
    encoder.format = "s16"
    result = bytearray()
    for converted in resampler.resample(frame) + resampler.resample(None):
        converted.pts = None
        for packet in encoder.encode(converted):
            result.extend(bytes(packet))
    for packet in encoder.encode(None):
        result.extend(bytes(packet))
    return bytes(result)


def encode_alaw_stream(chunks, stopped):
    """Keep resampler history across chunks so packet boundaries do not click."""
    import av
    import numpy as np
    resampler = av.AudioResampler(format="s16", layout="mono", rate=8000)
    encoder = av.CodecContext.create("pcm_alaw", "w")
    encoder.sample_rate = 8000
    encoder.layout = "mono"
    encoder.format = "s16"
    source_rate = None
    for chunk in chunks:
        if stopped():
            return
        if source_rate is None:
            source_rate = chunk.sample_rate
        if chunk.sample_rate != source_rate:
            raise TalkError("Speech sample rate changed during playback.")
        samples = np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(samples).all():
            raise TalkError("Speech audio contains invalid samples.")
        if not samples.size:
            continue
        frame = av.AudioFrame.from_ndarray(np.clip(samples, -1, 1), format="flt", layout="mono")
        frame.sample_rate = source_rate
        for converted in resampler.resample(frame):
            converted.pts = None
            for packet in encoder.encode(converted):
                yield bytes(packet)
    if stopped():
        return
    if source_rate is not None:
        for converted in resampler.resample(None):
            converted.pts = None
            for packet in encoder.encode(converted):
                yield bytes(packet)
        for packet in encoder.encode(None):
            yield bytes(packet)


class TapoTalk:
    def __init__(self, host, password):
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise TalkError("Enter the camera's local IPv4 address.") from None
        if address.version != 4 or not address.is_private or address.is_loopback or address.is_multicast or address.is_unspecified:
            raise TalkError("Use the camera's private IPv4 address for talk.")
        if not password:
            raise TalkError("Enter the Tapo account password for camera speaker access.")
        self.host = str(address)
        self._password = password
        self._socket = None
        self._reader = None
        self._session = None
        self._lock = threading.Lock()

    def __repr__(self):
        return f"TapoTalk(host={self.host!r})"

    def _readline(self):
        line = self._reader.readline(8193)
        if len(line) > 8192:
            raise TalkError("The camera returned an oversized talk header.")
        if not line:
            raise TalkError("The camera closed its speaker connection.")
        return line

    def _headers(self, http=False):
        status = None
        if http:
            line = self._readline()
            match = re.match(rb"HTTP/1\.[01] (\d{3})", line)
            if not match:
                raise TalkError("The camera returned an invalid talk response.")
            status = int(match[1])
        headers = {}
        for _ in range(64):
            line = self._readline()
            if line in {b"\r\n", b"\n"}:
                return status, headers
            name, separator, value = line.partition(b":")
            if not separator:
                raise TalkError("The camera returned an invalid talk header.")
            headers[name.decode("ascii").lower()] = value.decode("latin-1").strip()
        raise TalkError("The camera returned too many talk headers.")

    def _request_headers(self, authorization=None):
        lines = ["POST /stream HTTP/1.1", f"Host: {self.host}:8800", "Content-Type: multipart/mixed; boundary=--client-stream-boundary--",
                 "Connection: keep-alive", "Content-Length: 0"]
        if authorization:
            lines.append("Authorization: " + authorization)
        self._socket.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))

    def _part(self, content_type, body, extra=()):
        lines = [CLIENT_BOUNDARY.decode(), "Content-Type: " + content_type, f"Content-Length: {len(body)}"] + list(extra)
        self._socket.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body + b"\r\n")

    def _read_part(self):
        for _ in range(16):
            line = self._readline().strip()
            if line == DEVICE_BOUNDARY:
                break
        else:
            raise TalkError("The camera did not return a talk message boundary.")
        _, headers = self._headers()
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            raise TalkError("The camera returned an invalid talk message length.") from None
        if not 0 < length <= 65536:
            raise TalkError("The camera returned an unexpected talk message size.")
        body = self._reader.read(length)
        if len(body) != length:
            raise TalkError("The camera returned an incomplete talk message.")
        if headers.get("x-if-encrypt", "0") == "1":
            raise TalkError("This firmware requires encrypted talk control messages.")
        try:
            return json.loads(body)
        except ValueError:
            raise TalkError("The camera returned an unreadable talk message.") from None

    def open(self):
        self.close()
        try:
            self._socket = socket.create_connection((self.host, 8800), timeout=4)
            self._socket.settimeout(4)
            self._reader = self._socket.makefile("rb")
            self._request_headers()
            status, headers = self._headers(http=True)
            if status != 401:
                raise TalkError("The camera did not request an authenticated talk login.")
            length = int(headers.get("content-length", "0"))
            if not 0 <= length <= 65536:
                raise TalkError("The camera returned an unexpected login message size.")
            if length:
                self._reader.read(length)
            authorization = digest_authorization(headers.get("www-authenticate", ""), self._password)
            self._request_headers(authorization)
            status, headers = self._headers(http=True)
            if status == 401:
                raise TalkError("Camera speaker login was rejected. Enter the Tapo account password under Keys.")
            if status != 200:
                raise TalkError(f"Camera speaker connection returned HTTP {status}.")
            request = {"params": {"talk": {"mode": "aec"}, "method": "get"}, "seq": 3, "type": "request"}
            self._part("application/json", json.dumps(request, separators=(",", ":")).encode())
            response = self._read_part()
            params = response.get("params", {})
            code = response.get("error_code", params.get("error_code", 0))
            session = params.get("session_id")
            if code != 0 or not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session):
                raise TalkError("The camera declined the talk session. Close talk in the Tapo app and try again.")
            self._session = session
            return {"authenticated": True, "talk_session": True}
        except TalkError:
            self.close()
            raise
        except Exception:
            self.close()
            raise TalkError("Could not connect to the camera speaker on local port 8800.") from None

    def _audio_part(self, body):
        self._part("audio/mp2t", body, ("X-If-Encrypt: 0", "X-Session-Id: " + self._session))

    def play(self, samples, sample_rate, stopped):
        if stopped():
            return False
        alaw = encode_alaw(samples, sample_rate)
        with self._lock:
            try:
                if stopped():
                    return False
                self.open()
                transport = AudioTransport()
                self._audio_part(transport.header())
                start = time.monotonic()
                audio = alaw + b"\xd5" * 1600
                sent = 0
                while sent < len(audio) and not stopped():
                    packet = audio[sent:sent + 320]
                    self._audio_part(transport.payload(packet))
                    sent += len(packet)
                    delay = start + sent / 8000 - time.monotonic()
                    while delay > 0 and not stopped():
                        time.sleep(min(delay, 0.02))
                        delay = start + sent / 8000 - time.monotonic()
                return not stopped()
            except TalkError:
                raise
            except Exception:
                raise TalkError("The camera speaker connection ended during playback.") from None
            finally:
                self.close()

    def play_stream(self, chunks, stopped):
        """Send a continuous reply through one paced camera talk session."""
        if stopped():
            return False
        with self._lock:
            pending = bytearray()
            opened = False
            transport = AudioTransport()
            sent = 0
            started = 0.0

            def send_ready(final=False):
                nonlocal opened, sent, started
                if not opened and pending and (len(pending) >= 960 or final) and not stopped():
                    self.open()
                    self._audio_part(transport.header())
                    started = time.monotonic()
                    opened = True
                while opened and pending and (len(pending) >= 320 or final) and not stopped():
                    packet = bytes(pending[:320])
                    del pending[:320]
                    self._audio_part(transport.payload(packet))
                    sent += len(packet)
                    delay = started + sent / 8000 - time.monotonic()
                    while delay > 0 and not stopped():
                        time.sleep(min(delay, 0.02))
                        delay = started + sent / 8000 - time.monotonic()

            source = iter(chunks)
            try:
                for encoded in encode_alaw_stream(source, stopped):
                    if stopped():
                        return False
                    pending.extend(encoded)
                    send_ready()
                if stopped():
                    return False
                if opened or pending:
                    pending.extend(b"\xd5" * 1600)
                    send_ready(final=True)
                return opened and not stopped()
            except TalkError:
                raise
            except Exception as error:
                if stopped():
                    return False
                from cloud_voice import VoiceError
                if isinstance(error, VoiceError):
                    raise
                raise TalkError("Camera speech streaming ended. Check the voice service and camera connection.") from None
            finally:
                if hasattr(source, "close"):
                    source.close()
                self.close()

    def close(self):
        connection, reader = self._socket, self._reader
        self._socket = self._reader = self._session = None
        if connection:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if reader:
            try:
                reader.close()
            except OSError:
                pass
        if connection:
            connection.close()
