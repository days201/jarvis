"""Local ONVIF control and RTSP video for the owner's Tapo camera."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import math
import re
import secrets
import time
from urllib.parse import quote, urlsplit
import xml.etree.ElementTree as ET

import requests


SOAP = "http://www.w3.org/2003/05/soap-envelope"
DEVICE = "http://www.onvif.org/ver10/device/wsdl"
MEDIA = "http://www.onvif.org/ver10/media/wsdl"
PTZ = "http://www.onvif.org/ver20/ptz/wsdl"
SCHEMA = "http://www.onvif.org/ver10/schema"
WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
NS = {"s": SOAP, "d": DEVICE, "m": MEDIA, "p": PTZ, "tt": SCHEMA}
ET.register_namespace("s", SOAP)
ET.register_namespace("wsse", WSSE)
ET.register_namespace("wsu", WSU)
ET.register_namespace("tds", DEVICE)
ET.register_namespace("trt", MEDIA)
ET.register_namespace("tptz", PTZ)
ET.register_namespace("tt", SCHEMA)


class CameraError(Exception):
    """A camera error safe to show without raw requests or credentials."""


def child(parent, namespace, name, text=None, **attributes):
    element = ET.SubElement(parent, f"{{{namespace}}}{name}", attributes)
    if text is not None:
        element.text = str(text)
    return element


class Camera:
    def __init__(self, host, username, password):
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise CameraError("Enter the camera's local IP address.") from None
        if address.version != 4 or not address.is_private or address.is_loopback or address.is_unspecified or address.is_multicast:
            raise CameraError("Use the camera's private IPv4 address.")
        if not username or not password:
            raise CameraError("Enter the local Camera Account username and password.")
        self.host = str(address)
        self._username = username
        self._password = password
        self._offset = timedelta()
        self.device_url = f"http://{self.host}:2020/onvif/device_service"
        self.media_url = self.device_url
        self.ptz_url = self.device_url
        self.profiles = []
        self.profile = ""
        self.motion_defaults = {}
        self.supports_continuous = False
        self.continuous_timeout = 1.0

    def __repr__(self):
        return f"Camera(host={self.host!r})"

    def create_speaker(self, password=None):
        from tapo_talk import TapoTalk
        return TapoTalk(self.host, password if password is not None else self._password)

    def _validate_url(self, url, stream=False):
        parsed = urlsplit(url)
        allowed_scheme = {"rtsp"} if stream else {"http"}
        allowed_port = {554} if stream else {2020}
        try:
            port = parsed.port or (554 if stream else 80)
        except ValueError:
            raise CameraError("The camera returned an invalid service address.") from None
        if (parsed.hostname != self.host or parsed.scheme not in allowed_scheme
                or port not in allowed_port or parsed.username or parsed.password
                or parsed.fragment):
            raise CameraError("The camera returned an unexpected service address.")
        return url

    def _request(self, url, namespace, operation, fill=None, authenticated=True, request_timeout=(3, 5)):
        self._validate_url(url)
        envelope = ET.Element(f"{{{SOAP}}}Envelope")
        if authenticated:
            header = child(envelope, SOAP, "Header")
            security = child(header, WSSE, "Security", **{f"{{{SOAP}}}mustUnderstand": "true"})
            token = child(security, WSSE, "UsernameToken")
            nonce = secrets.token_bytes(20)
            created = (datetime.now(timezone.utc) + self._offset).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + self._password.encode()).digest()).decode()
            child(token, WSSE, "Username", self._username)
            child(token, WSSE, "Password", digest, Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest")
            child(token, WSSE, "Nonce", base64.b64encode(nonce).decode(), EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary")
            child(token, WSU, "Created", created)
        body = child(envelope, SOAP, "Body")
        command = child(body, namespace, operation)
        if fill:
            fill(command)
        action = f"{namespace}/{operation}"
        headers = {"Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"', "SOAPAction": f'"{action}"'}
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(url, data=ET.tostring(envelope, encoding="utf-8"), headers=headers, timeout=request_timeout, allow_redirects=False)
        except requests.RequestException:
            raise CameraError("The camera did not respond. Check its power and local network connection.") from None
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError:
            raise CameraError("The camera returned an unreadable response.") from None
        fault = root.find(".//s:Fault", NS)
        if fault is not None:
            if "NotAuthorized" in " ".join(fault.itertext()):
                raise CameraError("The camera rejected the login. Check the separate local Camera Account credentials.")
            raise CameraError(f"The camera did not accept {operation}.")
        if response.status_code != 200:
            raise CameraError(f"The camera returned HTTP {response.status_code} for {operation}.")
        return root

    def connect(self):
        before = datetime.now(timezone.utc)
        root = self._request(self.device_url, DEVICE, "GetSystemDateAndTime", authenticated=False)
        date = root.find(".//tt:UTCDateTime", NS)
        if date is not None:
            try:
                parts = [int(date.findtext(path, namespaces=NS)) for path in
                         ["tt:Date/tt:Year", "tt:Date/tt:Month", "tt:Date/tt:Day", "tt:Time/tt:Hour", "tt:Time/tt:Minute", "tt:Time/tt:Second"]]
                midpoint = before + (datetime.now(timezone.utc) - before) / 2
                self._offset = datetime(*parts, tzinfo=timezone.utc) - midpoint
            except (ValueError, TypeError):
                pass
        info = self._request(self.device_url, DEVICE, "GetDeviceInformation")
        result = {name.lower(): info.findtext(f".//d:{name}", namespaces=NS) for name in
                  ["Manufacturer", "Model", "FirmwareVersion", "HardwareId"]}
        capabilities = self._request(self.device_url, DEVICE, "GetCapabilities", lambda e: child(e, DEVICE, "Category", "All"))
        for name, attr in [("Media", "media_url"), ("PTZ", "ptz_url")]:
            url = capabilities.findtext(f".//tt:{name}/tt:XAddr", namespaces=NS)
            if not url:
                raise CameraError(f"The camera did not advertise its {name} service.")
            setattr(self, attr, self._validate_url(url))
        root = self._request(self.media_url, MEDIA, "GetProfiles")
        self.profiles = []
        for profile in root.findall(".//m:Profiles", NS):
            self.profiles.append({
                "token": profile.get("token"),
                "name": profile.findtext("tt:Name", namespaces=NS),
                "width": int(profile.findtext("tt:VideoEncoderConfiguration/tt:Resolution/tt:Width", "0", NS)),
                "height": int(profile.findtext("tt:VideoEncoderConfiguration/tt:Resolution/tt:Height", "0", NS)),
                "fps": int(profile.findtext("tt:VideoEncoderConfiguration/tt:RateControl/tt:FrameRateLimit", "0", NS)),
                "encoding": profile.findtext("tt:VideoEncoderConfiguration/tt:Encoding", namespaces=NS),
                "ptz": profile.find("tt:PTZConfiguration", NS) is not None,
            })
        controllable = [p for p in self.profiles if p["ptz"] and p["encoding"] == "H264"]
        if not controllable:
            raise CameraError("The camera did not advertise a controllable video profile.")
        self.profile = controllable[0]["token"]
        selected = next(p for p in root.findall(".//m:Profiles", NS) if p.get("token") == self.profile)
        speed = selected.find("tt:PTZConfiguration/tt:DefaultPTZSpeed/tt:PanTilt", NS)
        if speed is not None:
            self.motion_defaults = {"pan_speed": float(speed.get("x")), "tilt_speed": float(speed.get("y"))}
        result["profiles"] = self.profiles
        result["motion_defaults"] = self.motion_defaults
        self._read_continuous_options(selected.find("tt:PTZConfiguration", NS))
        result["continuous_tracking_supported"] = self.supports_continuous
        return result

    def _read_continuous_options(self, configuration):
        self.supports_continuous = False
        if configuration is None or not configuration.get("token"):
            return
        try:
            options = self._request(self.ptz_url, PTZ, "GetConfigurationOptions",
                                    lambda e: child(e, PTZ, "ConfigurationToken", configuration.get("token")))
            spaces = options.findall(".//tt:ContinuousPanTiltVelocitySpace", NS)
            valid = any(space.findtext("tt:URI", namespaces=NS) ==
                        "http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace"
                        and all(float(space.findtext(f"tt:{axis}Range/tt:Min", namespaces=NS)) <= -1
                                and float(space.findtext(f"tt:{axis}Range/tt:Max", namespaces=NS)) >= 1
                                for axis in ("X", "Y")) for space in spaces)
            def seconds(text):
                match = re.fullmatch(r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?", text or "")
                if not match or not any(match.groups()):
                    raise ValueError()
                return sum(float(value or 0) * scale for value, scale in zip(match.groups(), (3600, 60, 1)))
            low = seconds(options.findtext(".//tt:PTZTimeout/tt:Min", namespaces=NS))
            high = seconds(options.findtext(".//tt:PTZTimeout/tt:Max", namespaces=NS))
            self.supports_continuous = valid and 0 <= low <= 1 <= high
        except (CameraError, ValueError, TypeError):
            # Absolute movement remains available on devices without this mode.
            self.supports_continuous = False

    def continuous_move(self, pan, tilt, timeout=None):
        if not self.supports_continuous:
            raise CameraError("This camera has not advertised continuous pan and tilt control.")
        timeout = self.continuous_timeout if timeout is None else timeout
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (pan, tilt, timeout)) or not (-1 <= pan <= 1 and -1 <= tilt <= 1) or timeout != 1:
            raise CameraError("Continuous velocity must be between -1 and 1 with a one-second expiry.")
        def fill(e):
            child(e, PTZ, "ProfileToken", self.profile)
            velocity = child(e, PTZ, "Velocity")
            child(velocity, SCHEMA, "PanTilt", x=f"{pan:.3f}", y=f"{tilt:.3f}",
                  space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace")
            child(e, PTZ, "Timeout", "PT1S")
        self._request(self.ptz_url, PTZ, "ContinuousMove", fill, request_timeout=(0.5, 1.25))

    def position(self):
        root = self._request(self.ptz_url, PTZ, "GetStatus", lambda e: child(e, PTZ, "ProfileToken", self.profile))
        position = root.find(".//tt:Position/tt:PanTilt", NS)
        if position is None:
            raise CameraError("The camera did not return its pan and tilt position.")
        try:
            return float(position.get("x")), float(position.get("y"))
        except (TypeError, ValueError):
            raise CameraError("The camera returned an invalid position.") from None

    def move_to(self, x, y, speed=None):
        import math
        if not all(math.isfinite(value) for value in [x, y]):
            raise CameraError("Movement values must be finite numbers.")
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            raise CameraError("Movement exceeds the controller's limits.")
        if speed is not None and (not math.isfinite(speed) or not 0 < speed <= 1):
            raise CameraError("Speed must be greater than zero and no higher than one.")
        def fill(e):
            child(e, PTZ, "ProfileToken", self.profile)
            position = child(e, PTZ, "Position")
            child(position, SCHEMA, "PanTilt", x=str(x), y=str(y))
            if speed is not None:
                velocity = child(e, PTZ, "Speed")
                child(velocity, SCHEMA, "PanTilt", x=str(speed), y=str(speed))
        self._request(self.ptz_url, PTZ, "AbsoluteMove", fill)

    def wait_position(self, target, timeout=6, cancelled=None):
        deadline = time.monotonic() + timeout
        last = self.position()
        while max(abs(last[i] - target[i]) for i in [0, 1]) > 0.001:
            if cancelled is not None and cancelled.is_set():
                return last
            if time.monotonic() >= deadline:
                raise CameraError("The camera did not reach the requested position in time.")
            time.sleep(0.15)
            last = self.position()
        return last

    def stop(self):
        def fill(e):
            child(e, PTZ, "ProfileToken", self.profile)
            child(e, PTZ, "PanTilt", "true")
            child(e, PTZ, "Zoom", "false")
        self._request(self.ptz_url, PTZ, "Stop", fill)

    def stream_uri(self, low_quality=True):
        choices = [p for p in self.profiles if p["encoding"] == "H264"]
        if not choices:
            raise CameraError("No H.264 stream is available.")
        profile = min(choices, key=lambda p: p["width"] * p["height"]) if low_quality else max(choices, key=lambda p: p["width"] * p["height"])
        def fill(e):
            setup = child(e, MEDIA, "StreamSetup")
            child(setup, SCHEMA, "Stream", "RTP-Unicast")
            transport = child(setup, SCHEMA, "Transport")
            child(transport, SCHEMA, "Protocol", "RTSP")
            child(e, MEDIA, "ProfileToken", profile["token"])
        root = self._request(self.media_url, MEDIA, "GetStreamUri", fill)
        uri = root.findtext(".//tt:Uri", namespaces=NS)
        if not uri:
            raise CameraError("The camera did not return a video address.")
        return self._validate_url(uri, stream=True)

    def open_video(self, low_quality=True):
        import av
        av.logging.set_level(av.logging.PANIC)
        uri = urlsplit(self.stream_uri(low_quality))
        username = quote(self._username, safe="")
        password = quote(self._password, safe="")
        address = f"rtsp://{username}:{password}@{uri.netloc}{uri.path}"
        if uri.query:
            address += "?" + uri.query
        try:
            return av.open(address, options={"rtsp_transport": "tcp", "max_delay": "100000"}, timeout=(6, 4))
        except Exception:
            raise CameraError("Could not open the live video. Check the Camera Account and camera connection.") from None

