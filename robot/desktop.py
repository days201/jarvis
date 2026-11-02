"""A local desktop camera viewer with bounded pan and tilt controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .camera import Camera, CameraError
from .conversation import Conversation
from .robot import Cancelled, Robot


BACKGROUND = "#10161f"
PANEL = "#192330"
TEXT = "#edf3fa"
MUTED = "#a5b6c9"


class Desktop:
    def __init__(self, root, login=None, self_check=False, api_key=None, deepgram_key=None):
        self.root = root
        self.camera = None
        self.robot = None
        self.desk_position = None
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.video_stop = threading.Event()
        self.cancel_move = threading.Event()
        self.workers = ThreadPoolExecutor(max_workers=2)
        self.busy = False
        self.closing = False
        self.frame_count = 0
        self.audio_frames_seen = 0
        self.last_frame_size = None
        self.self_check = self_check
        self.check_result = 1
        self.photo = None
        self.status = tk.StringVar(value="Connect with the local Camera Account created in the Tapo app.")
        self.position_text = tk.StringVar(value="")
        self.details = tk.StringVar(value="Tapo C210 | Local video and pan/tilt")
        self.host = tk.StringVar(value="192.168.2.88")
        self.username = tk.StringVar()
        self.password = tk.StringVar()
        self.step = tk.DoubleVar(value=15)
        self.follow_people = tk.BooleanVar(value=True)
        self.small_gestures = tk.BooleanVar(value=True)
        self.allow_greetings = tk.BooleanVar(value=True)
        self.cloud_vision = tk.BooleanVar(value=True)
        self.robot_status = tk.StringVar(value="Cloud vision and local person following")
        root.title("Jarvis | Camera and voice")
        root.geometry("1280x880")
        root.minsize(1100, 800)
        root.configure(bg=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11), padding=(10, 10), background="#2c4056", foreground=TEXT)
        style.map("TButton", background=[("active", "#365875"), ("disabled", "#202b38")], foreground=[("disabled", "#6e8092")])
        style.configure("TEntry", padding=7, fieldbackground="#243245", foreground=TEXT)
        style.configure("TScale", background=PANEL)
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", background="#253447", foreground=TEXT, padding=(15, 9), font=("Segoe UI", 11))
        style.map("TNotebook.Tab", background=[("selected", "#365875")])
        style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 9))
        shell = ttk.Frame(root, padding=24)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        title = ttk.Frame(shell)
        title.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(title, text="Jarvis", font=("Segoe UI Semibold", 25)).pack(anchor="w")
        ttk.Label(title, textvariable=self.details, foreground=MUTED).pack(anchor="w", pady=(4, 0))
        self.login_panel = ttk.Frame(shell)
        self.login_panel.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        self.login_entries = []
        for column, (label, variable, secret) in enumerate([
            ("Camera IP", self.host, False), ("Camera Account username", self.username, False), ("Password", self.password, True)
        ]):
            group = ttk.Frame(self.login_panel)
            group.grid(row=0, column=column, sticky="ew", padx=(0, 12))
            self.login_panel.columnconfigure(column, weight=1)
            ttk.Label(group, text=label, foreground=MUTED).pack(anchor="w", pady=(0, 5))
            entry = ttk.Entry(group, textvariable=variable, show="*" if secret else "")
            entry.pack(fill="x")
            entry.bind("<Return>", lambda event: self.connect())
            self.login_entries.append(entry)
        self.connect_button = ttk.Button(self.login_panel, text="Connect", command=self.connect)
        self.connect_button.grid(row=0, column=3, sticky="s")
        self.content = ttk.Frame(shell)
        self.content.grid(row=2, column=0, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.preview = tk.Label(self.content, bg="#080d13", fg=MUTED, text="The live camera view will appear here.", font=("Segoe UI", 13))
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        self.side_tabs = ttk.Notebook(self.content, width=350)
        self.side_tabs.grid(row=0, column=1, sticky="ns")
        controls = tk.Frame(self.side_tabs, background=PANEL, padx=16, pady=18)
        self.controls_panel = controls
        self.side_tabs.add(controls, text="Movement")
        tk.Label(controls, text="Pan and tilt", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 16)).pack(anchor="w")
        pad = tk.Frame(controls, bg=PANEL)
        pad.pack(pady=10)
        self.buttons = []
        for text, row, col, x, y in [("Up", 0, 1, 0, 1), ("Left", 1, 0, -1, 0), ("Right", 1, 2, 1, 0), ("Down", 2, 1, 0, -1)]:
            button = ttk.Button(pad, text=text, width=5, command=lambda x=x, y=y: self.nudge(x, y))
            button.grid(row=row, column=col, padx=2, pady=3)
            self.buttons.append(button)
        self.stop_button = ttk.Button(pad, text="Stop", width=5, command=self.stop)
        self.stop_button.grid(row=1, column=1, padx=2, pady=3)
        tk.Label(controls, text="Step size in degrees", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w")
        scale_row = tk.Frame(controls, bg=PANEL)
        scale_row.pack(fill="x", pady=(6, 10))
        self.step_scale = ttk.Scale(scale_row, from_=1, to=45, variable=self.step, length=145)
        self.step_scale.pack(side="left")
        self.step_label = tk.Label(scale_row, text="15", bg=PANEL, fg=TEXT, width=3, font=("Segoe UI", 11))
        self.step_label.pack(side="right")
        self.step.trace_add("write", lambda *args: self.step_label.configure(text=str(round(self.step.get()))))
        tk.Label(controls, text="Uses the camera's default speed.", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 12))
        for text, command in [("Remember desk position", self.remember), ("Return to desk", self.return_to_desk), ("Nod", self.nod)]:
            button = ttk.Button(controls, text=text, command=command)
            button.pack(fill="x", pady=5)
            self.buttons.append(button)
        tk.Label(controls, textvariable=self.position_text, bg=PANEL, fg=MUTED, font=("Segoe UI", 10), justify="left").pack(anchor="w", pady=(12, 0))
        for label, variable in [("Follow people locally", self.follow_people), ("Small expressive gestures", self.small_gestures),
                                ("Occasional arrival greetings", self.allow_greetings), ("Send selected images to DeepSeek", self.cloud_vision)]:
            ttk.Checkbutton(controls, text=label, variable=variable, command=self.configure_robot).pack(anchor="w", pady=(5, 0))
        tk.Label(controls, textvariable=self.robot_status, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=285, justify="left").pack(anchor="w", pady=(10, 0))
        tk.Label(controls, text="Arrow keys move. Space stops and pauses following.", bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=285, justify="left").pack(anchor="w", pady=(8, 0))
        self.conversation = Conversation(self.side_tabs, api_key=api_key, deepgram_key=deepgram_key)
        self.side_tabs.add(self.conversation, text="Conversation")
        if api_key:
            self.side_tabs.select(self.conversation)
        ttk.Label(shell, textvariable=self.status, foreground=MUTED, wraplength=1120).grid(row=3, column=0, sticky="ew", pady=(16, 0))
        self.enable_controls(False)
        for key, x, y in [("Up", 0, 1), ("Down", 0, -1), ("Left", -1, 0), ("Right", 1, 0)]:
            root.bind(f"<{key}>", lambda event, x=x, y=y: self.key_move(event, x, y))
        root.bind("<space>", self.key_stop)
        root.after(50, self.poll)
        if login:
            self.host.set(login["host"])
            self.username.set(login["username"])
            self.password.set(login["password"])
            root.after(100, self.connect)
        if self_check:
            root.after(12000, self.finish_check)

    def enable_controls(self, enabled):
        for button in self.buttons:
            button.configure(state="normal" if enabled else "disabled")
        self.stop_button.configure(state="normal" if self.camera else "disabled")

    def connect(self):
        if self.busy or self.camera:
            return
        try:
            camera = Camera(self.host.get().strip(), self.username.get().strip(), self.password.get())
        except CameraError as error:
            self.status.set(str(error))
            return
        self.busy = True
        self.connect_button.configure(state="disabled")
        self.status.set("Connecting to the camera...")
        def job():
            try:
                info = camera.connect()
                position = camera.position()
                self.events.put(("connected", (camera, info, position)))
            except CameraError as error:
                self.events.put(("login_error", str(error)))
            except Exception:
                self.events.put(("login_error", "Could not connect to the camera."))
        self.workers.submit(job)

    def connected(self, camera, info, position):
        self.camera = camera
        self.busy = False
        self.desk_position = position
        self.password.set("")
        self.login_panel.grid_remove()
        main = max((p for p in info["profiles"] if p["encoding"] == "H264"), key=lambda p: p["width"] * p["height"])
        self.details.set(f'{info["model"]} | {camera.host} | Main stream {main["width"]} x {main["height"]} at {main["fps"]} fps')
        self.status.set("Connected. Opening live video. The current angle is your desk position.")
        self.set_position(position)
        self.enable_controls(True)
        self.robot = Robot(camera, position, emit=lambda kind, data: self.events.put((kind, data)))
        self.configure_robot()
        self.conversation.set_robot(self.robot)
        self.conversation.set_camera(camera)
        threading.Thread(target=self.video_loop, daemon=True, name="camera-video").start()

    def video_loop(self):
        import av
        retry = 0.5
        while not self.video_stop.is_set():
            resampler = None
            try:
                with self.camera.open_video(low_quality=True) as stream:
                    self.events.put(("audio_available", bool(stream.streams.audio)))
                    for packet in stream.demux():
                        if self.video_stop.is_set():
                            return
                        for frame in packet.decode():
                            retry = 0.5
                            if isinstance(frame, av.AudioFrame):
                                self.audio_frames_seen += 1
                                if self.conversation.speech.microphone_active:
                                    if resampler is None:
                                        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                                    frame.pts = None
                                    for audio_frame in resampler.resample(frame):
                                        self.conversation.feed_audio(audio_frame.to_ndarray().tobytes())
                                else:
                                    resampler = None
                            elif isinstance(frame, av.VideoFrame):
                                image = frame.to_image()
                                if self.robot:
                                    self.robot.frames.push(image)
                                try:
                                    self.frames.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self.frames.put_nowait(image)
                                except queue.Full:
                                    pass
            except Exception:
                pass
            if self.video_stop.is_set():
                return
            self.events.put(("audio_available", False))
            if self.robot:
                self.robot.disconnected()
            self.events.put(("video_error", "Camera stream disconnected. Reconnecting automatically..."))
            self.video_stop.wait(retry)
            retry = min(retry * 2, 8.0)

    def set_position(self, position):
        if self.desk_position is not None:
            pan = (position[0] - self.desk_position[0]) * 170
            tilt = (position[1] - self.desk_position[1]) * 35
            self.position_text.set(f"Approx. offset from desk\nPan {pan:+.1f} degrees\nTilt {tilt:+.1f} degrees")

    def action(self, label, function, resume_after=15):
        if self.busy or not self.camera or self.closing:
            return
        self.busy = True
        self.cancel_move.clear()
        self.enable_controls(False)
        self.status.set(label)
        epoch = self.robot.motion_epoch
        def job():
            try:
                with self.robot.manual(resume_after=resume_after, expected_epoch=epoch):
                    position = function()
                    self.robot.position = tuple(position)
                self.events.put(("action_done", position))
            except CameraError as error:
                self.events.put(("action_error", str(error)))
            except Exception:
                self.events.put(("action_error", "The camera operation failed."))
        self.workers.submit(job)

    def move(self, target):
        cancelled = Cancelled(self.cancel_move, self.robot.motion_cancel if self.robot else None)
        try:
            if cancelled.is_set():
                return self.camera.position()
            self.camera.move_to(*target)
            self.camera.wait_position(target, timeout=20, cancelled=cancelled)
        finally:
            self.camera.stop()
        return self.camera.position()

    def nudge(self, x, y):
        degrees = round(self.step.get())
        def job():
            pan, tilt = self.camera.position()
            target = (max(-1, min(1, pan + x * degrees / 170 * self.robot.motion.pan_right)),
                      max(-1, min(1, tilt + y * degrees / 35 * self.robot.motion.tilt_up)))
            return self.move(target)
        self.action("Moving the camera...", job)

    def remember(self):
        def job():
            position = self.camera.position()
            self.desk_position = position
            if self.robot:
                self.robot.desk_position = position
            return position
        self.action("Remembering the desk position...", job)

    def return_to_desk(self):
        if self.desk_position is not None:
            self.action("Returning to the desk position...", lambda: self.move(self.desk_position))

    def nod(self):
        def job():
            start = self.camera.position()
            sign = 1 if start[1] < 0.8 else -1
            try:
                self.move((start[0], max(-1, min(1, start[1] + sign * 3 / 35))))
                self.cancel_move.wait(0.15)
            finally:
                if not self.cancel_move.is_set() and not self.robot.motion_cancel.is_set():
                    self.move(start)
                else:
                    self.camera.stop()
            return self.camera.position()
        self.action("Nodding...", job, resume_after=0.45)

    def stop(self):
        if not self.camera or self.closing:
            return
        self.cancel_move.set()
        if self.robot:
            self.robot.stop()
        self.status.set("Stopping the camera...")
        def job():
            try:
                self.camera.stop()
                with self.robot.motion_lock:
                    position = self.camera.position()
                    self.robot.position = tuple(position)
                self.events.put(("stopped", position))
            except CameraError as error:
                self.events.put(("stop_error", str(error)))
        self.workers.submit(job)

    def configure_robot(self):
        if self.robot:
            self.robot.configure(following=self.follow_people.get(), gestures=self.small_gestures.get(),
                                 greetings=self.allow_greetings.get(), cloud_vision=self.cloud_vision.get())

    def key_move(self, event, x, y):
        if self.side_tabs.select() == str(self.conversation):
            return
        if self.camera and not isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
            self.nudge(x, y)
            return "break"

    def key_stop(self, event):
        if self.side_tabs.select() == str(self.conversation):
            return
        if self.camera and not isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
            self.stop()
            return "break"

    def poll(self):
        if self.closing:
            return
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "connected":
                self.connected(*data)
            elif kind == "audio_available":
                self.conversation.set_microphone(data)
            elif kind == "robot_position":
                self.set_position(data)
            elif kind == "robot_following":
                self.follow_people.set(data)
            elif kind == "robot_target":
                if self.robot:
                    self.robot_status.set(self.robot.status)
            elif kind == "robot_error":
                self.robot_status.set(data)
            elif kind == "person_arrived":
                self.conversation.greet()
            elif kind == "login_error":
                self.busy = False
                self.connect_button.configure(state="normal")
                self.password.set("")
                self.status.set(data)
            elif kind in {"action_done", "stopped"}:
                if kind == "action_done":
                    self.busy = False
                    self.enable_controls(True)
                self.set_position(data)
                self.status.set("Stopped." if self.cancel_move.is_set() else "Ready.")
            elif kind in {"action_error", "stop_error", "video_error"}:
                if kind == "action_error":
                    self.busy = False
                    self.enable_controls(True)
                self.status.set(data)
        try:
            image = self.frames.get_nowait()
        except queue.Empty:
            image = None
        if image is not None:
            self.frame_count += 1
            self.last_frame_size = image.size
            size = (max(100, self.preview.winfo_width() - 12), max(100, self.preview.winfo_height() - 12))
            image.thumbnail(size, Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="")
            if self.frame_count == 1:
                self.status.set("Live video connected. Use the buttons or arrow keys to move the camera.")
        self.root.after(50, self.poll)

    def finish_check(self):
        self.side_tabs.select(self.controls_panel)
        self.root.update_idletasks()
        movement_fits = all(w.winfo_y() + w.winfo_height() <= self.controls_panel.winfo_height() for w in self.controls_panel.winfo_children())
        self.side_tabs.select(self.conversation)
        self.root.update_idletasks()
        result = {"connected": self.camera is not None, "displayed_frames": self.frame_count,
                  "video_dimensions": self.last_frame_size,
                  "audio_frames_received": self.audio_frames_seen,
                  "microphone_available": self.conversation.microphone_available,
                  "voice_output": self.conversation.output.get(),
                  "camera_speaker_ready": self.conversation.speech._camera_speaker is not None,
                  "tts_ready": self.conversation.speech.tts_ready.is_set(),
                  "recognition_ready": self.conversation.speech.recognition_ready.is_set(),
                  "transcription_provider": self.conversation.speech.recognition_provider,
                  "tts_provider": self.conversation.speech.tts_provider,
                  "hands_free_enabled": self.conversation.speech.hands_free.is_set(),
                  "voice_state": self.conversation.speech.voice_state,
                  "controls_enabled": all(b.instate(["!disabled"]) for b in self.buttons),
                  "button_states": {str(b.cget("text")): str(b.cget("state")) for b in self.buttons},
                  "busy": self.busy, "status": self.status.get(),
                  "step_degrees": self.step.get(),
                  "maximum_step_degrees": float(self.step_scale.cget("to")),
                  "panel_children_fit": movement_fits,
                  "conversation_children_fit": all(w.winfo_y() + w.winfo_height() <= self.conversation.winfo_height() for w in self.conversation.winfo_children() if w.winfo_ismapped()),
                  "talk_enabled": self.conversation.talk_button.instate(["!disabled"]),
                  "preview_size": [self.preview.winfo_width(), self.preview.winfo_height()],
                  "window_size": [self.root.winfo_width(), self.root.winfo_height()]}
        required = [result["connected"], result["displayed_frames"] > 0, result["controls_enabled"], result["panel_children_fit"], result["conversation_children_fit"], result["tts_ready"], result["recognition_ready"], result["microphone_available"]]
        self.check_result = 0 if all(required) else 1
        print(json.dumps(result), flush=True)
        self.close()

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.video_stop.set()
        self.cancel_move.set()
        self.password.set("")
        self.conversation.close()
        if self.robot:
            self.robot.close()
        if self.camera:
            def stop_on_close():
                try:
                    self.camera.stop()
                except Exception:
                    pass
            threading.Thread(target=stop_on_close, daemon=True).start()
        self.workers.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin-login", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    login = json.loads(sys.stdin.read()) if args.stdin_login else None
    root = tk.Tk()
    api_key = (login or {}).get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
    deepgram_key = (login or {}).get("deepgram_key") or os.environ.get("DEEPGRAM_API_KEY")
    app = Desktop(root, login=login, self_check=args.self_check, api_key=api_key, deepgram_key=deepgram_key)
    del api_key
    del deepgram_key
    del login
    root.mainloop()
    return app.check_result if args.self_check else 0


if __name__ == "__main__":
    raise SystemExit(main())

