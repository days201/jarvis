"""Conversation controls displayed beside Jarvis's camera preview."""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .assistant import AssistantError, DeepSeekClient, SentenceBuffer
from .speech import Speech, clean_speech
from .tapo_talk import TalkError


class Conversation(ttk.Frame):
    def __init__(self, parent, api_key=None, deepgram_key=None):
        super().__init__(parent, padding=12)
        self.client = None
        self.events = queue.Queue()
        self.cancelled = threading.Event()
        self.busy = False
        self.closed = False
        self.microphone_available = False
        self.camera = None
        self.robot = None
        self.speaker_check_busy = False
        self.speaker_issue = None
        self.output_chosen = False
        self.output = tk.StringVar(value="Computer speakers")
        self.speaker_password = tk.StringVar()
        self._listen_expected = None
        self.turn_error = False
        self.turn = 0
        self.key = tk.StringVar()
        self.transcription_key = tk.StringVar()
        self.prompt = tk.StringVar()
        self.status = tk.StringVar(value="Starting speech services...")
        self.speak_enabled = tk.BooleanVar(value=True)
        self.hands_free_enabled = tk.BooleanVar(value=True)
        self.voice_status = tk.StringVar(value="Voice is starting")
        self._voice_turn = False
        self._automatic_turn = False
        self._end_after_reply = False
        self._new_visual_session = True
        self._last_interaction = -100.0
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        ttk.Label(self, text="DeepSeek V4.1 Flash", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        self.speech_label = tk.StringVar(value="Piper voice | Local transcription")
        ttk.Label(self, textvariable=self.speech_label, foreground="#a5b6c9", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=(3, 9))
        self.history = ScrolledText(self, width=32, height=12, wrap="word", font=("Segoe UI", 10),
                                   bg="#0b131c", fg="#edf3fa", insertbackground="#edf3fa", relief="flat", padx=10, pady=10)
        self.history.grid(row=2, column=0, sticky="nsew")
        self.history.tag_configure("user", foreground="#8dcaff", font=("Segoe UI Semibold", 10))
        self.history.tag_configure("assistant", foreground="#edf3fa")
        self.history.tag_configure("error", foreground="#f1b19c")
        self.history.configure(state="disabled")
        self.key_window = tk.Toplevel(self)
        self.key_window.title("Jarvis | Keys")
        self.key_window.transient(self.winfo_toplevel())
        self.key_window.resizable(False, False)
        self.key_window.protocol("WM_DELETE_WINDOW", self.key_window.withdraw)
        self.key_window.withdraw()
        self.key_panel = ttk.Frame(self.key_window, padding=14)
        self.key_panel.grid(row=0, column=0, sticky="ew")
        self.key_panel.columnconfigure(0, weight=1)
        ttk.Label(self.key_panel, text="DeepSeek API key", foreground="#a5b6c9", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        ttk.Entry(self.key_panel, textvariable=self.key, show="*").grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(self.key_panel, text="Use key", command=self.apply_key).grid(row=1, column=1, padx=(6, 0), pady=(4, 0))
        ttk.Label(self.key_panel, text="Deepgram key for transcription and voice", foreground="#a5b6c9", font=("Segoe UI", 9)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(9, 0))
        ttk.Entry(self.key_panel, textvariable=self.transcription_key, show="*").grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(self.key_panel, text="Use Flux", command=self.apply_transcription_key).grid(row=3, column=1, padx=(6, 0), pady=(4, 0))
        ttk.Label(self.key_panel, text="Speaker: Tapo account password", foreground="#a5b6c9", font=("Segoe UI", 9)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(9, 0))
        ttk.Entry(self.key_panel, textvariable=self.speaker_password, show="*").grid(row=5, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(self.key_panel, text="Connect speaker", command=self.connect_speaker).grid(row=5, column=1, padx=(6, 0), pady=(4, 0))
        self.input = ttk.Entry(self, textvariable=self.prompt)
        self.input.grid(row=4, column=0, sticky="ew", pady=(10, 8))
        self.input.bind("<Return>", self.enter)
        row = ttk.Frame(self)
        row.grid(row=5, column=0, sticky="ew")
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        self.talk_button = ttk.Button(row, text="Talk", command=self.talk, state="disabled")
        self.talk_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.send_button = ttk.Button(row, text="Send", command=self.send)
        self.send_button.grid(row=0, column=1, sticky="ew")
        self.stop_button = ttk.Button(self, text="Stop speech and movement", command=self.stop_reply)
        self.stop_button.grid(row=6, column=0, sticky="ew", pady=(7, 0))
        settings = ttk.Frame(self)
        settings.grid(row=7, column=0, sticky="ew", pady=(7, 0))
        ttk.Checkbutton(settings, text="Speak replies", variable=self.speak_enabled).pack(side="left")
        ttk.Button(settings, text="New chat", command=self.new_chat).pack(side="right")
        ttk.Button(settings, text="Keys", width=4, command=self.show_keys).pack(side="right", padx=(0, 5))
        voice_row = ttk.Frame(self)
        voice_row.grid(row=8, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(voice_row, text='Listen for "Hey Jarvis" or "Jarvis"',
                        variable=self.hands_free_enabled, command=self.toggle_hands_free).pack(anchor="w")
        ttk.Label(voice_row, textvariable=self.voice_status, foreground="#8dcaff", font=("Segoe UI", 9)).pack(anchor="w", pady=(3, 0))
        output_row = ttk.Frame(self)
        output_row.grid(row=9, column=0, sticky="ew", pady=(9, 0))
        output_row.columnconfigure(1, weight=1)
        ttk.Label(output_row, text="Voice output", font=("Segoe UI", 9)).grid(row=0, column=0, padx=(0, 7))
        selector = ttk.Combobox(output_row, textvariable=self.output, values=["Camera speaker", "Computer speakers"], state="readonly", width=18)
        selector.grid(row=0, column=1, sticky="ew")
        selector.bind("<<ComboboxSelected>>", self.select_output)
        ttk.Label(self, textvariable=self.status, wraplength=300, foreground="#a5b6c9", font=("Segoe UI", 9)).grid(row=10, column=0, sticky="ew", pady=(9, 0))
        self.privacy_text = tk.StringVar(value="Questions and selected snapshots go to DeepSeek.\nAudio stays local. Follow-ups stay open 15s.")
        ttk.Label(self, textvariable=self.privacy_text,
                  wraplength=300, foreground="#a5b6c9", font=("Segoe UI", 8)).grid(row=11, column=0, sticky="ew", pady=(7, 0))
        self.speech = Speech(lambda kind, value: self.events.put((kind, value)), deepgram_key=deepgram_key)
        if api_key:
            self.key.set(api_key)
            self.apply_key()
        else:
            self.after(100, self.show_keys)
        self.after(50, self.poll)

    def append(self, text, tag="assistant"):
        self.history.configure(state="normal")
        self.history.insert("end", text, tag)
        self.history.see("end")
        self.history.configure(state="disabled")

    def apply_key(self):
        if self.busy:
            return
        try:
            self.client = DeepSeekClient(self.key.get())
        except AssistantError as error:
            self.status.set(str(error))
            return
        self.key.set("")
        self.key_window.withdraw()
        self.status.set("Ready for questions. Speech services are starting." if not self.speech.tts_ready.is_set() else 'Ready. Say "Hey Jarvis" and ask a question.')
        self.update_buttons()

    def enter(self, event):
        self.send()
        return "break"

    def show_keys(self):
        if not self.closed:
            self.key_window.deiconify()
            self.key_window.lift()

    def apply_transcription_key(self):
        if self.busy:
            self.status.set("Wait for the reply to finish before changing transcription.")
            return
        key = self.transcription_key.get().strip()
        if not key:
            self.status.set("Enter a Deepgram API key to use Flux.")
            return
        try:
            self.speech.set_transcription_key(key)
        except ValueError as error:
            self.status.set(str(error))
            return
        self.transcription_key.set("")
        self._listen_expected = None
        self.prompt.set("")
        self.status.set("Starting Flux transcription and Colin's voice. Wake words stay local.")
        if self.client:
            self.key_window.withdraw()

    def update_buttons(self):
        pending_recognition = self._listen_expected is not None and not self.speech.listening.is_set()
        self.send_button.configure(state="normal" if self.client and not self.busy and self._listen_expected is None else "disabled")
        can_talk = self.client and self.microphone_available and self.speech.recognition_ready.is_set() and not self.busy and not pending_recognition
        self.talk_button.configure(state="normal" if can_talk else "disabled", text="Finish" if self.speech.listening.is_set() else "Talk")
        self.sync_hands_free()

    def sync_hands_free(self):
        # Speaker authentication must not disable a working microphone or the
        # follow-up window. Computer output can already play manual replies.
        ready = (self.client is not None and self.microphone_available
                 and self.speech.recognition_ready.is_set() and self.speech.tts_ready.is_set())
        self.speech.set_hands_free(self.hands_free_enabled.get() and ready)
        if not self.hands_free_enabled.get():
            self.voice_status.set("Wake word listening is off")
        elif not ready:
            self.voice_status.set("Waiting for camera, speech tools, and API key")

    def toggle_hands_free(self):
        self._listen_expected = None
        self.speech.cancel_listening()
        self.prompt.set("")
        self.sync_hands_free()
        self.update_buttons()

    def set_microphone(self, available):
        self.microphone_available = available
        self.speech.set_microphone(available)
        if not available:
            self._listen_expected = None
        self.update_buttons()

    def set_camera(self, camera):
        self.camera = camera
        self._check_speaker()

    def set_robot(self, robot):
        self.robot = robot

    def greet(self):
        if (self.client and self.robot and self.robot.greetings and self.speak_enabled.get()
                and self.speech.hands_free.is_set() and self.speech.voice_state == "wake"
                and not self.busy and self._listen_expected is None):
            self.send(automatic=True, text_override="A local detector reports that a person has arrived. Give one brief, natural greeting in character.")

    def _check_speaker(self, password=None):
        if not self.camera or self.speaker_check_busy:
            return
        self.speaker_check_busy = True
        self.status.set("Connecting the camera speaker...")
        def worker():
            try:
                speaker = self.camera.create_speaker(password)
                speaker.open()
                speaker.close()
                self.events.put(("speaker_ready", speaker))
            except TalkError as error:
                self.events.put(("speaker_error", str(error)))
        threading.Thread(target=worker, daemon=True, name="jarvis-speaker-login").start()

    def connect_speaker(self):
        if not self.camera:
            self.status.set("Connect the camera before connecting its speaker.")
            return
        password = self.speaker_password.get()
        self.speaker_password.set("")
        if not password:
            self.status.set("Enter the Tapo account password for the camera speaker.")
            return
        self._check_speaker(password)

    def select_output(self, event=None):
        try:
            destination = "camera" if self.output.get() == "Camera speaker" else "computer"
            self.speech.select_output(destination)
            self.output_chosen = True
            self.status.set("Voice output set to " + self.output.get().lower() + ".")
        except TalkError as error:
            self.output.set("Camera speaker" if self.speech.output == "camera" else "Computer speakers")
            self.status.set(str(error))

    def feed_audio(self, pcm):
        self.speech.feed(pcm)

    def talk(self):
        if self.speech.listening.is_set():
            self.speech.finish_listening()
            self.status.set("Finishing speech recognition...")
        elif self.microphone_available and not self.busy:
            identifier = self.speech.start_listening()
            if identifier:
                self._listen_expected = identifier
                self.prompt.set("")
                self.status.set("Listening through the camera. Speak, then pause or click Finish.")
        self.update_buttons()

    def send(self, voice=False, automatic=False, text_override=None):
        text = (text_override if text_override is not None else self.prompt.get()).strip()
        if not self.client or self.busy or not text or self._listen_expected is not None:
            return
        self.speech.stop_speaking()
        self.speech.set_turn_busy(True)
        self.turn += 1
        turn = self.turn
        self.cancelled = threading.Event()
        cancelled = self.cancelled
        speak_reply = self.speak_enabled.get()
        self.busy = True
        self.turn_error = False
        self._voice_turn = voice
        self._automatic_turn = automatic
        self._end_after_reply = text.lower().strip(".!? ") in {"goodbye", "bye", "that's all", "that is all", "go to sleep"}
        new_conversation = not automatic and (self._new_visual_session if voice else time.monotonic() - self._last_interaction > 15)
        self._new_visual_session = False
        self._last_interaction = time.monotonic()
        self.prompt.set("")
        self.append("Presence\n" if automatic else "You\n", "user")
        self.append("Someone arrived.\n\n" if automatic else text + "\n\n")
        self.append("Jarvis\n", "user")
        self.status.set("Waiting for DeepSeek...")
        self.update_buttons()
        client = self.client
        def worker():
            buffer = SentenceBuffer()
            spoken = self.speech.begin_reply(cancelled) if speak_reply else None
            def narrate(chunk):
                if spoken is not None:
                    for phrase in buffer.add(chunk):
                        spoken.write(clean_speech(phrase))
            try:
                for chunk in client.reply(text, cancelled, robot=self.robot, new_conversation=new_conversation,
                                          automatic=automatic,
                                          on_speech=narrate,
                                          on_action=lambda name: self.events.put(("camera_action", (turn, name)))):
                    if cancelled.is_set():
                        break
                    self.events.put(("token", (turn, chunk)))
                if not cancelled.is_set() and speak_reply:
                    for phrase in buffer.finish():
                        spoken.write(clean_speech(phrase))
            except AssistantError as error:
                self.events.put(("chat_error", (turn, str(error))))
            finally:
                if spoken is not None:
                    spoken.finish()
                self.events.put(("chat_done", (turn, client.metrics)))
        threading.Thread(target=worker, daemon=True, name="jarvis-chat").start()

    def stop_reply(self):
        self.cancelled.set()
        self._listen_expected = None
        self.speech.stop_speaking()
        if self.robot:
            self.robot.stop()
        self.speech.cancel_listening()
        self.update_buttons()
        if self.client:
            threading.Thread(target=self.client.cancel, daemon=True).start()
        if not self.busy:
            self.speech.set_turn_busy(False)
        self.status.set("Stopped.")

    def new_chat(self):
        if self.busy or self.speech.listening.is_set():
            return
        self.speech.stop_speaking()
        self.speech.reset_voice_context()
        if self.client:
            self.client.clear()
        self._new_visual_session = True
        self._last_interaction = -100.0
        self.history.configure(state="normal")
        self.history.delete("1.0", "end")
        self.history.configure(state="disabled")
        self.status.set("New conversation.")

    def poll(self):
        if self.closed:
            return
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "token":
                turn, text = value
                if turn == self.turn:
                    self.append(text)
                    self.status.set("DeepSeek is replying...")
            elif kind == "camera_action":
                turn, name = value
                if turn == self.turn:
                    labels = {"look": "Looking...", "look_at": "Centering the camera...", "capture_frame": "Taking a fresh look...",
                              "follow_person": "Updating person following...", "stop_motion": "Stopping movement...", "gesture": "Making a small gesture..."}
                    self.status.set(labels.get(name, "Checking camera action..."))
            elif kind == "speaker_ready":
                self.speaker_check_busy = False
                self.speaker_issue = None
                self.speech.set_camera_speaker(value, select=not self.output_chosen)
                self.output.set("Camera speaker" if self.speech.output == "camera" else "Computer speakers")
                if self.client is not None:
                    self.key_window.withdraw()
                self.status.set("Camera speaker connected. Ready for questions.")
                self.update_buttons()
            elif kind == "speaker_error":
                self.speaker_check_busy = False
                self.speaker_issue = value + (" Wake words remain available; replies use computer speakers."
                                              if self.speech.output == "computer" else "")
                self.status.set(self.speaker_issue)
                self.update_buttons()
            elif kind == "chat_error":
                turn, message = value
                if turn == self.turn:
                    self.turn_error = True
                    self.append("\n" + message, "error")
                    self.status.set(message)
                    if self._voice_turn and self.speak_enabled.get() and not self.cancelled.is_set():
                        self.speech.speak(message, cancelled=self.cancelled)
            elif kind == "chat_done":
                turn, metrics = value
                if turn == self.turn:
                    self.busy = False
                    self.speech.set_turn_busy(False)
                    if not self.turn_error and not self.cancelled.is_set() and not self._automatic_turn and not self._end_after_reply:
                        self.speech.open_followup(15)
                    self.append("\n\n")
                    first = metrics.get("first_text_seconds")
                    if not self.turn_error:
                        self.status.set("Stopped." if self.cancelled.is_set() else f"Ready. First text in {first:.2f}s." if first is not None else "Ready.")
                    self.update_buttons()
            elif kind in {"recognition_ready", "tts_ready"}:
                self.update_buttons()
                if self.speech.tts_ready.is_set() and self.speech.recognition_ready.is_set() and not self.busy and not self.speech.listening.is_set():
                    self.status.set(self.speaker_issue or 'Ready. Say "Hey Jarvis" and ask a question.')
            elif kind in {"recognizer_changed", "tts_provider_changed"}:
                value = self.speech.recognition_provider
                cloud = value == "Deepgram Flux"
                voice = "Colin, British male" if self.speech.tts_provider == "Deepgram Colin" else "Piper voice"
                self.speech_label.set(voice + " | " + ("Flux transcription" if cloud else "Local transcription"))
                self.privacy_text.set("Questions and selected snapshots go to DeepSeek.\n" + (
                    "Speech and reply text go to Deepgram. Wake detection stays local." if cloud else "Audio stays local. Follow-ups stay open 15s."))
                self.status.set(self.speaker_issue or ('Flux and Colin selected. Say "Hey Jarvis" and ask a question.' if cloud else "Local speech selected."))
                self.update_buttons()
            elif kind == "voice_state":
                labels = {"wake": 'Ready. Say "Hey Jarvis" or "Jarvis"',
                          "followup": "Listening for a follow-up. No wake word needed.",
                          "listening": "Listening to your question...", "thinking": "Thinking...",
                          "transcribing": "Finishing your transcription...",
                          "speaking": "Speaking through " + self.output.get().lower(),
                          "offline": "Camera microphone is reconnecting...", "muted": "Wake word listening is off"}
                if self.hands_free_enabled.get() or value == "listening":
                    self.voice_status.set(labels.get(value, value))
                self.update_buttons()
            elif kind in {"wake_detected", "followup_started"}:
                if self.speech.hands_free.is_set() and not self.busy:
                    self._listen_expected = value
                    self._new_visual_session = kind == "wake_detected"
                    self.prompt.set("")
                    self.status.set("Listening through the camera. Pause when your question is complete.")
                    self.update_buttons()
            elif kind in {"listen_timeout", "listen_cancelled"}:
                if self._listen_expected == value:
                    self._listen_expected = None
                    self.prompt.set("")
                    self.status.set('Ready. Say "Jarvis" to try again.')
                    self.update_buttons()
            elif kind == "output_idle":
                self._last_interaction = time.monotonic()
                self.update_buttons()
            elif kind == "partial":
                identifier, text = value
                if self._listen_expected == identifier:
                    self.prompt.set(text)
            elif kind == "recognized":
                identifier, text = value
                if self._listen_expected != identifier:
                    continue
                self._listen_expected = None
                self.prompt.set(text)
                self.update_buttons()
                if text:
                    normalized = text.lower().strip(".!? ")
                    if normalized in {"stop", "stop moving", "stop movement", "stop following", "stop tracking"}:
                        self.prompt.set("")
                        self.stop_reply()
                        self.speech.cancel_listening()
                    elif normalized in {"cancel", "never mind", "nevermind", "stop listening", "forget it"}:
                        self.prompt.set("")
                        self.speech.cancel_listening()
                        self.status.set('Cancelled. Say "Jarvis" when you need me.')
                    else:
                        self.send(voice=True)
                else:
                    self.speech.cancel_listening()
                    self.status.set('I did not catch that. Say "Jarvis" and try again.')
            elif kind in {"tts_error", "recognition_error"}:
                if kind == "recognition_error":
                    self._listen_expected = None
                self.status.set(value)
                self.update_buttons()
        self.after(50, self.poll)

    def close(self):
        self.closed = True
        self.cancelled.set()
        self.speech.close()
        self.speaker_password.set("")
        self.transcription_key.set("")
        if self.client:
            threading.Thread(target=self.client.cancel, daemon=True).start()
