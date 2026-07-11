import sys
import os
import tempfile

def ensure_single_instance():
    """
    Prevents multiple instances of HushLine running simultaneously.
    Uses a lock file in the temp directory.
    If another instance is already running, exit immediately.
    """
    lock_file = os.path.join(tempfile.gettempdir(), "hushline.lock")

    try:
        if sys.platform == "win32":
            import msvcrt
            lock = open(lock_file, 'w')
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            lock = open(lock_file, 'w')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # write our PID so we can identify the process
        lock.write(str(os.getpid()))
        lock.flush()

        # keep the lock file handle open for the lifetime of the process
        sys._hushline_lock = lock

    except (IOError, OSError):
        # lock already held — another instance is running
        print("HushLine is already running.")
        sys.exit(0)

ensure_single_instance()


# Add _MEIPASS directory to PATH on Windows so ctypes can locate PortAudio DLL (for sounddevice/PyInstaller)
if getattr(sys, 'frozen', False) and sys.platform == 'win32':
    os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')

import asyncio
import threading
import time

import numpy as np
import sounddevice as sd

import config
from audio.calibrator import Calibrator
from audio.noise_tracker import NoiseTracker
from audio.vad_engine import VADEngine
from intelligence.confidence_engine import ConfidenceEngine
from intelligence.event_queue import EventQueue
from intelligence.state_machine import StateMachine
from media.media_controller import MediaController
from ui.qt_overlay import QtOverlayRuntime
from ui.settings_store import SettingsStore
from ui.tray_app import TrayApp


class AmbientApp:
    def __init__(self):
        self.settings_store = SettingsStore()
        self.settings = self.settings_store.settings

        self.event_queue = EventQueue()
        self.media_controller = MediaController()
        self.state_machine = StateMachine(self.media_controller)
        self.noise_tracker = NoiseTracker()
        self.paused = False
        self.vad_engine = VADEngine(
            aggressiveness=int(self.settings["vad_aggressiveness"])
        )
        self.confidence_engine = ConfidenceEngine(
            speech_trigger=int(self.settings["speech_trigger"]),
            silence_trigger=int(self.settings["silence_trigger"]),
        )
        self.state_machine.resume_delay = float(self.settings["resume_delay"])
        self.running = True
        self._stopping = False

        self.overlay = QtOverlayRuntime(
            settings=self.settings,
            media_controller=self.media_controller,
            on_open_settings=self.open_settings,
            on_save_settings=self.save_settings,
            on_preview_settings=self.apply_settings,
            on_reset_appearance=self.reset_appearance,
        )
        self.tray = TrayApp(
            on_open_settings=self.open_settings,
            on_toggle_island=self.toggle_island,
            on_toggle_pause=self.toggle_paused,
            on_recalibrate=self.run_calibration_async,
            on_quit=self.stop,
            get_color=self.get_state_color,
        )

    def get_state_color(self, state: str) -> str:
        colors = self.settings.get("state_colors", {})
        return colors.get(state, colors.get("idle", "#95A5A6"))

    def open_settings(self):
        if self._stopping:
            return
        self.overlay.open_settings()

    def save_settings(self, settings):
        self.settings = self.settings_store.save(settings)
        self.apply_settings(self.settings)
        return self.settings

    def reset_appearance(self):
        self.settings = self.settings_store.reset_appearance()
        self.apply_settings(self.settings)
        return self.settings

    def toggle_island(self):
        next_settings = dict(self.settings)
        next_settings["show_island"] = not bool(self.settings.get("show_island", True))
        self.save_settings(next_settings)

    def run_calibration(self):
        self.overlay.set_state(
            "calibrating",
            {
                "notification": {
                    "title": "Calibrating",
                    "body": "Stay quiet for a moment",
                }
            },
        )

        def on_phase(phase, remaining):
            if phase == "silence":
                title = "Calibration: stay silent"
                body = f"Measuring room noise - {remaining}s"
            else:
                title = "Calibration: speak now"
                body = f"Speak normally - {remaining}s"
            self.overlay.set_state(
                "calibrating",
                {"notification": {"title": title, "body": body}},
            )

        calibrator = Calibrator()
        while self.running and not calibrator.run(
            self.noise_tracker, on_phase=on_phase
        ):
            time.sleep(1)
        if self.running:
            self.overlay.set_state("listening")
            self.tray.set_state("listening")

    def run_calibration_async(self):
        threading.Thread(target=self.run_calibration, daemon=True).start()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio warning:", status)

        volume = np.sqrt(np.mean(indata[:, 0] ** 2)) * 1000
        threshold = self.noise_tracker.get_threshold()

        if volume < threshold * 0.5:
            self.noise_tracker.update(volume)

        is_speech = self.vad_engine.detect_speech(indata, threshold)
        event = self.confidence_engine.process(is_speech)

        if event:
            self.event_queue.push(event)

    def apply_settings(self, settings):
        self.settings = settings
        self.state_machine.resume_delay = float(settings["resume_delay"])
        self.confidence_engine.speech_trigger = int(settings["speech_trigger"])
        self.confidence_engine.silence_trigger = int(settings["silence_trigger"])

        new_agg = int(settings["vad_aggressiveness"])
        if new_agg != self.vad_engine.aggressiveness:
            self.vad_engine = VADEngine(aggressiveness=new_agg)

        self.overlay.apply_settings(settings)
        self.tray.set_state(self.overlay.overlay.state)
        if self.paused:
            self.tray.set_paused(True)
        print("  SETTINGS -> applied")

    def start_background_runtime(self):
        threading.Thread(target=self._runtime_loop, daemon=True).start()
        threading.Thread(target=self._media_poll_loop, daemon=True).start()
        threading.Thread(target=self.tray.run, daemon=True).start()
        self.run_calibration_async()

    def _runtime_loop(self):
        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=config.FRAME_SIZE,
            channels=1,
            dtype="float32",
            callback=self.audio_callback,
        )

        print("System ready. Tray icon and Qt island active.\n")
        try:
            with stream:
                while self.running:
                    event = self.event_queue.pop()
                    if self.paused:
                        time.sleep(0.01)
                        continue
                    if event:
                        self.state_machine.process_event(event)
                        self._handle_event_state(event)
                    self.state_machine.tick()
                    time.sleep(0.01)
        except Exception as exc:
            print(f"Runtime stopped: {exc}")
            self.overlay.set_state(
                "error",
                {"notification": {"title": "Runtime error", "body": str(exc)}},
            )

    def _media_poll_loop(self):
        while self.running:
            self.push_media_info()
            time.sleep(2)

    def push_media_info(self):
        try:
            info = asyncio.run(self.media_controller.get_media_info())
        except Exception:
            info = {
                "title": "Nothing playing",
                "artist": "-",
                "playing": False,
                "progress": 0,
            }
        self.overlay.set_media(info)

    def _handle_event_state(self, event: str):
        if event == "user_speaking":
            state = "speaking"
            payload = {
                "notification": {"title": "Music paused", "body": "Speech detected"}
            }
        elif event == "user_silent":
            state = "listening"
            payload = {
                "notification": {"title": "Listening", "body": "Silence detected"}
            }
        elif event == "user_present":
            state = "listening"
            payload = {}
        elif event == "user_absent":
            state = "paused"
            payload = {"notification": {"title": "Music paused", "body": "User absent"}}
        else:
            return

        self.overlay.set_state(state, payload)
        self.tray.set_state(state)
        self.push_media_info()

    def toggle_paused(self):
        self.set_paused(not self.paused)

    def set_paused(self, paused: bool):
        self.paused = paused
        self.state_machine.set_paused(paused)
        if paused:
            self.overlay.set_state(
                "paused",
                {"notification": {"title": "Paused", "body": "Tray pause mode"}},
            )
            self.tray.set_paused(True)
        else:
            restored_state = self.state_machine.state or "idle"
            self.overlay.set_state(restored_state)
            self.tray.set_paused(False)
            self.tray.set_state(restored_state)

    def stop(self):
        if self._stopping:
            return
        self._stopping = True
        self.running = False
        try:
            self.tray.stop()
        except Exception:
            pass
        try:
            self.overlay.stop()
        except Exception:
            pass

        def force_exit():
            time.sleep(0.6)
            os._exit(0)

        threading.Thread(target=force_exit, daemon=True).start()

    def run(self):
        import signal
        from PySide6.QtCore import QTimer

        # handle KeyboardInterrupt, yes make ctrl+c work
        signal.signal(signal.SIGINT, lambda sig, frame: self.stop())

        # force qt event to refresh every 200ms.
        self.sigint_timer = QTimer()
        self.sigint_timer.start(200)
        self.sigint_timer.timeout.connect(lambda: None)  # dummy target

        self.start_background_runtime()
        self.overlay.run()


def main():
    AmbientApp().run()


if __name__ == "__main__":
    main()
