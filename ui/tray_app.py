import threading
import logging
import os
import sys
from typing import Callable

try:
    import pystray
except (ImportError, ValueError):
    if sys.platform.startswith("linux"):
        for key in list(sys.modules.keys()):
            if key == "pystray" or key.startswith("pystray."):
                del sys.modules[key]
        os.environ["PYSTRAY_BACKEND"] = "xorg"
        import pystray
    else:
        raise

from PIL import Image, ImageDraw

logging.getLogger("pystray").setLevel(logging.CRITICAL)


class TrayApp:
    """Native system tray host for Hushline."""

    STATE_SUBTITLE = {
        "idle": "Idle",
        "listening": "Listening - will pause on speech",
        "speaking": "Speech detected - pausing music",
        "paused": "Music paused - waiting for silence",
        "calibrating": "Calibrating background noise",
        "error": "Needs attention",
    }

    def __init__(
        self,
        on_open_settings: Callable[[], None],
        on_toggle_island: Callable[[], None],
        on_toggle_pause: Callable[[], None],
        on_recalibrate: Callable[[], None],
        on_quit: Callable[[], None],
        get_color: Callable[[str], str],
    ):
        self.on_open_settings = on_open_settings
        self.on_toggle_island = on_toggle_island
        self.on_toggle_pause = on_toggle_pause
        self.on_recalibrate = on_recalibrate
        self.on_quit_cb = on_quit
        self.get_color = get_color

        self.current_state = "idle"
        self.paused = False
        self.icon = None
        self._lock = threading.RLock()

    def _make_icon_image(self, color: str) -> Image.Image:
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, 60, 60], fill=color)
        draw.ellipse([18, 18, 46, 46], fill=(0, 0, 0, 0))
        return img

    def _is_docked(self) -> bool:
        """Checks if the system tray backend is actively docked.

        Safely falls back if running in a headless or tray-less Linux session.
        """
        if not self.icon:
            return False
        # Xorg fallback engine specific check
        if hasattr(self.icon, "_systray_manager") and not self.icon._systray_manager:
            return False
        return True

    def set_state(self, state: str) -> None:
        with self._lock:
            self.current_state = state
            if not self._is_docked():
                return

            display_state = "paused" if self.paused else state
            color = self.get_color(display_state)
            self.icon.icon = self._make_icon_image(color)
            self.icon.title = f"Hushline - {display_state.capitalize()}"
            self.icon.menu = self._make_menu()
            try:
                self.icon.icon = self._make_icon_image(color)
                self.icon.title = f"Ambient - {display_state.capitalize()}"
                self.icon.menu = self._make_menu()
            except Exception:
                pass

    def _make_menu(self) -> pystray.Menu:
        state = "paused" if self.paused else self.current_state
        subtitle = self.STATE_SUBTITLE.get(state, state.capitalize())
        pause_label = "Resume" if self.paused else "Pause"
        return pystray.Menu(
            pystray.MenuItem(f"* {state.capitalize()}", None, enabled=False),
            pystray.MenuItem(subtitle, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Settings", self._open_settings),
            pystray.MenuItem("Show / Hide Island", self._toggle_island),
            pystray.MenuItem(pause_label, self._toggle_pause),
            pystray.MenuItem("Recalibrate", self._on_recalibrate),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit),
        )

    def _open_settings(self, icon=None, item=None) -> None:
        self.on_open_settings()

    def _toggle_island(self, icon=None, item=None) -> None:
        self.on_toggle_island()

    def _toggle_pause(self, icon=None, item=None) -> None:
        self.on_toggle_pause()

    def _on_recalibrate(self, icon=None, item=None) -> None:
        self.on_recalibrate()

    def _on_quit(self, icon=None, item=None) -> None:
        self.stop()
        self.on_quit_cb()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused
            if not self._is_docked():
                return
            try:
                self.icon.menu = self._make_menu()
                self.set_state(self.current_state)
            except Exception:
                pass

    def stop(self) -> None:
        if self.icon:
            try:
                self.icon.stop()
            except Exception:
                pass

    def run(self) -> None:
        self.icon = pystray.Icon(
            name="hushline",
            icon=self._make_icon_image(self.get_color("idle")),
            title="Hushline - Idle",
            menu=self._make_menu(),
        )

        # Async status check to cleanly alert the user on startup if a tray bar is missing
        def _check_dock_status():
            import time

            time.sleep(1.0)
            if not self._is_docked():
                print(
                    "  TRAY -> No system tray manager found. Running cleanly in overlay-only mode."
                )

        threading.Thread(target=_check_dock_status, daemon=True).start()

        try:
            self.icon.run()
        except Exception:
            pass
