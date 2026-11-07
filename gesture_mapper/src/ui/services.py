"""
Engine: the one object that owns the camera, hands it between the preview
pump and the live control loop, and fronts every mutation the pages make
(gesture CRUD, mapping saves, settings).

Threading model:
  main thread   — all Qt, all Engine method calls
  pump thread   — detector.next_gestures() at camera rate while previewing;
                  parks (and signals idle) while the control loop owns the
                  camera reads
  control thread — run.main(stop_event=…, detector=…), the real product

The camera is opened exactly once per process; run.main's exit closes only
the capture (detector.close() keeps the landmarker), so the pump reopens it
and preview resumes seamlessly after Stop.
"""

import threading
import traceback

from PySide6.QtCore import QObject, QTimer, Signal

import run                      # the engine of the whole product
from detector import GestureDetector
from context import get_active_context
from normalizer import normalize
import gesture_library
import recognizer
import mapper
import camera as camera_mod
import handicon

import numpy as np


class Engine(QObject):
    camera_ready = Signal()
    camera_failed = Signal(str)
    control_changed = Signal(bool)          # True = live control running
    context_changed = Signal(str)
    library_changed = Signal()
    mappings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector: GestureDetector | None = None
        self._camera_error: str | None = None
        self._starting_camera = False

        self._pump_active = threading.Event()
        self._pump_idle = threading.Event()
        self._pump_idle.set()
        self._pump_quit = False
        self._pump_thread: threading.Thread | None = None

        self._control_thread: threading.Thread | None = None
        self._stop_ev: threading.Event | None = None
        self._context = "default"

        self._ctx_timer = QTimer(self)
        self._ctx_timer.setInterval(600)
        self._ctx_timer.timeout.connect(self._poll_context)
        self._ctx_timer.start()

        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(500)
        self._watch_timer.timeout.connect(self._watch_control)

    # ── Camera / preview ─────────────────────────────────────────────────────

    def start_camera(self) -> None:
        """Open the camera off the main thread (first open triggers the macOS
        permission prompt and can block)."""
        if self.detector or self._starting_camera:
            return
        self._starting_camera = True

        def target():
            try:
                det = GestureDetector()
                det.open()
            except Exception as e:
                self._camera_error = str(e)
                self._starting_camera = False
                self.camera_failed.emit(str(e))
                return
            self.detector = det
            self._starting_camera = False
            self._pump_quit = False
            self._pump_thread = threading.Thread(target=self._pump, daemon=True)
            self._pump_active.set()
            self._pump_thread.start()
            self.camera_ready.emit()

        threading.Thread(target=target, daemon=True).start()

    def _pump(self):
        det = self.detector
        while not self._pump_quit:
            if self._pump_active.is_set():
                self._pump_idle.clear()
                try:
                    if not det.is_open:
                        det.open()
                    det.next_gestures()
                except Exception:
                    self._pump_idle.set()
                    self._pump_active.clear()
            else:
                self._pump_idle.set()
                self._pump_active.wait(0.1)

    def camera_status_text(self) -> str:
        if self._camera_error:
            return ("Camera unavailable.\nSystem Settings → Privacy & Security "
                    "→ Camera, then relaunch.")
        if self.detector is None:
            return "Starting camera…"
        return "Waiting for frames…"

    def camera_name(self) -> str:
        try:
            cams = camera_mod.list_cameras()
            idx = camera_mod.index_for_name()
            for i, name in cams:
                if i == idx:
                    return name
        except Exception:
            pass
        return "Camera"

    # ── Live control ─────────────────────────────────────────────────────────

    @property
    def control_running(self) -> bool:
        return self._control_thread is not None and self._control_thread.is_alive()

    def start_control(self) -> bool:
        if self.control_running or self.detector is None:
            return False
        # Hand the camera to the control loop: park the pump first so exactly
        # one thread reads frames.
        self._pump_active.clear()
        self._pump_idle.wait(timeout=2.0)

        self._stop_ev = threading.Event()

        def target():
            try:
                run.main(stop_event=self._stop_ev, detector=self.detector)
            except Exception:
                traceback.print_exc()
                try:
                    run._release_all()
                except Exception:
                    pass

        self._control_thread = threading.Thread(target=target, daemon=True)
        self._control_thread.start()
        self._watch_timer.start()
        self.control_changed.emit(True)
        return True

    def stop_control(self) -> None:
        if not self.control_running:
            return
        self._stop_ev.set()
        self._control_thread.join(timeout=5.0)
        self._control_thread = None
        self._watch_timer.stop()
        self._pump_active.set()          # pump reopens the camera and resumes
        self.control_changed.emit(False)

    def _watch_control(self):
        """Crash detection: if the control thread died without stop_control(),
        recover the preview and tell the UI."""
        if self._control_thread is not None and not self._control_thread.is_alive():
            self._control_thread = None
            self._watch_timer.stop()
            self._pump_active.set()
            self.control_changed.emit(False)

    def shutdown(self):
        if self.control_running:
            self.stop_control()
        self._pump_quit = True
        self._pump_active.set()
        if self._pump_thread:
            self._pump_thread.join(timeout=2.0)
        if self.detector:
            self.detector.close()

    # ── Context ──────────────────────────────────────────────────────────────

    @property
    def context(self) -> str:
        return self._context

    def _poll_context(self):
        try:
            if self.control_running:
                with run._ctx_lock:
                    ctx = run._shared_ctx[0]
            else:
                ctx = get_active_context()
        except Exception:
            return
        if ctx != self._context:
            self._context = ctx
            self.context_changed.emit(ctx)

    # ── Gesture library ──────────────────────────────────────────────────────

    def gesture_names(self) -> list[str]:
        return gesture_library.list_gestures()

    def capture_sample(self) -> np.ndarray | None:
        """One normalized sample from the hand currently in view (first hand)."""
        det = self.detector
        if det is None or not det.last_hands:
            return None
        _, lms = det.last_hands[0]
        return normalize(lms)

    def save_gesture(self, name: str, samples: list[np.ndarray]) -> None:
        template = np.mean(samples, axis=0)
        gesture_library.add(name, template)
        recognizer.reload_library()
        handicon.reload()
        self.library_changed.emit()

    def delete_gesture(self, name: str) -> None:
        gesture_library.delete(name)
        recognizer.reload_library()
        handicon.reload()
        self.library_changed.emit()

    # ── Mappings ─────────────────────────────────────────────────────────────

    def mappings(self) -> dict:
        return mapper.tables()

    def save_mappings(self, tables: dict) -> None:
        mapper.save_tables(tables)
        self.mappings_changed.emit()

    def restore_default_mappings(self) -> None:
        mapper.save_tables(mapper.default_tables())
        self.mappings_changed.emit()

    def bindings_for_gesture(self, gesture: str) -> list[tuple[str, str, str]]:
        """[(context, hand, command_name)] everywhere this gesture is bound."""
        out = []
        for ctx, table in mapper.tables().items():
            for (hand, g), cmd in table.items():
                if g == gesture:
                    out.append((ctx, hand, cmd.name))
        return out

    # ── Settings ─────────────────────────────────────────────────────────────

    def get_settings(self) -> dict:
        return run._load_settings()

    def update_settings(self, **kwargs) -> None:
        s = run._load_settings()
        s.update(kwargs)
        run._save_settings(s)

    # ── Permissions ──────────────────────────────────────────────────────────

    @staticmethod
    def camera_permission() -> bool | None:
        """True granted, False denied/restricted, None not yet asked."""
        try:
            import AVFoundation
            st = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
                AVFoundation.AVMediaTypeVideo)
            return {0: None, 1: False, 2: False, 3: True}.get(st)
        except Exception:
            return None

    @staticmethod
    def input_permission() -> bool:
        """Accessibility / input-posting permission (what the executor needs)."""
        try:
            import Quartz
            return bool(Quartz.CGPreflightPostEventAccess())
        except Exception:
            return False

    @staticmethod
    def request_input_permission() -> None:
        try:
            import Quartz
            Quartz.CGRequestPostEventAccess()
        except Exception:
            pass
