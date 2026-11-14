from __future__ import annotations

import threading
import traceback

from PySide6 import QtCore

from whistlemamba.inference import (
    InferenceCancelled,
    InferenceSettings,
    WhistleMambaPredictor,
)
from whistlemamba.project import AnalysisProject


class InferenceWorker(QtCore.QObject):
    progress = QtCore.Signal(object)
    overlay = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    cancelled = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        audio_path: str,
        config_path: str,
        checkpoint_path: str,
        project: AnalysisProject,
        settings: InferenceSettings,
        resume_path: str,
    ):
        super().__init__()
        self.audio_path = audio_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.project = project
        self.settings = settings
        self.resume_path = resume_path
        self._cancel = threading.Event()

    @QtCore.Slot()
    def request_cancel(self) -> None:
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            predictor = WhistleMambaPredictor(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                settings=self.settings,
            )
            result = predictor.analyze_file(
                self.audio_path,
                project=self.project,
                progress=self.progress.emit,
                cancelled=self._cancel.is_set,
                overlay=self.overlay.emit,
                resume_path=self.resume_path,
            )
            self.completed.emit(result)
        except InferenceCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception:  # noqa: BLE001 - worker boundary serializes every failure for the UI
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()
