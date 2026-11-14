"""Native Qt front end, local hidden WSL scientific worker; no network transport."""
import json
import os
import subprocess
import tempfile
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6 import QtCore
from whistlemamba.project import AnalysisProject


@dataclass(frozen=True)
class InferenceSettings:
    # Wire representation. The scientific worker constructs and validates the
    # original InferenceSettings before loading a model or touching resume state.
    device: str = "auto"
    batch_size: int = 1
    stride_seconds: float | None = None
    tracker_mode: str = "learned"
    presence_gating: bool = False
    presence_threshold: float = 0.5
    deterministic: bool = True


def mounted(p):
    p = Path(p).resolve()
    if len(p.drive) != 2 or p.drive[1] != ":":
        raise ValueError("Select files on a local Windows drive.")
    return "/mnt/" + p.drive[0].lower() + "/" + "/".join(p.parts[1:])


class InferenceWorker(QtCore.QObject):
    progress = QtCore.Signal(object)
    overlay = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    cancelled = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, *, audio_path, config_path, checkpoint_path, project, settings, resume_path):
        super().__init__()
        self.paths = dict(audio=audio_path, config=config_path, checkpoint=checkpoint_path, resume=resume_path)
        self.project, self.settings = project, settings
        self._lock = threading.Lock()
        self._directory = None
        self._cancel = False

    def request_cancel(self):
        with self._lock:
            self._cancel = True
            if self._directory is not None:
                (self._directory / "cancel").touch()

    @QtCore.Slot()
    def run(self):
        try:
            app = Path(__file__).resolve().parents[1]
            cfg = json.loads((app.parent / "LocalRuntime.json").read_text(encoding="utf-8-sig"))
            with tempfile.TemporaryDirectory(prefix="whistlemamba-native-") as tmp:
                root = Path(tmp)
                with self._lock:
                    self._directory = root
                    if self._cancel:
                        (root / "cancel").touch()
                try:
                    self._execute(app, cfg, root)
                finally:
                    with self._lock:
                        self._directory = None
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    def _execute(self, app, cfg, root):
        paths = {k: mounted(v) for k, v in self.paths.items()}
        (root / "request.json").write_text(json.dumps(dict(paths=paths, settings=asdict(self.settings),
                project=self.project.to_dict())), encoding="utf-8")
        cmd = [str(Path(os.environ["SystemRoot"]) / "System32/wsl.exe"), "-d", cfg["distribution"],
               "--cd", mounted(app), "--exec", "/usr/bin/env", "PYTHONPATH=.", "PYTHONDONTWRITEBYTECODE=1",
               cfg["linux_python"], "-u", "-m", "whistlemamba_studio.windows_inference", mounted(root)]
        result = None
        with (root / "worker.log").open("wb") as log, subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW) as proc:
            for line in proc.stdout:
                if not line.startswith("STUDIO:"):
                    continue
                e = json.loads(line[7:])
                if e["event"] == "progress":
                    self.progress.emit(SimpleNamespace(**e["value"]))
                elif e["event"] == "overlay":
                    name = "overlay-" + str(int(e["index"])) + ".npz"
                    with np.load(root / name, allow_pickle=False) as data:
                        self.overlay.emit(SimpleNamespace(**{k: data[k] for k in data.files}))
                    (root / name).unlink()
                elif e["event"] in ("completed", "cancelled"):
                    result = e
            rc = proc.wait()
        if rc or result is None:
            raise RuntimeError((root / "worker.log").read_text(errors="replace")[-4000:] or f"Model worker exited ({rc}).")
        if result["event"] == "cancelled":
            self.cancelled.emit("Analysis cancelled; resume state preserved.")
        else:
            p = AnalysisProject.load(root / "result.json")
            if (not p.analysis_complete or p.audio != self.project.audio or p.project_id != self.project.project_id):
                raise ValueError("Analysis result does not match this recording and project.")
            self.completed.emit(SimpleNamespace(project=p, **result["counts"]))
