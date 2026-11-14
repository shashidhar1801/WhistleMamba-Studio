from __future__ import annotations

import logging
import os
from importlib.resources import files
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from PySide6 import QtMultimedia
except ImportError:  # Linux installations may omit the system PulseAudio library.
    QtMultimedia = None  # type: ignore[assignment]

from whistlemamba.editing import ContourEditor
from whistlemamba.export import export_bundle
from whistlemamba.input_mode import LIMITED_NOTICE, rate_supported
from whistlemamba.project import AnalysisProject, Contour, ContourPoint

from .spectrogram_view import SpectrogramView
if os.name == "nt":
    from .windows_worker import InferenceSettings, InferenceWorker
else:
    from whistlemamba.inference import InferenceSettings
    from .worker import InferenceWorker
from .recording_browser import Welcome, browser, local_path, ROLE

LOGGER = logging.getLogger(__name__)


def packaged_config_path() -> str:
    return str(
        files("whistlemamba.resources").joinpath("csr01_inference.json")
    )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        initial_wav: str | None = None,
        initial_project: str | None = None,
        initial_config: str | None = None,
        initial_checkpoint: str | None = None,
    ):
        super().__init__()
        self.setWindowTitle("WhistleMamba Studio")
        self.resize(1250, 790)
        self.setAcceptDrops(True)
        self.prefs = QtCore.QSettings("WhistleMamba", "Studio")
        self.project: AnalysisProject | None = None
        self.editor: ContourEditor | None = None
        self.project_path: Path | None = None
        self.dirty = False
        self.worker: InferenceWorker | None = None
        self.worker_thread: QtCore.QThread | None = None

        self.view = SpectrogramView()
        self.pages = QtWidgets.QStackedWidget()
        self.home = Welcome()
        self.pages.addWidget(self.home)
        self.pages.addWidget(self.view)
        self.setCentralWidget(self.pages)
        self._create_playback()
        self._create_actions()
        self._create_menus()
        self._create_audio_dock()
        self._create_analysis_dock(initial_config, initial_checkpoint)
        self._create_overlay_dock()
        self._create_editor_dock()
        self._create_queue_dock()
        self._create_status_bar()
        self.home.browse.connect(self._choose_audio)
        self.home.folder.connect(self._choose_directory)
        self.home.paste.connect(self._paste_path)
        self.home.project.connect(self._choose_project)
        self.panels = self.findChildren(QtWidgets.QDockWidget)
        self.panel_menu = self.menuBar().addMenu("&View")
        self.full_layout_action = self.panel_menu.addAction("Restore full review layout")
        self.full_layout_action.triggered.connect(self._full_review_layout)
        self.panel_menu.addSeparator()
        for dock in self.panels:
            self.panel_menu.addAction(dock.toggleViewAction())
            dock.hide()
        self.run_button.setEnabled(False)
        self.view.contour_selected.connect(self._select_contour_id)

        if initial_project:
            self.open_project(initial_project)
        elif initial_wav:
            self.open_audio(initial_wav)

    def _create_playback(self) -> None:
        self.player = None
        self.audio_output = None
        if QtMultimedia is None:
            return
        self.audio_output = QtMultimedia.QAudioOutput(self)
        self.player = QtMultimedia.QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.6)
        self.player.positionChanged.connect(self._playback_position_changed)
        self.player.durationChanged.connect(self._playback_duration_changed)
        self.view.cursor.sigPositionChangeFinished.connect(self._cursor_seek)

    def _create_actions(self) -> None:
        self.open_wav_action = QtGui.QAction("Open WAV files…", self, shortcut=QtGui.QKeySequence.Open)
        self.open_wav_action.triggered.connect(self._choose_audio)
        self.open_directory_action = QtGui.QAction("Choose folder…", self)
        self.open_directory_action.triggered.connect(self._choose_directory)
        self.open_project_action = QtGui.QAction("Open project…", self, shortcut="Ctrl+Shift+O")
        self.open_project_action.triggered.connect(self._choose_project)
        self.save_action = QtGui.QAction("Save", self, shortcut=QtGui.QKeySequence.Save)
        self.save_action.triggered.connect(self.save_project)
        self.save_as_action = QtGui.QAction("Save as…", self, shortcut="Ctrl+Shift+S")
        self.save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        self.export_action = QtGui.QAction("Export bundle…", self, shortcut="Ctrl+E")
        self.export_action.triggered.connect(self._choose_export)
        self.quit_action = QtGui.QAction("Quit", self, shortcut=QtGui.QKeySequence.Quit)
        self.quit_action.triggered.connect(self.close)
        self.undo_action = QtGui.QAction("Undo", self, shortcut=QtGui.QKeySequence.Undo)
        self.undo_action.triggered.connect(self._undo)
        self.redo_action = QtGui.QAction("Redo", self, shortcut=QtGui.QKeySequence.Redo)
        self.redo_action.triggered.connect(self._redo)
        self.play_action = QtGui.QAction("Play/Pause", self, shortcut="Space")
        self.play_action.triggered.connect(self._toggle_playback)
        if self.player is None:
            self.play_action.setEnabled(False)
            self.play_action.setToolTip("Playback unavailable: Qt multimedia system library is missing")

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addActions(
            [
                self.open_wav_action,
                self.open_directory_action,
                self.open_project_action,
                self.save_action,
                self.save_as_action,
                self.export_action,
            ]
        )
        file_menu.addSeparator()
        file_menu.addAction("Choose folder including subfolders…", self._choose_tree)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addActions([self.undo_action, self.redo_action])
        playback_menu = self.menuBar().addMenu("&Playback")
        playback_menu.addAction(self.play_action)
        toolbar = self.addToolBar("Recordings")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_wav_action)
        toolbar.addAction(self.open_directory_action)
        toolbar.addAction("Paste path…",self._paste_path)
        toolbar.addSeparator()
        toolbar.addAction(self.play_action)
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setMaximumWidth(200)
        self.position_slider.sliderReleased.connect(self._slider_seek)
        toolbar.addWidget(self.position_slider)

    def _create_audio_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Audio / Project", self)
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.audio_path_label = QtWidgets.QLabel("No WAV loaded")
        self.audio_path_label.setWordWrap(True)
        self.audio_path_label.setMaximumWidth(320)
        self.hash_label = QtWidgets.QLabel("—")
        self.hash_label.setWordWrap(True)
        self.hash_label.setMaximumWidth(320)
        self.hash_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.rate_label = QtWidgets.QLabel("—")
        self.channels_label = QtWidgets.QLabel("—")
        self.duration_label = QtWidgets.QLabel("—")
        self.project_label = QtWidgets.QLabel("Unsaved")
        self.project_label.setWordWrap(True)
        form.addRow("WAV", self.audio_path_label)
        form.addRow("SHA-256", self.hash_label)
        form.addRow("Sample rate", self.rate_label)
        form.addRow("Channels", self.channels_label)
        form.addRow("Duration", self.duration_label)
        form.addRow("Project", self.project_label)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _path_row(self, value: str, chooser) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit]:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit(value)
        button = QtWidgets.QPushButton("…")
        button.setMaximumWidth(34)
        button.clicked.connect(chooser)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget, edit

    def _create_analysis_dock(self, config: str | None, checkpoint: str | None) -> None:
        dock = QtWidgets.QDockWidget("Analysis", self)
        self.analysis_dock = dock
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        config_row, self.config_edit = self._path_row(
            config or packaged_config_path(), self._choose_config
        )
        checkpoint_default = checkpoint or os.environ.get("WHISTLEMAMBA_CHECKPOINT", "")
        checkpoint_row, self.checkpoint_edit = self._path_row(
            checkpoint_default, self._choose_checkpoint
        )
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["cuda"])
        self.batch_spin = QtWidgets.QSpinBox(minimum=1, maximum=16, value=1)
        self.stride_spin = QtWidgets.QDoubleSpinBox(minimum=0.1, maximum=4.0, value=2.0)
        self.stride_spin.setSuffix(" s")
        self.stride_spin.setEnabled(False)
        self.tracker_combo = QtWidgets.QComboBox()
        self.tracker_combo.addItems(["learned", "both", "heuristic"])
        self.presence_check = QtWidgets.QCheckBox("Off (frozen CSR01 setting)")
        self.presence_check.setChecked(False)
        self.presence_check.setEnabled(False)
        self.run_button = QtWidgets.QPushButton("Run analysis")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        self.progress_bar = QtWidgets.QProgressBar(minimum=0, maximum=1000)
        self.progress_label = QtWidgets.QLabel("Idle")
        advanced = QtWidgets.QWidget()
        advanced_form = QtWidgets.QFormLayout(advanced)
        for name,value in [("Configuration",config_row),("Model weights",checkpoint_row),("Device",self.device_combo),("Batch size",self.batch_spin),("Window stride",self.stride_spin),("Tracker",self.tracker_combo),("Presence gate",self.presence_check)]:
            advanced_form.addRow(name,value)
        self.presence_check.setText("Off")
        more = QtWidgets.QCheckBox("Show model settings")
        self.model_settings_toggle = more
        more.toggled.connect(advanced.setVisible)
        advanced.hide()
        form.addRow(more)
        form.addRow(advanced)
        form.addRow(buttons)
        form.addRow(self.progress_bar)
        form.addRow(self.progress_label)
        self.run_button.clicked.connect(self._start_inference)
        self.cancel_button.clicked.connect(self._cancel_inference)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _create_overlay_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Display", self)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        self.centerline_check = QtWidgets.QCheckBox("Centerline probability")
        self.contours_check = QtWidgets.QCheckBox("Contours")
        self.endpoints_check = QtWidgets.QCheckBox("Endpoints")
        self.junctions_check = QtWidgets.QCheckBox("Junctions")
        for check in (
            self.centerline_check,
            self.contours_check,
            self.endpoints_check,
            self.junctions_check,
        ):
            check.setChecked(True)
        self.fmin_spin = QtWidgets.QDoubleSpinBox(minimum=0, maximum=200_000, value=4_000)
        self.fmax_spin = QtWidgets.QDoubleSpinBox(minimum=1, maximum=200_000, value=96_000)
        self.dynamic_spin = QtWidgets.QDoubleSpinBox(minimum=20, maximum=160, value=80)
        for spin in (self.fmin_spin, self.fmax_spin):
            spin.setSuffix(" Hz")
            spin.setDecimals(0)
        self.dynamic_spin.setSuffix(" dB")
        layout.addRow(self.centerline_check)
        layout.addRow(self.contours_check)
        layout.addRow(self.endpoints_check)
        layout.addRow(self.junctions_check)
        layout.addRow("Minimum", self.fmin_spin)
        layout.addRow("Maximum", self.fmax_spin)
        layout.addRow("Dynamic range", self.dynamic_spin)
        for check in (
            self.centerline_check,
            self.contours_check,
            self.endpoints_check,
            self.junctions_check,
        ):
            check.toggled.connect(self._update_overlay_visibility)
        self.fmin_spin.editingFinished.connect(self._update_frequency_range)
        self.fmax_spin.editingFinished.connect(self._update_frequency_range)
        self.dynamic_spin.valueChanged.connect(self.view.set_dynamic_range)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _create_editor_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Contour editor", self)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.contour_list = QtWidgets.QListWidget()
        self.contour_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.contour_list.itemSelectionChanged.connect(self._contour_selection_changed)
        layout.addWidget(self.contour_list, 1)
        point_form = QtWidgets.QFormLayout()
        self.point_index = QtWidgets.QSpinBox(minimum=0, maximum=0)
        self.point_time = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e9, decimals=6)
        self.point_time.setSuffix(" s")
        self.point_frequency = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e7, decimals=1)
        self.point_frequency.setSuffix(" Hz")
        point_form.addRow("Point", self.point_index)
        point_form.addRow("Time", self.point_time)
        point_form.addRow("Frequency", self.point_frequency)
        layout.addLayout(point_form)
        self.point_index.valueChanged.connect(self._point_selection_changed)
        point_buttons = QtWidgets.QHBoxLayout()
        for label, slot in (
            ("Move", self._move_point),
            ("Add", self._add_point),
            ("Delete", self._delete_point),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            point_buttons.addWidget(button)
        layout.addLayout(point_buttons)
        contour_buttons = QtWidgets.QGridLayout()
        actions = (
            ("New", self._new_contour),
            ("Delete contour", self._delete_contour),
            ("Split here", self._split_contour),
            ("Merge selected", self._merge_contours),
            ("Accept", lambda: self._set_review("accepted")),
            ("Reject", lambda: self._set_review("rejected")),
        )
        for index, (label, slot) in enumerate(actions):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            contour_buttons.addWidget(button, index // 2, index % 2)
        layout.addLayout(contour_buttons)
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Reviewer notes")
        self.notes_edit.setMaximumHeight(90)
        notes_button = QtWidgets.QPushButton("Save notes")
        notes_button.clicked.connect(self._save_notes)
        layout.addWidget(self.notes_edit)
        layout.addWidget(notes_button)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _create_queue_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Recordings", self)
        self.recordings_dock = dock
        self.queue_list = QtWidgets.QListWidget()
        self.queue_list.itemClicked.connect(self._open_list_item)
        self.queue_list.itemActivated.connect(self._open_list_item)
        self.queue_list.setAccessibleName("Recordings — click a name or press Enter to open")
        self.queue_list.setMinimumWidth(180)
        self.queue_list.setMaximumHeight(120)
        dock.setWidget(self.queue_list)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _create_status_bar(self) -> None:
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready to open a recording")

    def _choose_audio(self) -> None:
        paths = browser(self, False, settings=self.prefs)
        if paths: self._intake(paths)

    def _choose_directory(self) -> None:
        paths = browser(self, True, settings=self.prefs)
        if paths: self._intake(paths)

    def _choose_tree(self) -> None:
        paths = browser(self, True, settings=self.prefs)
        if paths: self._intake(paths, recursive=True)

    def _paste_path(self):
        value, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Open by path",
            "Paste a Windows or Linux file or folder path. For several recordings, use one path per line:")
        if ok and value.strip(): self._intake(value.splitlines())

    def dragEnterEvent(self,event):
        if self.worker is None and event.mimeData().hasUrls() and all(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self,event):
        if event.mimeData().hasUrls() and all(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()
            self._intake([u.toLocalFile() for u in event.mimeData().urls()])

    def _list_item(self, path):
        for i in range(self.queue_list.count()):
            item=self.queue_list.item(i)
            if item.data(ROLE)==str(path): return item
        item=QtWidgets.QListWidgetItem(path.name)
        item.setData(ROLE,str(path));item.setToolTip(str(path))
        self.queue_list.addItem(item)
        return item

    def _open_list_item(self,item):
        if not item: return
        if self.project and self.project.audio.path == item.data(ROLE): return
        self.open_audio(item.data(ROLE))
        if self.project: self.queue_list.setCurrentItem(self._list_item(Path(self.project.audio.path)))

    def _intake(self, values, recursive=False):
        if self.worker is not None:
            self.status.showMessage("Finish or cancel the current analysis before switching recordings."); return
        paths=[]
        try:
            for value in values:
                if not str(value).strip(): continue
                path=local_path(value).resolve()
                if path.is_dir():
                    files = path.rglob("*") if recursive else path.iterdir()
                    paths.extend(sorted((f for f in files if f.suffix.casefold()==".wav" and f.is_file()),key=lambda p:str(p).casefold()))
                elif path.is_file() and path.suffix.casefold()==".wav": paths.append(path)
                else: raise ValueError(f"Not a local WAV file or folder: {path}")
            if not paths: raise ValueError("No WAV files in this folder. Choose the folder that contains the recordings.")
        except (OSError,ValueError) as exc:
            self._error("Could not select recordings",str(exc)); return
        paths=list(dict.fromkeys(paths))
        for p in paths: self._list_item(p)
        self.recordings_dock.show()
        if not self.project or str(paths[0]) != self.project.audio.path: self.open_audio(paths[0])
        if self.project: self.queue_list.setCurrentItem(self._list_item(Path(self.project.audio.path)))

    def _choose_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open WhistleMamba project", filter="WhistleMamba project (*.wmproject.json *.json)"
        )
        if path:
            self.open_project(path)

    def _choose_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select frozen config", filter="CSR01 config (*.json)")
        if path:
            self.config_edit.setText(path)

    def _choose_checkpoint(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select verified CSR01 inference weights", filter="SafeTensors (*.safetensors)")
        if path:
            self.checkpoint_edit.setText(path)

    def open_audio(self, path: str | Path) -> None:
        if self.worker is not None:
            self.status.showMessage("Finish or cancel the current analysis before switching recordings."); return
        if not self._confirm_discard():
            return
        try:
            self.project = AnalysisProject.create(path)
        except Exception as exc:  # noqa: BLE001 - GUI boundary reports all load failures
            self._error("Could not open WAV", str(exc))
            return
        self.project_path = None
        self.editor = ContourEditor(self.project)
        self.dirty = False
        self._update_project_ui()
        if self.player is not None:
            self.player.setSource(QtCore.QUrl.fromLocalFile(self.project.audio.path))
        self.view.set_audio(self.project.audio.path, self.project)
        self._recording_ready()

    def open_project(self, path: str | Path) -> None:
        if self.worker is not None:
            self.status.showMessage("Finish or cancel the current analysis before switching projects."); return
        if not self._confirm_discard():
            return
        try:
            project = AnalysisProject.load(path)
        except Exception as exc:  # noqa: BLE001 - GUI boundary reports all load failures
            self._error("Could not open project", str(exc))
            return
        audio_path = Path(project.audio.path)
        if not audio_path.is_file():
            replacement, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Locate project WAV", filter="WAV audio (*.wav *.WAV)"
            )
            if not replacement:
                return
            if AnalysisProject.create(replacement).audio.sha256 != project.audio.sha256:
                self._error("Audio hash mismatch", "The selected WAV is not the project source audio.")
                return
            project.audio = project.audio.__class__(**{**project.audio.__dict__, "path": str(Path(replacement).resolve())})
        elif not project.verify_audio():
            self._error("Audio hash mismatch", "The project WAV has changed; it was not opened.")
            return
        self.project = project
        self.project_path = Path(path).resolve()
        self.editor = ContourEditor(project)
        self.dirty = False
        self._update_project_ui()
        if self.player is not None:
            self.player.setSource(QtCore.QUrl.fromLocalFile(project.audio.path))
        self.view.set_audio(project.audio.path, project)
        self._recording_ready()
        self.status.showMessage("Project loaded; source WAV hash verified")

    def _full_review_layout(self):
        for panel in self.panels:
            panel.show()
        self.model_settings_toggle.setChecked(True)

    def _recording_ready(self):
        audio=self.project.audio;path=Path(audio.path)
        first=self.pages.currentWidget() is self.home
        self.pages.setCurrentWidget(self.view)
        if first:
            self._full_review_layout()
        self.setWindowTitle(f"WhistleMamba Studio — {path.name}")
        self.prefs.setValue("recordings/lastFolder",str(path.parent))
        self.queue_list.setCurrentItem(self._list_item(path))
        compatible=audio.duration_s>=4 and rate_supported(audio.sample_rate_hz)
        self.run_button.setEnabled(compatible)
        self.progress_bar.setValue(0)
        self.progress_label.setWordWrap(True)
        reasons=[]
        if not rate_supported(audio.sample_rate_hz): reasons.append(f"{audio.sample_rate_hz/1000:g} kHz recording; use 96 kHz or at least 192 kHz")
        if audio.duration_s<4: reasons.append(f"{audio.duration_s:.2f} s recording; at least 4 s required")
        self.progress_label.setText((LIMITED_NOTICE if audio.sample_rate_hz==96000 else "Ready to analyze") if compatible else "Preview only: "+"; ".join(reasons)+".")
        self.run_button.setToolTip(self.progress_label.text() if reasons else "")
        self.status.showMessage(f"{path.name}   ·   {audio.duration_s:.2f} s   ·   {audio.sample_rate_hz/1000:g} kHz   ·   {audio.channels} channel(s)")

    def _update_project_ui(self) -> None:
        if self.project is None:
            return
        audio = self.project.audio
        self.audio_path_label.setText(Path(audio.path).name)
        self.hash_label.setText(audio.sha256)
        self.rate_label.setText(f"{audio.sample_rate_hz:,} Hz")
        self.channels_label.setText(str(audio.channels))
        self.duration_label.setText(f"{audio.duration_s:.3f} s")
        self.project_label.setText(self.project_path.name if self.project_path else "Unsaved")
        self._refresh_contour_list()

    def save_project(self, *, save_as: bool = False) -> bool:
        if self.project is None:
            return False
        destination = self.project_path
        if save_as or destination is None:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save WhistleMamba project",
                str(Path(self.project.audio.path).with_suffix(".wmproject.json")),
                "WhistleMamba project (*.wmproject.json)",
            )
            if not selected:
                return False
            destination = Path(selected)
            if not str(destination).endswith(".wmproject.json"):
                destination = Path(str(destination) + ".wmproject.json")
        try:
            self.project_path = self.project.save(destination)
        except Exception as exc:  # noqa: BLE001 - GUI boundary reports all save failures
            self._error("Could not save project", str(exc))
            return False
        self.dirty = False
        self.project_label.setText(str(self.project_path))
        self.status.showMessage("Project saved; source WAV was not modified")
        return True

    def _choose_export(self) -> None:
        if self.project is None:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Export analysis bundle")
        if not directory:
            return
        try:
            outputs = export_bundle(self.project, directory)
        except Exception as exc:  # noqa: BLE001 - GUI boundary reports all export failures
            self._error("Export failed", str(exc))
            return
        self.status.showMessage(f"Exported {len(outputs)} audited artifacts to {directory}")

    def _start_inference(self) -> None:
        if self.project is None:
            self._error("No audio", "Open a WAV before running inference.")
            return
        config = Path(self.config_edit.text()).expanduser()
        checkpoint = Path(self.checkpoint_edit.text()).expanduser()
        if not config.is_file() or not checkpoint.is_file():
            self._error(
                "Model files missing",
                "Select the packaged CSR01 config and authorized local epoch-12 BASE safetensors.",
            )
            return
        settings = InferenceSettings(
            device=self.device_combo.currentText(),
            batch_size=self.batch_spin.value(),
            stride_seconds=self.stride_spin.value(),
            tracker_mode=self.tracker_combo.currentText(),
            presence_gating=self.presence_check.isChecked(),
        )
        app_data = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation))
        # Each tracker/batch choice needs its own state: resuming another mode
        # is correctly rejected by the predictor rather than silently reused.
        resume = app_data / "resume" / f"{self.project.audio.sha256}-{settings.tracker_mode}-b{settings.batch_size}.json"
        legacy = resume.with_name(f"{self.project.audio.sha256}.json")
        if not resume.exists() and legacy.is_file():
            import json
            from dataclasses import asdict
            try:
                if json.loads(legacy.read_text())["identity"]["settings"] == asdict(settings):
                    resume = legacy
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Preserve older state; never use an unidentified resume.
        resume.parent.mkdir(parents=True, exist_ok=True)
        self.worker_thread = QtCore.QThread(self)
        self.worker = InferenceWorker(
            audio_path=self.project.audio.path,
            config_path=str(config),
            checkpoint_path=str(checkpoint),
            project=self.project,
            settings=settings,
            resume_path=str(resume),
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._inference_progress)
        self.worker.overlay.connect(self.view.set_model_overlay)
        self.worker.completed.connect(self._inference_completed)
        self.worker.cancelled.connect(lambda message: self.status.showMessage(message))
        self.worker.failed.connect(lambda trace: self._error("Inference failed", trace))
        self.worker.finished.connect(self._inference_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_label.setText("Loading model…")
        self.worker_thread.start()

    def _cancel_inference(self) -> None:
        if self.worker is not None:
            self.worker.request_cancel()
            self.progress_label.setText("Cancellation requested; finishing current batch…")

    @QtCore.Slot(object)
    def _inference_progress(self, event: object) -> None:
        self.progress_bar.setValue(round(event.fraction * 1000))
        self.progress_label.setText(
            f"{event.completed_windows}/{event.total_windows} windows — {event.fraction:.1%}"
        )

    @QtCore.Slot(object)
    def _inference_completed(self, result: object) -> None:
        self.project = result.project
        self.editor = ContourEditor(self.project)
        self.dirty = True
        self._refresh_contour_list()
        self.view.set_project(self.project)
        self.status.showMessage(
            f"Analysis complete: {result.learned_contours} learned and "
            f"{result.heuristic_contours} heuristic contours"
        )

    @QtCore.Slot()
    def _inference_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.worker_thread = None

    def _update_overlay_visibility(self) -> None:
        self.view.set_overlay_visibility(
            centerline=self.centerline_check.isChecked(),
            contours=self.contours_check.isChecked(),
            endpoints=self.endpoints_check.isChecked(),
            junctions=self.junctions_check.isChecked(),
        )

    def _update_frequency_range(self) -> None:
        low, high = self.fmin_spin.value(), self.fmax_spin.value()
        if low >= high:
            self.status.showMessage("Frequency minimum must be below maximum")
            return
        self.view.set_frequency_range(low, high)

    def _refresh_contour_list(self, selected_id: str | None = None) -> None:
        self.contour_list.clear()
        if self.project is None:
            return
        for contour in self.project.edited_contours:
            item = QtWidgets.QListWidgetItem(
                f"{contour.contour_id[:10]}  {contour.source}  {contour.review_state}  "
                f"({len(contour.points)} points)"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, contour.contour_id)
            self.contour_list.addItem(item)
            if contour.contour_id == selected_id:
                item.setSelected(True)
        self.view.refresh_contours()

    def _selected_ids(self) -> list[str]:
        return [
            item.data(QtCore.Qt.ItemDataRole.UserRole) for item in self.contour_list.selectedItems()
        ]

    def _selected_contour(self) -> Contour | None:
        if self.project is None:
            return None
        identifiers = self._selected_ids()
        if len(identifiers) != 1:
            return None
        return next(
            (item for item in self.project.edited_contours if item.contour_id == identifiers[0]),
            None,
        )

    def _select_contour_id(self, contour_id: str) -> None:
        for index in range(self.contour_list.count()):
            item = self.contour_list.item(index)
            item.setSelected(item.data(QtCore.Qt.ItemDataRole.UserRole) == contour_id)
        self.view.select_contour(contour_id)

    def _contour_selection_changed(self) -> None:
        contour = self._selected_contour()
        if contour is None:
            return
        self.point_index.setMaximum(max(0, len(contour.points) - 1))
        self.point_index.setValue(0)
        self.notes_edit.setPlainText(contour.notes)
        self._point_selection_changed()
        self.view.select_contour(contour.contour_id)

    def _point_selection_changed(self) -> None:
        contour = self._selected_contour()
        if contour is None or not contour.points:
            return
        point = contour.points[min(self.point_index.value(), len(contour.points) - 1)]
        self.point_time.setValue(point.time_s)
        self.point_frequency.setValue(point.frequency_hz)

    def _after_edit(self, selected_id: str | None = None) -> None:
        self.dirty = True
        self._refresh_contour_list(selected_id)

    def _move_point(self) -> None:
        contour = self._selected_contour()
        if contour and self.editor:
            self.editor.move_point(
                contour.contour_id,
                self.point_index.value(),
                ContourPoint(self.point_time.value(), self.point_frequency.value()),
            )
            self._after_edit(contour.contour_id)

    def _add_point(self) -> None:
        contour = self._selected_contour()
        if contour and self.editor:
            self.editor.add_point(
                contour.contour_id,
                ContourPoint(self.point_time.value(), self.point_frequency.value()),
            )
            self._after_edit(contour.contour_id)

    def _delete_point(self) -> None:
        contour = self._selected_contour()
        if contour and self.editor:
            try:
                self.editor.delete_point(contour.contour_id, self.point_index.value())
            except ValueError as exc:
                self._error("Cannot delete point", str(exc))
                return
            self._after_edit(contour.contour_id)

    def _new_contour(self) -> None:
        if self.editor:
            contour = Contour.create(
                [ContourPoint(self.point_time.value(), self.point_frequency.value())],
                source="manual",
            )
            self.editor.add_contour(contour)
            self._after_edit(contour.contour_id)

    def _delete_contour(self) -> None:
        identifiers = self._selected_ids()
        if self.editor:
            for contour_id in identifiers:
                self.editor.delete_contour(contour_id)
            self._after_edit()

    def _split_contour(self) -> None:
        contour = self._selected_contour()
        if contour and self.editor:
            try:
                left, _ = self.editor.split(contour.contour_id, self.point_index.value())
            except ValueError as exc:
                self._error("Cannot split contour", str(exc))
                return
            self._after_edit(left)

    def _merge_contours(self) -> None:
        identifiers = self._selected_ids()
        if len(identifiers) != 2 or self.editor is None:
            self.status.showMessage("Select exactly two contours to merge")
            return
        merged = self.editor.merge(*identifiers)
        self._after_edit(merged)

    def _set_review(self, state: str) -> None:
        identifiers = self._selected_ids()
        if self.editor:
            for contour_id in identifiers:
                self.editor.set_review_state(contour_id, state)
            self._after_edit(identifiers[0] if len(identifiers) == 1 else None)

    def _save_notes(self) -> None:
        contour = self._selected_contour()
        if contour and self.editor:
            self.editor.set_notes(contour.contour_id, self.notes_edit.toPlainText())
            self._after_edit(contour.contour_id)

    def _undo(self) -> None:
        if self.editor and self.editor.undo():
            self._after_edit()

    def _redo(self) -> None:
        if self.editor and self.editor.redo():
            self._after_edit()

    def _toggle_playback(self) -> None:
        if self.player is None or QtMultimedia is None:
            self.status.showMessage("Playback unavailable: Qt multimedia system library is missing")
            return
        if self.player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _playback_position_changed(self, milliseconds: int) -> None:
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(milliseconds)
        self.view.cursor.setValue(milliseconds / 1000.0)

    def _playback_duration_changed(self, milliseconds: int) -> None:
        self.position_slider.setRange(0, milliseconds)

    def _cursor_seek(self) -> None:
        if self.player is not None:
            self.player.setPosition(max(0, round(float(self.view.cursor.value()) * 1000)))

    def _slider_seek(self) -> None:
        if self.player is not None:
            self.player.setPosition(self.position_slider.value())

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        response = QtWidgets.QMessageBox.question(
            self,
            "Unsaved edits",
            "Save the current non-destructive project edits before continuing?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if response == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_project()
        return response == QtWidgets.QMessageBox.StandardButton.Discard

    def _error(self, title: str, details: str) -> None:
        LOGGER.error("%s\n%s", title, details)
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(details.splitlines()[-1] if details else title)
        dialog.setDetailedText(details)
        dialog.exec()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None:
            self._cancel_inference()
            event.ignore()
            self.status.showMessage("Cancel analysis before closing")
            return
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()
