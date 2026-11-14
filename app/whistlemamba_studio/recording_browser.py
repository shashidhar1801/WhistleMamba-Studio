"""File intake for the desktop GUI. Paths never leave the local machine."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from PySide6 import QtCore, QtWidgets

WAV_FILTER = "WAV recordings (*.[wW][aA][vV])"
ROLE = QtCore.Qt.ItemDataRole.UserRole


def local_path(value: str | Path) -> Path:
    # Explorer's “Copy as path”, drive-qualified paths and local file URLs.
    value = str(value).strip().strip('"')
    if value.lower().startswith("file:"):
        url = QtCore.QUrl(value)
        if not url.isLocalFile() or url.host() not in ("", "localhost"):
            raise ValueError("Choose a local recording or a mounted drive.")
        value = unquote(url.path(QtCore.QUrl.ComponentFormattingOption.FullyEncoded))
    if os.name != "nt":
        match = re.fullmatch(r"/?([A-Za-z]):[\\/](.*)", value, flags=re.S)
        if match:
            return Path("/mnt") / match[1].lower() / match[2].replace("\\", "/")
        if value.startswith("\\\\"):
            raise ValueError("Use the recording’s mounted drive letter or Linux path instead of a network address.")
    return Path(value).expanduser()


def places() -> list[tuple[str, Path]]:
    result = []
    def add(label, path):
        if path and path.is_dir() and path not in [p for _, p in result]:
            result.append((label, path))
    home = os.environ.get("WHISTLEMAMBA_WINDOWS_HOME")
    if not home and sys.platform.startswith("linux"):
        # Handles the usual launch from a workspace under /mnt/<drive>/Users/<owner>.
        for candidate in (Path.cwd(), Path(__file__).resolve()):
            if len(candidate.parts) > 4 and candidate.parts[1] == "mnt" and candidate.parts[3].lower() == "users":
                home = str(Path(*candidate.parts[:5])); break
        if not home:
            name = os.environ.get("USERNAME")
            if name: home = str(Path("/mnt/c/Users") / name)
    home = local_path(home) if home else Path.home()
    for label in ("Downloads", "Desktop", "Documents"):
        value = os.environ.get("WHISTLEMAMBA_WINDOWS_" + label.upper())
        add(label, local_path(value) if value else home / label)
    add("Windows home" if home != Path.home() else "Home", home)
    if Path("/mnt").is_dir():
        for p in sorted(Path("/mnt").glob("[a-z]")):
            add(p.name.upper() + ": drive", p)
    add("Local home", Path.home())
    return result


def browser(parent, folder: bool, *, settings):
    dialog = QtWidgets.QFileDialog(parent)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setWindowTitle("Choose a recording folder" if folder else "Open WAV recordings")
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
    dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory if folder else QtWidgets.QFileDialog.FileMode.ExistingFiles)
    dialog.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, folder)
    if not folder:
        dialog.setNameFilter(WAV_FILTER)
    favorites = places()
    last = Path(settings.value("recordings/lastFolder", "", type=str) or "/nonexistent")
    default = dict(favorites).get("Desktop") if os.name == "nt" else None
    dialog.setDirectory(str(last if last.is_dir() else (default or (favorites[0][1] if favorites else Path.home()))))
    dialog.setSidebarUrls([QtCore.QUrl.fromLocalFile(str(p)) for _,p in favorites])
    # QFileDialog normally labels mount points 'c' and 'd'. Give recognizable Explorer names.
    sidebar = dialog.findChild(QtWidgets.QListView, "sidebar")
    if sidebar:
        names = {str(p): label for label,p in favorites}
        model = sidebar.model()
        for i in range(model.rowCount()):
            idx = model.index(i,0)
            u = idx.data(QtCore.Qt.ItemDataRole.UserRole+1)
            if isinstance(u, QtCore.QUrl) and u.toLocalFile() in names:
                model.setData(idx, names[u.toLocalFile()], QtCore.Qt.ItemDataRole.DisplayRole)
    dialog.setLabelText(QtWidgets.QFileDialog.DialogLabel.Accept, "Choose folder" if folder else "Open selected")
    dialog.resize(920,570)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted: return []
    return [Path(p) for p in dialog.selectedFiles()]


class Welcome(QtWidgets.QWidget):
    browse = QtCore.Signal()
    folder = QtCore.Signal()
    paste = QtCore.Signal()
    project = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("welcome")
        outer = QtWidgets.QVBoxLayout(self); outer.setContentsMargins(32,24,32,24)
        outer.addStretch()
        panel = QtWidgets.QFrame(); panel.setObjectName("startPanel");panel.setMaximumWidth(650)
        lay = QtWidgets.QVBoxLayout(panel);lay.setContentsMargins(34,32,34,31);lay.setSpacing(14)
        label = QtWidgets.QLabel("WHISTLEMAMBA  /  STUDIO");label.setObjectName("eyebrow");lay.addWidget(label)
        headline = QtWidgets.QLabel("Open a recording");headline.setObjectName("heading");lay.addWidget(headline)
        hint = QtWidgets.QLabel("Open WAV files to see the waveform, spectrogram and full contour editor. You choose when to run the detector.")
        hint.setWordWrap(True);hint.setObjectName("lead");lay.addWidget(hint)
        row = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open WAV files…");self.open_button.setObjectName("primary")
        self.folder_button = QtWidgets.QPushButton("Choose folder…")
        for button,signal in [(self.open_button,self.browse),(self.folder_button,self.folder)]:
            button.setMinimumHeight(40);button.clicked.connect(signal.emit);row.addWidget(button)
        row.addStretch();lay.addLayout(row)
        self.paste_button=QtWidgets.QPushButton("Paste a file or folder path…");self.paste_button.setObjectName("textLink")
        self.paste_button.clicked.connect(self.paste.emit);lay.addWidget(self.paste_button,0,QtCore.Qt.AlignmentFlag.AlignLeft)
        sep=QtWidgets.QFrame();sep.setFrameShape(QtWidgets.QFrame.Shape.HLine);lay.addWidget(sep)
        foot=QtWidgets.QLabel("You can also drop WAV files here, or reopen a saved review.");foot.setWordWrap(True);lay.addWidget(foot)
        b=QtWidgets.QPushButton("Open saved project…");b.setObjectName("textLink");b.clicked.connect(self.project.emit)
        lay.addWidget(b,0,QtCore.Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(panel,0,QtCore.Qt.AlignmentFlag.AlignHCenter);outer.addStretch()
        self.setStyleSheet("""
        #welcome {background:#edf4f5;color:#203c46} #startPanel {background:white;border:1px solid #c5d6da;border-radius:12px}
        #startPanel QLabel {color:#29434e;background:transparent} #startPanel #eyebrow {font-size:12px;font-weight:700;color:#256a70}
        #startPanel #heading {font-size:32px;font-weight:700;color:#143941} #startPanel #lead {font-size:16px}
        #startPanel QPushButton {background:#f2f7f8;color:#204851;border:1px solid #abbfc4;border-radius:5px;padding:6px 13px}
        #startPanel QPushButton:hover {background:#dfedef} #startPanel QPushButton#primary {background:#076b70;color:white;border-color:#076b70;font-weight:700}
        #startPanel QPushButton#primary:hover {background:#00565c} #startPanel QPushButton#textLink {padding:4px 0;border:0;background:white;color:#12636b}
        #startPanel QPushButton:focus {border:2px solid #187c86} """)

