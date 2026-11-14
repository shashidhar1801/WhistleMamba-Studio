from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .main_window import MainWindow


def _configure_local_logging() -> Path:
    root = Path(
        QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppLocalDataLocation
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "whistlemamba-studio.log"
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger(__name__).info(
        "WhistleMamba-Studio started; telemetry=false remote_crash_reports=false"
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whistlemamba-studio")
    parser.add_argument("--wav")
    parser.add_argument("--project")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("WhistleMamba-Studio")
    app.setApplicationVersion("0.2.0b2")
    app.setOrganizationName("WhistleMamba")
    _configure_local_logging()
    window = MainWindow(
        initial_wav=arguments.wav,
        initial_project=arguments.project,
        initial_config=arguments.config,
        initial_checkpoint=arguments.checkpoint,
    )
    window.show()
    return app.exec()
