from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from whistlemamba.project import AnalysisProject, Contour
from .preview import VIEW_SECONDS
from whistlemamba.spectrogram import (
    SpectrogramSettings,
    SpectrogramTileCache,
    read_waveform_region,
)

PALETTE = {
    "learned": "#00A6D6",
    "heuristic": "#E69F00",
    "manual": "#CC79A7",
    "endpoint": "#009E73",
    "junction": "#D55E00",
}


def grid_rect(times, frequencies):
    """QRectF addresses pixel edges; input coordinates are pixel centres."""
    dt = float(times[1] - times[0]) if len(times)>1 else 0.002
    df = float(frequencies[1] - frequencies[0]) if len(frequencies)>1 else 125.0
    return QtCore.QRectF(float(times[0])-dt/2, float(frequencies[0])-df/2,
                         dt*len(times), df*len(frequencies))


class SpectrogramView(QtWidgets.QWidget):
    contour_selected = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        pg.setConfigOption("imageAxisOrder", "row-major")
        self.audio_path: Path | None = None
        self.project: AnalysisProject | None = None
        self.settings = SpectrogramSettings()
        self.cache = SpectrogramTileCache(maximum_tiles=8)
        self.viewport_seconds = VIEW_SECONDS
        self._contour_items: list[pg.PlotDataItem] = []
        self._overlay_start: float | None = None

        self.canvas = pg.GraphicsLayoutWidget()
        self.waveform_plot = self.canvas.addPlot(row=0, col=0)
        self.waveform_plot.setMaximumHeight(110)
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.hideAxis("bottom")
        self.spectrogram_plot = self.canvas.addPlot(row=1, col=0)
        self.spectrogram_plot.setLabel("left", "Frequency", units="Hz")
        self.spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.spectrogram_plot.setXLink(self.waveform_plot)
        self.spectrogram_plot.setMouseEnabled(x=True, y=True)
        self.spectrogram_image = pg.ImageItem()
        self.centerline_image = pg.ImageItem(opacity=0.48)
        self.spectrogram_plot.addItem(self.spectrogram_image)
        self.spectrogram_plot.addItem(self.centerline_image)
        self.endpoint_item = pg.ScatterPlotItem(size=7, pen=None, brush=PALETTE["endpoint"])
        self.junction_item = pg.ScatterPlotItem(
            size=8, symbol="x", pen=pg.mkPen(PALETTE["junction"], width=2)
        )
        self.spectrogram_plot.addItem(self.endpoint_item)
        self.spectrogram_plot.addItem(self.junction_item)
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#FFFFFF", width=1))
        self.spectrogram_plot.addItem(self.cursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._tile_timer = QtCore.QTimer(self, interval=150, singleShot=True)
        self._tile_timer.timeout.connect(self.render_visible_tile)
        self.spectrogram_plot.sigXRangeChanged.connect(lambda *_: self._tile_timer.start())
        self.spectrogram_plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self._set_centerline_lut()

    def _set_centerline_lut(self) -> None:
        colors = np.zeros((256, 4), dtype=np.ubyte)
        colors[:, 0] = np.linspace(0, 0, 256).astype(np.ubyte)
        colors[:, 1] = np.linspace(0, 212, 256).astype(np.ubyte)
        colors[:, 2] = np.linspace(0, 255, 256).astype(np.ubyte)
        colors[:, 3] = np.linspace(0, 230, 256).astype(np.ubyte)
        self.centerline_image.setLookupTable(colors)
        self.centerline_image.setLevels((0.0, 1.0))

    def set_audio(self, path: str | Path, project: AnalysisProject) -> None:
        self.audio_path = Path(path)
        self.project = project
        self.cache.clear()
        self._overlay_start = None
        self.centerline_image.clear()
        self.endpoint_item.setData([], [])
        self.junction_item.setData([], [])
        self.spectrogram_image.clear()
        self.waveform_plot.clear()
        duration = max(project.audio.duration_s, 0.001)
        self.spectrogram_plot.setXRange(0.0, min(duration, self.viewport_seconds), padding=0)
        self.spectrogram_plot.setYRange(
            self.settings.f_min_hz,
            min(self.settings.f_max_hz, project.audio.sample_rate_hz / 2),
            padding=0,
        )
        self.waveform_plot.setXRange(0.0, min(duration, self.viewport_seconds), padding=0)
        self.cursor.setValue(0.0)
        self.render_visible_tile()
        self.refresh_contours()

    def set_project(self, project: AnalysisProject) -> None:
        self.project = project
        self.refresh_contours()

    def set_frequency_range(self, low_hz: float, high_hz: float) -> None:
        self.settings = SpectrogramSettings(
            **{**self.settings.__dict__, "f_min_hz": low_hz, "f_max_hz": high_hz}
        )
        self.cache.clear()
        self.spectrogram_plot.setYRange(low_hz, high_hz, padding=0)
        self.render_visible_tile()

    def set_dynamic_range(self, value_db: float) -> None:
        self.settings = SpectrogramSettings(
            **{**self.settings.__dict__, "dynamic_range_db": value_db}
        )
        self.cache.clear()
        self.render_visible_tile()

    @QtCore.Slot()
    def render_visible_tile(self) -> None:
        if self.audio_path is None or self.project is None:
            return
        visible = self.spectrogram_plot.viewRange()[0]
        duration = self.project.audio.duration_s
        start = max(0.0, min(float(visible[0]), max(0.0, duration - 0.1)))
        end = min(duration, max(start + 0.1, float(visible[1])))
        try:
            tile = self.cache.get(
                self.audio_path,
                start_s=start,
                duration_s=end - start,
                settings=self.settings,
                maximum_time_bins=3_000,
            )
        except (OSError, ValueError):
            return
        if tile.magnitude_db.size:
            self.spectrogram_image.setImage(tile.magnitude_db, autoLevels=False)
            self.spectrogram_image.setLevels((tile.display_min_db, tile.display_max_db))
            self.spectrogram_image.setRect(grid_rect(tile.time_s, tile.frequency_hz))
        waveform, sample_rate = read_waveform_region(
            self.audio_path, start_s=start, duration_s=end - start
        )
        if waveform.size:
            mono = waveform[:, 0]
            factor = max(1, len(mono) // 4_000)
            samples = mono[::factor]
            times = start + np.arange(len(samples)) * factor / sample_rate
            self.waveform_plot.clear()
            self.waveform_plot.plot(times, samples, pen=pg.mkPen("#B3B3B3", width=1))

    @QtCore.Slot(object)
    def set_model_overlay(self, overlay: object) -> None:
        self._overlay_start = float(overlay.start_s)
        probability = np.asarray(overlay.centerline_probability)
        self.centerline_image.setImage(probability, autoLevels=False)
        self.centerline_image.setRect(grid_rect(overlay.time_axis_s, overlay.frequency_axis_hz))
        endpoint = np.max(np.asarray(overlay.endpoint_probability), axis=0)
        ef, et = np.nonzero(endpoint >= 0.45)
        if len(ef) > 2_000:
            order = np.argsort(endpoint[ef, et])[-2_000:]
            ef, et = ef[order], et[order]
        self.endpoint_item.setData(
            x=np.asarray(overlay.time_axis_s)[et],
            y=np.asarray(overlay.frequency_axis_hz)[ef],
        )
        junction = np.asarray(overlay.junction_probability)
        jf, jt = np.nonzero(junction >= 0.70)
        if len(jf) > 1_000:
            order = np.argsort(junction[jf, jt])[-1_000:]
            jf, jt = jf[order], jt[order]
        self.junction_item.setData(
            x=np.asarray(overlay.time_axis_s)[jt],
            y=np.asarray(overlay.frequency_axis_hz)[jf],
        )

    def set_overlay_visibility(
        self,
        *,
        centerline: bool,
        contours: bool,
        endpoints: bool,
        junctions: bool,
    ) -> None:
        self.centerline_image.setVisible(centerline)
        self.endpoint_item.setVisible(endpoints)
        self.junction_item.setVisible(junctions)
        for item in self._contour_items:
            item.setVisible(contours)

    def refresh_contours(self) -> None:
        for item in self._contour_items:
            self.spectrogram_plot.removeItem(item)
        self._contour_items.clear()
        if self.project is None:
            return
        for contour in self.project.edited_contours:
            if contour.review_state == "rejected":
                alpha = 70
                style = QtCore.Qt.PenStyle.DashLine
            else:
                alpha = 230
                style = QtCore.Qt.PenStyle.SolidLine
            color = QtGui.QColor(PALETTE[contour.source])
            color.setAlpha(alpha)
            item = self.spectrogram_plot.plot(
                [point.time_s for point in contour.points],
                [point.frequency_hz for point in contour.points],
                pen=pg.mkPen(color, width=2, style=style),
            )
            item.setProperty("contour_id", contour.contour_id)
            item.setToolTip(f"{contour.contour_id}\n{contour.source} / {contour.review_state}")
            self._contour_items.append(item)

    def select_contour(self, contour_id: str) -> None:
        for item in self._contour_items:
            selected = item.property("contour_id") == contour_id
            item.setZValue(20 if selected else 10)
            pen = item.opts["pen"]
            pen.setWidth(4 if selected else 2)
            item.setPen(pen)

    def _mouse_clicked(self, event: object) -> None:
        if self.project is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        position = self.spectrogram_plot.vb.mapSceneToView(event.scenePos())
        x, y = float(position.x()), float(position.y())
        best: tuple[float, Contour] | None = None
        for contour in self.project.edited_contours:
            times = np.asarray([point.time_s for point in contour.points])
            frequencies = np.asarray([point.frequency_hz for point in contour.points])
            if len(times) == 1:
                distance = abs(times[0] - x) / 0.05 + abs(frequencies[0] - y) / 1_000.0
            elif times[0] <= x <= times[-1]:
                distance = abs(float(np.interp(x, times, frequencies)) - y) / 1_000.0
            else:
                continue
            if best is None or distance < best[0]:
                best = (distance, contour)
        if best and best[0] <= 2.0:
            self.contour_selected.emit(best[1].contour_id)
