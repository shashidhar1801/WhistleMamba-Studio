from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from functools import wraps

import numpy as np

from .project import AnalysisProject, Contour, sha256_file, utc_now
from .registry import validate_project_identity

def _bound_export(fn):
    @wraps(fn)
    def checked(project, path, *args, **kwargs):
        validate_project_identity(project.model)
        if not project.analysis_complete or not project.verify_audio():
            raise ValueError("Export requires completed analysis and unchanged audio")
        if Path(path).resolve()==Path(project.audio.path).resolve():
            raise ValueError("Cannot overwrite source audio")
        return fn(project,path,*args,**kwargs)
    return checked


def _selected_contours(
    project: AnalysisProject, *, include_rejected: bool = False
) -> list[Contour]:
    return [
        contour
        for contour in project.edited_contours
        if include_rejected or contour.review_state != "rejected"
    ]


@_bound_export
def export_contour_csv(
    project: AnalysisProject,
    path: str | Path,
    *,
    include_rejected: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "contour_id",
                "node_index",
                "time_s",
                "frequency_hz",
                "confidence",
                "source",
                "review_state",
                "notes",
            ],
        )
        writer.writeheader()
        for contour in _selected_contours(project, include_rejected=include_rejected):
            for node_index, point in enumerate(contour.points):
                writer.writerow(
                    {
                        "contour_id": contour.contour_id,
                        "node_index": node_index,
                        "time_s": f"{point.time_s:.9f}",
                        "frequency_hz": f"{point.frequency_hz:.3f}",
                        "confidence": f"{point.confidence:.6f}",
                        "source": contour.source,
                        "review_state": contour.review_state,
                        "notes": contour.notes,
                    }
                )
    return destination


@_bound_export
def export_raven_selection_table(
    project: AnalysisProject,
    path: str | Path,
    *,
    include_rejected: bool = False,
) -> Path:
    """Export one Raven selection per contour using seconds and hertz."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Annotation",
        "Contour ID",
        "Mean Confidence",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for selection, contour in enumerate(
            _selected_contours(project, include_rejected=include_rejected), start=1
        ):
            times = np.asarray([point.time_s for point in contour.points])
            frequencies = np.asarray([point.frequency_hz for point in contour.points])
            confidences = np.asarray([point.confidence for point in contour.points])
            writer.writerow(
                {
                    "Selection": selection,
                    "View": "Spectrogram 1",
                    "Channel": 1,
                    "Begin Time (s)": f"{times.min():.9f}",
                    "End Time (s)": f"{times.max():.9f}",
                    "Low Freq (Hz)": f"{frequencies.min():.3f}",
                    "High Freq (Hz)": f"{frequencies.max():.3f}",
                    "Annotation": contour.review_state,
                    "Contour ID": contour.contour_id,
                    "Mean Confidence": f"{confidences.mean():.6f}",
                }
            )
    return destination


@_bound_export
def export_reproducibility_report(project: AnalysisProject, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = asdict(project.model) if project.model else None
    payload: dict[str, Any] = {
        "report_format": "WhistleMamba reproducibility report v1",
        "generated_utc": utc_now(),
        "project_id": project.project_id,
        "project_schema_version": project.schema_version,
        "audio": asdict(project.audio),
        "audio_hash_verified": project.verify_audio(),
        "model": model,
        "inference_settings": project.inference_settings,
        "original_contour_count": len(project.original_contours),
        "edited_contour_count": len(project.edited_contours),
        "edit_count": len(project.edit_log),
        "official_evaluation_accessed": project.official_evaluation_accessed,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


@_bound_export
def export_project_json(project: AnalysisProject, path: str | Path) -> Path:
    return project.save(path)


def export_batch_summary(projects: Sequence[AnalysisProject], path: str | Path) -> Path:
    destination = Path(path)
    for project in projects:
        validate_project_identity(project.model)
        if not project.analysis_complete or not project.verify_audio() or destination.resolve()==Path(project.audio.path).resolve():
            raise ValueError("Invalid source or destination for batch export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "project_id",
            "audio_path",
            "audio_sha256",
            "duration_s",
            "model_id",
            "checkpoint_sha256",
            "original_contours",
            "edited_contours",
            "accepted",
            "rejected",
            "pending",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for project in projects:
            states = {state: 0 for state in ("accepted", "rejected", "pending")}
            for contour in project.edited_contours:
                states[contour.review_state] += 1
            writer.writerow(
                {
                    "project_id": project.project_id,
                    "audio_path": project.audio.path,
                    "audio_sha256": project.audio.sha256,
                    "duration_s": f"{project.audio.duration_s:.6f}",
                    "model_id": project.model.model_id if project.model else "",
                    "checkpoint_sha256": project.model.checkpoint_sha256 if project.model else "",
                    "original_contours": len(project.original_contours),
                    "edited_contours": len(project.edited_contours),
                    **states,
                }
            )
    return destination


@_bound_export
def export_figure(
    project: AnalysisProject,
    path: str | Path,
    *,
    frequency_range_hz: tuple[float, float] | None = None,
    dynamic_range_db: float = 80.0,
    include_rejected: bool = False,
    dpi: int = 300,
) -> Path:
    """Export a publication-ready PNG or PDF spectrogram overlay."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from .spectrogram import SpectrogramSettings, compute_spectrogram_tile

    destination = Path(path)
    if destination.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("figure export must use .png or .pdf")
    low_hz, high_hz = frequency_range_hz or (
        4_000.0,
        min(96_000.0, project.audio.sample_rate_hz / 2),
    )
    tile = compute_spectrogram_tile(
        project.audio.path,
        start_s=0.0,
        duration_s=project.audio.duration_s,
        settings=SpectrogramSettings(
            f_min_hz=low_hz,
            f_max_hz=high_hz,
            dynamic_range_db=dynamic_range_db,
        ),
        maximum_time_bins=8_000,
    )

    # A download is rendered in an HTTP worker, not the GUI's main thread.
    # Own the figure directly: no pyplot manager, window, or global backend
    # switch that could interfere with the desktop editor during teardown.
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.pcolormesh(
        tile.time_s,
        tile.frequency_hz,
        tile.magnitude_db,
        shading="auto",
        rasterized=True,
        cmap="magma",
        vmin=tile.display_min_db,
        vmax=tile.display_max_db,
    )
    colors = {"learned": "#00D4FF", "heuristic": "#F4D03F", "manual": "#FFFFFF"}
    for contour in _selected_contours(project, include_rejected=include_rejected):
        axis.plot(
            [point.time_s for point in contour.points],
            [point.frequency_hz for point in contour.points],
            color=colors[contour.source],
            linewidth=1.2,
            alpha=0.95,
        )
    axis.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(low_hz, high_hz))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(destination, dpi=dpi)
    finally:
        figure.clear()
    return destination


def export_bundle(project: AnalysisProject, directory: str | Path) -> dict[str, str]:
    from .registry import validate_project_identity
    validate_project_identity(project.model)
    if not project.analysis_complete or not project.verify_audio():
        raise ValueError("Export requires completed analysis and the original, unchanged audio")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "contours_csv": export_contour_csv(project, root / "contours.csv"),
        "raven_table": export_raven_selection_table(project, root / "raven_selections.txt"),
        "project_json": export_project_json(project, root / "analysis.wmproject.json"),
        "reproducibility_report": export_reproducibility_report(
            project, root / "reproducibility.json"
        ),
        "figure_png": export_figure(project, root / "spectrogram_overlay.png"),
        "figure_pdf": export_figure(project, root / "spectrogram_overlay.pdf"),
    }
    manifest = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in outputs.items()}
    manifest_path = root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in outputs.items()} | {"manifest": str(manifest_path)}
