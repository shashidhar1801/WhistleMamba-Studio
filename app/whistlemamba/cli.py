from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from . import __version__
from .export import export_bundle
from .inference import InferenceProgress, InferenceSettings, WhistleMambaPredictor
from .project import AnalysisProject
from .registry import IDENTITY


def _progress(event: InferenceProgress) -> None:
    percent = 100.0 * event.fraction
    print(
        f"[{event.completed_windows}/{event.total_windows}] {percent:6.2f}% "
        f"window_start_s={event.window_start_s:.3f}",
        flush=True,
    )


def _analyze(arguments: argparse.Namespace) -> int:
    settings = InferenceSettings(
        device=arguments.device,
        batch_size=arguments.batch_size,
        stride_seconds=arguments.stride_seconds,
        tracker_mode=arguments.tracker,
        presence_gating=arguments.presence_gating,
        presence_threshold=arguments.presence_threshold,
    )
    predictor = WhistleMambaPredictor(
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        settings=settings,
        expected_checkpoint_sha256=arguments.checkpoint_sha256,
        expected_config_sha256=arguments.config_sha256,
    )
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = predictor.analyze_file(
        arguments.wav,
        progress=_progress,
        resume_path=arguments.resume_state or output / "inference.resume.json",
    )
    project_path = result.project.save(output / "analysis.wmproject.json")
    exports = export_bundle(result.project, output / "exports") if not arguments.no_exports else {}
    print(
        json.dumps(
            {
                "project": str(project_path),
                "windows": result.windows,
                "learned_contours": result.learned_contours,
                "heuristic_contours": result.heuristic_contours,
                "resumed_windows": result.resumed_windows,
                "runtime_kernel": result.runtime_kernel,
                "exports": exports,
                "official_evaluation_accessed": False,
            },
            indent=2,
        )
    )
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    project = AnalysisProject.load(arguments.project)
    print(
        json.dumps(
            {
                "project_id": project.project_id,
                "audio": project.audio.__dict__,
                "audio_hash_verified": project.verify_audio(),
                "model": project.model.__dict__ if project.model else None,
                "original_contours": len(project.original_contours),
                "edited_contours": len(project.edited_contours),
                "edits": len(project.edit_log),
                "analysis_complete": project.analysis_complete,
                "official_evaluation_accessed": project.official_evaluation_accessed,
            },
            indent=2,
        )
    )
    return 0


def _export(arguments: argparse.Namespace) -> int:
    project = AnalysisProject.load(arguments.project)
    print(json.dumps(export_bundle(project, arguments.output), indent=2))
    return 0


def _doctor(_: argparse.Namespace) -> int:
    optional = {name: bool(importlib.util.find_spec(name)) for name in ("PySide6", "pyqtgraph", "mamba_ssm")}
    payload = {
        "whistlemamba": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "optional_dependencies": optional,
        "cpu_portable_mamba2": False,
        "registered_models": [IDENTITY["model_id"]],
        "official_evaluation_accessed": False,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _studio(arguments: argparse.Namespace) -> int:
    try:
        from whistlemamba_studio.application import main as studio_main
    except ImportError as exc:
        raise SystemExit("Install the GUI with: pip install -e '.[studio]'") from exc
    return studio_main(arguments.studio_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whistlemamba", description="WhistleMamba analysis tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="analyze one WAV with CSR01 joint BASE epoch 12")
    analyze.add_argument("--wav", required=True)
    analyze.add_argument("--config", help="Defaults to the immutable packaged CSR01 configuration")
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    analyze.add_argument("--batch-size", type=int, default=1)
    analyze.add_argument("--stride-seconds", type=float)
    analyze.add_argument("--tracker", choices=("learned", "heuristic", "both"), default="learned")
    analyze.set_defaults(presence_gating=False,presence_threshold=0.5)
    analyze.add_argument("--checkpoint-sha256")
    analyze.add_argument("--config-sha256")
    analyze.add_argument("--resume-state")
    analyze.add_argument("--no-exports", action="store_true")
    analyze.set_defaults(function=_analyze)

    inspect = commands.add_parser("inspect", help="inspect and hash-check a saved project")
    inspect.add_argument("project")
    inspect.set_defaults(function=_inspect)

    export = commands.add_parser("export", help="export CSV, Raven, JSON, PNG, PDF, and provenance")
    export.add_argument("project")
    export.add_argument("--output", required=True)
    export.set_defaults(function=_export)

    doctor = commands.add_parser("doctor", help="report runtime and dependency readiness")
    doctor.set_defaults(function=_doctor)

    studio = commands.add_parser("studio", help="launch WhistleMamba-Studio")
    studio.add_argument("studio_args", nargs=argparse.REMAINDER)
    studio.set_defaults(function=_studio)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
