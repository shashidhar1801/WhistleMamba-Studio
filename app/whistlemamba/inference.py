from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch

from .config import dataclass_from_mapping, load_yaml
from .data import load_audio_window
from .input_mode import compatibility_metadata, rate_supported
from .features import FeatureConfig, MultiResolutionSTFT, feature_config_signature
from .canonical import canonical_state_dict_sha256
from .registry import IDENTITY, config_resource, validate_project_identity
from .decoder import inference_graph, track_cached_window, _physical_track
from safetensors.torch import load as load_tensor_bytes
from .model import WhistleMambaConfig, WhistleMambaTrack
from .project import (
    AnalysisProject,
    Contour,
    ContourPoint,
    ModelIdentity,
    sha256_file,
    utc_now,
)
from .tracking import (
    HeuristicContourTracker,
    LearnedContourTracker,
    TrackerConfig,
    stitch_overlapping_tracks,
)
from .utils import autocast_context

TrackerMode = Literal["learned", "heuristic", "both"]
ProgressCallback = Callable[["InferenceProgress"], None]
CancelCallback = Callable[[], bool]
OverlayCallback = Callable[["WindowOverlay"], None]


class InferenceCancelled(RuntimeError):
    """Raised after a resumable analysis is cancelled by the caller."""


@dataclass(frozen=True)
class InferenceSettings:
    device: str = "auto"
    batch_size: int = 1
    stride_seconds: float | None = None
    tracker_mode: TrackerMode = "learned"
    presence_gating: bool = False
    presence_threshold: float = 0.5
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.stride_seconds is not None and self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")
        if self.tracker_mode not in {"learned", "heuristic", "both"}:
            raise ValueError("tracker_mode must be learned, heuristic, or both")
        if not 0 <= self.presence_threshold <= 1:
            raise ValueError("presence_threshold must be within [0, 1]")


@dataclass(frozen=True)
class InferenceProgress:
    completed_windows: int
    total_windows: int
    window_start_s: float
    phase: str

    @property
    def fraction(self) -> float:
        return self.completed_windows / max(self.total_windows, 1)


@dataclass(frozen=True)
class WindowOverlay:
    start_s: float
    time_axis_s: np.ndarray
    frequency_axis_hz: np.ndarray
    centerline_probability: np.ndarray
    endpoint_probability: np.ndarray
    junction_probability: np.ndarray
    presence_probability: float


@dataclass(frozen=True)
class InferenceResult:
    project: AnalysisProject
    windows: int
    learned_contours: int
    heuristic_contours: int
    resumed_windows: int
    runtime_kernel: str


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _window_starts(duration_s: float, window_s: float, stride_s: float) -> list[float]:
    if duration_s <= window_s:
        return [0.0]
    final_start = max(0.0, duration_s - window_s)
    starts = list(np.arange(0.0, final_start + 1e-9, stride_s, dtype=np.float64))
    if not starts or abs(starts[-1] - final_start) > 1e-6:
        starts.append(final_start)
    return [float(value) for value in starts]


def _track_payload(track: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "time_s": np.asarray(track["time_s"], dtype=np.float64).tolist(),
        "frequency_hz": np.asarray(track["frequency_hz"], dtype=np.float64).tolist(),
        "node_confidence": np.asarray(track["node_confidence"], dtype=np.float64).tolist(),
        "mean_confidence": float(track.get("mean_confidence", 0.0)),
        "mean_edge_probability": float(track.get("mean_edge_probability", 0.0)),
    }


def _track_arrays(track: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "time_s": np.asarray(track["time_s"], dtype=np.float64),
        "frequency_hz": np.asarray(track["frequency_hz"], dtype=np.float64),
        "node_confidence": np.asarray(track["node_confidence"], dtype=np.float64),
        "mean_confidence": float(track.get("mean_confidence", 0.0)),
        "mean_edge_probability": float(track.get("mean_edge_probability", 0.0)),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class WhistleMambaPredictor:
    """Reusable, GUI-independent WhistleMamba-Track-v1 inference engine."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path,
        settings: InferenceSettings | None = None,
        expected_checkpoint_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        allow_non_scientific_backend: bool = False,
    ):
        settings = settings or InferenceSettings()
        self.config_path = Path(config_path or config_resource()).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        import hashlib
        config_bytes=self.config_path.read_bytes()
        self.config_sha256=hashlib.sha256(config_bytes).hexdigest()
        if self.config_sha256 != IDENTITY["config_sha256"]:
            raise ValueError("Wrong CSR01 configuration SHA-256")
        if self.checkpoint_path.stat().st_size > 100_000_000:
            raise ValueError("Expected inference-only CSR01 safetensors, not a training checkpoint")
        asset=self.checkpoint_path.read_bytes()
        self.checkpoint_sha256=hashlib.sha256(asset).hexdigest()
        if self.checkpoint_sha256!=IDENTITY["checkpoint_sha256"]:
            raise ValueError("Wrong CSR01 inference weights SHA-256")
        if expected_config_sha256 and self.config_sha256 != expected_config_sha256:
            raise ValueError("configuration SHA-256 does not match the model registry")
        if expected_checkpoint_sha256 and self.checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 does not match the model registry")

        if allow_non_scientific_backend or not settings.deterministic or settings.presence_gating or settings.stride_seconds not in (None,2.0):
            raise ValueError("CSR01 requires the frozen backend, 2-s stride and no presence gating")
        raw = json.loads(config_bytes)
        self.raw_config = raw
        self.feature_config = dataclass_from_mapping(FeatureConfig, raw.get("features"))
        if feature_config_signature(self.feature_config)!=IDENTITY["feature_signature"]:
            raise ValueError("CSR01 normalization/feature signature mismatch")
        self.model_config = dataclass_from_mapping(WhistleMambaConfig, raw.get("model"))
        self.tracker_config = dataclass_from_mapping(TrackerConfig, raw.get("tracker"))
        self.stitching_config = self._stitching_config(raw)
        self.settings = settings
        self.device = _resolve_device(settings.device)
        if self.device.type!="cuda":
            raise RuntimeError("This CSR01 candidate requires the official CUDA Mamba-2 backend")
        self.runtime_kernel = "cuda_fused"

        if settings.deterministic:
            import random
            random.seed(17)
            torch.set_num_threads(8)
            torch.use_deterministic_algorithms(True, warn_only=True)
            np.random.seed(17)
            torch.manual_seed(17)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(17)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.model = WhistleMambaTrack(self.model_config)
        if self.model.effective_backend != "mamba2":
            raise RuntimeError("Official Mamba-2 runtime required")
        state=load_tensor_bytes(asset)
        if canonical_state_dict_sha256(state)!=IDENTITY["canonical_state_dict_sha256"]:
            raise ValueError("CSR01 BASE tensor identity mismatch")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.frontend = MultiResolutionSTFT(self.feature_config).to(self.device).eval()
        self.frequency_axis, self.time_axis = (
            axis.detach().cpu().numpy() for axis in self.frontend.grid()
        )
        self.learned_tracker = LearnedContourTracker(self.model.edge_scorer, self.tracker_config)
        self.heuristic_tracker = HeuristicContourTracker(self._heuristic_tracker_config(raw))

    @staticmethod
    def _stitching_config(raw: Mapping[str, Any]) -> dict[str, float]:
        direct = raw.get("stitching")
        if isinstance(direct, Mapping):
            return {key: float(value) for key, value in direct.items()}
        nested = raw.get("milestone_i", {})
        if isinstance(nested, Mapping):
            closure = nested.get("validation_closure", {})
            if isinstance(closure, Mapping) and isinstance(closure.get("cross_window_stitching"), Mapping):
                return {key: float(value) for key, value in closure["cross_window_stitching"].items()}
        return {}

    @staticmethod
    def _heuristic_tracker_config(raw: Mapping[str, Any]) -> TrackerConfig:
        nested = raw.get("milestone_i", {})
        if isinstance(nested, Mapping):
            closure = nested.get("validation_closure", {})
            if isinstance(closure, Mapping) and isinstance(closure.get("heuristic_tracker"), Mapping):
                return dataclass_from_mapping(TrackerConfig, closure["heuristic_tracker"])
        heuristic = raw.get("heuristic_tracker")
        return dataclass_from_mapping(TrackerConfig, heuristic or raw.get("tracker"))

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_id=IDENTITY["model_id"],
            canonical_state_dict_sha256=IDENTITY["canonical_state_dict_sha256"],
            normalization_identity_sha256=IDENTITY["normalization_identity_sha256"],
            checkpoint_sha256=self.checkpoint_sha256,
            config_sha256=self.config_sha256,
            backend=self.model.effective_backend,
            kernel=self.runtime_kernel,
            device=str(self.device),
        )

    def _extract_window(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch_index: int,
        start_s: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        presence = float(torch.sigmoid(outputs["presence_logit"])[batch_index].item())
        graph=inference_graph(outputs,self.model.edge_scorer,self.tracker_config,batch_index=batch_index)
        def decode(mode,cfg):
            if self.settings.tracker_mode not in (mode,"both"):
                return []
            tracks,_=track_cached_window(graph,cfg,mode=mode)
            return [_physical_track(t,window_start_s=start_s) for t in tracks]
        return decode("learned",self.tracker_config),decode("heuristic",self._heuristic_tracker_config(self.raw_config)),presence

    def analyze_file(
        self,
        audio_path: str | Path,
        *,
        project: AnalysisProject | None = None,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
        overlay: OverlayCallback | None = None,
        resume_path: str | Path | None = None,
    ) -> InferenceResult:
        audio = Path(audio_path).expanduser().resolve()
        with sf.SoundFile(audio) as handle:
            duration_s = len(handle) / handle.samplerate
            source_rate = handle.samplerate
            if handle.format not in {"WAV","WAVEX","RF64"} or duration_s<4 or not rate_supported(source_rate):
                raise ValueError("Supported input: WAV of at least 4 seconds: 96 kHz (experimental) or 192 kHz and above.")
        adaptation = compatibility_metadata(source_rate)
        keep_bins = int(np.searchsorted(self.frequency_axis, source_rate/2, side="right")) if adaptation else None
        project = project or AnalysisProject.create(audio)
        if project.model is not None:
            validate_project_identity(project.model)
        if project.audio.sha256 != sha256_file(audio):
            raise ValueError("project audio hash does not match the selected WAV")

        stride = self.settings.stride_seconds or self.feature_config.window_seconds / 2.0
        starts = _window_starts(duration_s, self.feature_config.window_seconds, stride)
        learned_raw: list[dict[str, Any]] = []
        heuristic_raw: list[dict[str, Any]] = []
        completed: list[float] = []
        resume = Path(resume_path).expanduser().resolve() if resume_path else None
        if resume in (audio,self.checkpoint_path,self.config_path):
            raise ValueError("Resume path must not overwrite an input")
        if resume and resume.is_file():
            state = json.loads(resume.read_text(encoding="utf-8"))
            identity = state.get("identity", {})
            expected = {
                **adaptation,
                "deployment": asdict(self.model_identity),
                "audio_sha256": project.audio.sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "config_sha256": self.config_sha256,
                "settings": asdict(self.settings),
            }
            if identity != expected:
                raise ValueError("resume state belongs to different audio, model, config, or settings")
            completed = [float(value) for value in state.get("completed_window_starts_s", [])]
            learned_raw = [_track_arrays(item) for item in state.get("learned_tracks", [])]
            heuristic_raw = [_track_arrays(item) for item in state.get("heuristic_tracks", [])]
        resumed_windows = len(completed)
        pending = [value for value in starts if not any(abs(value - done) < 1e-8 for done in completed)]

        for batch_start in range(0, len(pending), self.settings.batch_size):
            if cancelled and cancelled():
                raise InferenceCancelled(f"analysis cancelled; resume state: {resume or 'not configured'}")
            batch_starts = pending[batch_start : batch_start + self.settings.batch_size]
            waveforms = torch.stack(
                [
                    load_audio_window(
                        audio,
                        start_s=start,
                        duration_s=self.feature_config.window_seconds,
                        target_sample_rate=self.feature_config.sample_rate,
                    )
                    for start in batch_starts
                ]
            ).to(self.device)
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                features = self.frontend(waveforms).half().float()
                with autocast_context(self.device, "bfloat16"):
                    outputs = self.model(features)
            if keep_bins is not None:
                # Keep the model input/grid intact. Cut every spatial output at
                # source Nyquist BEFORE NMS, node-budget ranking and both trackers.
                # This also bounds the live overlay, not just the exported tracks.
                outputs = _bandlimit_outputs(outputs, keep_bins)
            for batch_index, start in enumerate(batch_starts):
                learned, heuristic, presence = self._extract_window(outputs, batch_index, start)
                learned_raw.extend(learned)
                heuristic_raw.extend(heuristic)
                completed.append(start)
                if overlay:
                    overlay(
                        WindowOverlay(
                            start_s=start,
                            time_axis_s=self.time_axis + start,
                            frequency_axis_hz=self.frequency_axis[:keep_bins],
                            centerline_probability=torch.sigmoid(
                                outputs["centerline_logits"][batch_index, 0]
                            ).detach().float().cpu().numpy(),
                            endpoint_probability=torch.sigmoid(
                                outputs["endpoint_logits"][batch_index]
                            ).detach().float().cpu().numpy(),
                            junction_probability=torch.sigmoid(
                                outputs["junction_logits"][batch_index, 0]
                            ).detach().float().cpu().numpy(),
                            presence_probability=presence,
                        )
                    )
                if progress:
                    progress(
                        InferenceProgress(
                            completed_windows=len(completed),
                            total_windows=len(starts),
                            window_start_s=start,
                            phase="inference",
                        )
                    )
            if resume:
                _atomic_json(
                    resume,
                    {
                        "schema_version": 1,
                        "updated_utc": utc_now(),
                        "complete": False,
                        "identity": {
                            **adaptation,
                            "deployment": asdict(self.model_identity),
                            "audio_sha256": project.audio.sha256,
                            "checkpoint_sha256": self.checkpoint_sha256,
                            "config_sha256": self.config_sha256,
                            "settings": asdict(self.settings),
                        },
                        "completed_window_starts_s": completed,
                        "learned_tracks": [_track_payload(item) for item in learned_raw],
                        "heuristic_tracks": [_track_payload(item) for item in heuristic_raw],
                    },
                )

        learned = stitch_overlapping_tracks(learned_raw, **self.stitching_config)
        heuristic = stitch_overlapping_tracks(heuristic_raw, **self.stitching_config)
        contours = self._contours(learned, "learned") + self._contours(heuristic, "heuristic")
        project.set_inference_result(
            contours,
            model=self.model_identity,
            settings={
                **asdict(self.settings),
                "resolved_device": str(self.device),
                "runtime_contract": ("Studio-experimental-96k-v1" if adaptation else "CSR01-stage-I-deterministic-v2"),
                **adaptation,
                "runtime_kernel": self.runtime_kernel,
                "window_seconds": self.feature_config.window_seconds,
                "stride_seconds": stride,
                "presence_gating_default": False,
            },
        )
        if resume:
            state = json.loads(resume.read_text(encoding="utf-8"))
            state["complete"] = True
            state["completed_utc"] = utc_now()
            _atomic_json(resume, state)
        return InferenceResult(
            project=project,
            windows=len(starts),
            learned_contours=len(learned),
            heuristic_contours=len(heuristic),
            resumed_windows=resumed_windows,
            runtime_kernel=self.runtime_kernel,
        )

    @staticmethod
    def _contours(tracks: Iterable[Mapping[str, Any]], source: str) -> list[Contour]:
        result: list[Contour] = []
        for track in tracks:
            points = [
                ContourPoint(float(time_s), float(frequency_hz), float(confidence))
                for time_s, frequency_hz, confidence in zip(
                    track["time_s"], track["frequency_hz"], track["node_confidence"]
                )
            ]
            if points:
                result.append(Contour.create(points, source=source))  # type: ignore[arg-type]
        return result


def _bandlimit_outputs(outputs, keep_bins):
    spatial = {"centerline_logits", "endpoint_logits", "junction_logits", "embeddings", "tangent"}
    return {k: v[..., :keep_bins, :] if k in spatial else v for k, v in outputs.items()}
