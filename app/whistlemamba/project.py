from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import soundfile as sf

SCHEMA_VERSION = 1
ContourSource = Literal["learned", "heuristic", "manual"]
ReviewState = Literal["pending", "accepted", "rejected"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AudioIdentity:
    path: str
    sha256: str
    sample_rate_hz: int
    channels: int
    frames: int
    duration_s: float

    @classmethod
    def from_file(cls, path: str | Path) -> AudioIdentity:
        resolved = Path(path).expanduser().resolve()
        info = sf.info(resolved)
        return cls(
            path=str(resolved),
            sha256=sha256_file(resolved),
            sample_rate_hz=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
            duration_s=float(info.duration),
        )


@dataclass(frozen=True)
class ContourPoint:
    time_s: float
    frequency_hz: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.time_s < 0:
            raise ValueError("time_s cannot be negative")
        if self.frequency_hz < 0:
            raise ValueError("frequency_hz cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass
class Contour:
    contour_id: str
    source: ContourSource
    points: list[ContourPoint]
    review_state: ReviewState = "pending"
    notes: str = ""
    parent_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.contour_id:
            raise ValueError("contour_id is required")
        if len(self.points) < 1:
            raise ValueError("a contour requires at least one point")
        times = [point.time_s for point in self.points]
        if times != sorted(times):
            raise ValueError("contour points must be ordered by time_s")

    @classmethod
    def create(
        cls,
        points: Iterable[ContourPoint],
        *,
        source: ContourSource,
        contour_id: str | None = None,
        parent_ids: Iterable[str] = (),
    ) -> Contour:
        return cls(
            contour_id=contour_id or str(uuid.uuid4()),
            source=source,
            points=sorted(points, key=lambda point: point.time_s),
            parent_ids=list(parent_ids),
        )


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    checkpoint_sha256: str
    config_sha256: str
    backend: str
    kernel: str
    device: str
    canonical_state_dict_sha256: str
    normalization_identity_sha256: str


@dataclass
class AnalysisProject:
    audio: AudioIdentity
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = SCHEMA_VERSION
    created_utc: str = field(default_factory=utc_now)
    modified_utc: str = field(default_factory=utc_now)
    model: ModelIdentity | None = None
    inference_settings: dict[str, Any] = field(default_factory=dict)
    original_contours: list[Contour] = field(default_factory=list)
    edited_contours: list[Contour] = field(default_factory=list)
    edit_log: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    analysis_complete: bool = False
    official_evaluation_accessed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported project schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.official_evaluation_accessed:
            raise ValueError("public inference projects cannot access official evaluation data")

    @classmethod
    def create(cls, audio_path: str | Path) -> AnalysisProject:
        return cls(audio=AudioIdentity.from_file(audio_path))

    def set_inference_result(
        self,
        contours: Iterable[Contour],
        *,
        model: ModelIdentity,
        settings: Mapping[str, Any],
    ) -> None:
        originals = [contour_from_dict(contour_to_dict(item)) for item in contours]
        self.original_contours = originals
        self.edited_contours = [contour_from_dict(contour_to_dict(item)) for item in originals]
        self.model = model
        self.inference_settings = dict(settings)
        self.analysis_complete = True
        self.modified_utc = utc_now()

    def verify_audio(self) -> bool:
        path = Path(self.audio.path)
        return path.is_file() and sha256_file(path) == self.audio.sha256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        if destination == Path(self.audio.path).resolve():
            raise ValueError("Cannot overwrite source audio with a project")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.modified_utc = utc_now()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> AnalysisProject:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("project file must contain a JSON object")
        payload["audio"] = AudioIdentity(**payload["audio"])
        if payload.get("model") is not None:
            payload["model"] = ModelIdentity(**payload["model"])
            from .registry import validate_project_identity
            validate_project_identity(payload["model"])
        payload["original_contours"] = [
            contour_from_dict(item) for item in payload.get("original_contours", [])
        ]
        payload["edited_contours"] = [
            contour_from_dict(item) for item in payload.get("edited_contours", [])
        ]
        return cls(**payload)


def contour_to_dict(contour: Contour) -> dict[str, Any]:
    return asdict(contour)


def contour_from_dict(payload: Mapping[str, Any]) -> Contour:
    values = dict(payload)
    values["points"] = [ContourPoint(**point) for point in values.get("points", [])]
    return Contour(**values)
