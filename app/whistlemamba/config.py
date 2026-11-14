from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML object must be a mapping: {path}")
    return data


def dataclass_from_mapping(cls: type[T], values: Mapping[str, Any] | None) -> T:
    """Construct a dataclass while rejecting unknown keys.

    Unknown configuration keys are treated as errors because silent typos can
    invalidate a manuscript experiment.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    values = dict(values or {})
    valid = {field.name for field in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)
