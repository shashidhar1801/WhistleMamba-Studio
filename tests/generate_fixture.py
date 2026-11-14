#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def _sweep(
    time: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    start_hz: float,
    end_hz: float,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = (time >= start_s) & (time <= end_s)
    local = (time[active] - start_s) / (end_s - start_s)
    frequency = start_hz + (end_hz - start_hz) * (3 * local**2 - 2 * local**3)
    phase = 2 * np.pi * np.cumsum(frequency) / 192_000
    envelope = np.sin(np.pi * local) ** 2
    signal = np.zeros_like(time)
    signal[active] = amplitude * envelope * np.sin(phase)
    return signal, time[active], frequency


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description="Generate temporary audio for the test suite.")
    parser.add_argument("output",type=Path)
    output=parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = 192_000
    duration_s = 4.0
    time = np.arange(round(sample_rate * duration_s), dtype=np.float64) / sample_rate
    rng = np.random.default_rng(17)
    waveform = rng.normal(0.0, 0.0025, size=time.shape)
    contours = [
        ("synthetic_rising", 0.35, 2.65, 9_000.0, 42_000.0, 0.22),
        ("synthetic_falling", 1.05, 3.55, 48_000.0, 14_000.0, 0.18),
        ("synthetic_short", 2.85, 3.70, 26_000.0, 35_000.0, 0.14),
    ]
    rows: list[dict[str, object]] = []
    for contour_id, start_s, end_s, start_hz, end_hz, amplitude in contours:
        component, contour_time, contour_frequency = _sweep(
            time,
            start_s=start_s,
            end_s=end_s,
            start_hz=start_hz,
            end_hz=end_hz,
            amplitude=amplitude,
        )
        waveform += component
        sample_indices = np.linspace(0, len(contour_time) - 1, 220, dtype=int)
        rows.extend(
            {
                "contour_id": contour_id,
                "node_index": node_index,
                "time_s": f"{contour_time[index]:.9f}",
                "frequency_hz": f"{contour_frequency[index]:.3f}",
                "provenance": "deterministic_synthetic_not_DCLDE",
            }
            for node_index, index in enumerate(sample_indices)
        )
    waveform = np.clip(waveform, -0.98, 0.98).astype(np.float32)
    wav_path = output / "synthetic_whistles_4s.wav"
    csv_path = output / "synthetic_whistles_4s_contours.csv"
    sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema_version": 1,
        "description": "Deterministic synthetic crossing whistle demo; no corpus audio used.",
        "seed": 17,
        "sample_rate_hz": sample_rate,
        "duration_s": duration_s,
        "license": "CC0-1.0",
        "wav_sha256": _sha256(wav_path),
        "contours_sha256": _sha256(csv_path),
        "official_evaluation_accessed": False,
    }
    (output / "synthetic_whistles_4s_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
