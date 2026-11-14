from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


@dataclass(frozen=True)
class SpectrogramSettings:
    n_fft: int = 2048
    win_length: int = 1536
    hop_length: int = 384
    f_min_hz: float = 4_000.0
    f_max_hz: float = 96_000.0
    dynamic_range_db: float = 80.0
    channel: int | None = None

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("FFT, window, and hop lengths must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if not 0 <= self.f_min_hz < self.f_max_hz:
            raise ValueError("frequency limits must satisfy 0 <= min < max")
        if self.dynamic_range_db <= 0:
            raise ValueError("dynamic range must be positive")


@dataclass(frozen=True)
class SpectrogramTile:
    start_s: float
    end_s: float
    frequency_hz: np.ndarray
    time_s: np.ndarray
    magnitude_db: np.ndarray
    display_min_db: float
    display_max_db: float


def read_waveform_region(
    path: str | Path,
    *,
    start_s: float,
    duration_s: float,
) -> tuple[np.ndarray, int]:
    if start_s < 0 or duration_s <= 0:
        raise ValueError("start_s must be non-negative and duration_s must be positive")
    with sf.SoundFile(path) as handle:
        sample_rate = int(handle.samplerate)
        start_frame = min(round(start_s * sample_rate), len(handle))
        frames = max(1, round(duration_s * sample_rate))
        handle.seek(start_frame)
        waveform = handle.read(frames, dtype="float32", always_2d=True)
    return np.asarray(waveform, dtype=np.float32), sample_rate


def compute_spectrogram_tile(
    path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    settings: SpectrogramSettings | None = None,
    maximum_time_bins: int = 4_000,
) -> SpectrogramTile:
    settings = settings or SpectrogramSettings()
    waveform, sample_rate = read_waveform_region(path, start_s=start_s, duration_s=duration_s)
    if waveform.shape[0] < 2:
        raise ValueError("audio region is too short for spectrogram rendering")
    if settings.channel is not None:
        if not 0 <= settings.channel < waveform.shape[1]:
            raise ValueError(f"channel {settings.channel} is unavailable")
        channels = waveform[:, settings.channel : settings.channel + 1]
    else:
        channels = waveform
    nperseg = min(settings.win_length, waveform.shape[0])
    nfft = max(settings.n_fft, nperseg)
    noverlap = min(nperseg - 1, max(0, nperseg - settings.hop_length))
    channel_power: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    times: np.ndarray | None = None
    for channel in range(channels.shape[1]):
        frequencies, times, spectrum = stft(
            channels[:, channel],
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
            padded=False,
        )
        channel_power.append(np.square(np.abs(spectrum)))
    assert frequencies is not None and times is not None
    power = np.max(np.stack(channel_power), axis=0)
    band = (frequencies >= settings.f_min_hz) & (
        frequencies <= min(settings.f_max_hz, sample_rate / 2)
    )
    frequencies = frequencies[band]
    power = power[band]
    if power.shape[1] > maximum_time_bins:
        factor = int(np.ceil(power.shape[1] / maximum_time_bins))
        usable = power.shape[1] // factor * factor
        power = power[:, :usable].reshape(power.shape[0], -1, factor).max(axis=2)
        times = times[:usable].reshape(-1, factor).mean(axis=1)
    magnitude_db = 10.0 * np.log10(np.maximum(power, 1e-16))
    maximum = float(np.percentile(magnitude_db, 99.5)) if magnitude_db.size else 0.0
    minimum = maximum - settings.dynamic_range_db
    return SpectrogramTile(
        start_s=start_s,
        end_s=start_s + duration_s,
        frequency_hz=frequencies,
        time_s=times + start_s,
        magnitude_db=magnitude_db,
        display_min_db=minimum,
        display_max_db=maximum,
    )


class SpectrogramTileCache:
    """Small LRU cache for viewport-sized, long-recording spectrogram tiles."""

    def __init__(self, *, maximum_tiles: int = 8):
        if maximum_tiles < 1:
            raise ValueError("maximum_tiles must be positive")
        self.maximum_tiles = maximum_tiles
        self._tiles: OrderedDict[tuple[object, ...], SpectrogramTile] = OrderedDict()

    def get(
        self,
        path: str | Path,
        *,
        start_s: float,
        duration_s: float,
        settings: SpectrogramSettings,
        maximum_time_bins: int = 4_000,
    ) -> SpectrogramTile:
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        key = (
            str(resolved),
            stat.st_size,
            stat.st_mtime_ns,
            round(start_s, 6),
            round(duration_s, 6),
            settings,
            maximum_time_bins,
        )
        if key in self._tiles:
            self._tiles.move_to_end(key)
            return self._tiles[key]
        tile = compute_spectrogram_tile(
            resolved,
            start_s=start_s,
            duration_s=duration_s,
            settings=settings,
            maximum_time_bins=maximum_time_bins,
        )
        self._tiles[key] = tile
        self._tiles.move_to_end(key)
        while len(self._tiles) > self.maximum_tiles:
            self._tiles.popitem(last=False)
        return tile

    def clear(self) -> None:
        self._tiles.clear()
