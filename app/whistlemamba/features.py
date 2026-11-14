from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 192_000
    n_fft_short: int = 2048
    win_length_short: int = 1536  # 8 ms at 192 kHz, zero-padded to n_fft_short
    n_fft_long: int = 4096
    win_length_long: int = 3072   # 16 ms at 192 kHz
    hop_length: int = 384         # 2 ms
    f_min_hz: float = 4_000.0
    f_max_hz: float = 96_000.0
    frequency_resolution_hz: float = 125.0
    window_seconds: float = 4.0
    db_floor: float = -160.0
    eps: float = 1e-8
    local_kernel_freq: int = 25
    local_kernel_time: int = 31
    normalization: Literal["robust_window", "global_zscore", "none"] = "robust_window"
    channel_means: tuple[float, float, float] | None = None
    channel_stds: tuple[float, float, float] | None = None

    @property
    def target_frames(self) -> int:
        return int(round(self.window_seconds * self.sample_rate / self.hop_length))

    @property
    def target_frequency_bins(self) -> int:
        return int(round((self.f_max_hz - self.f_min_hz) / self.frequency_resolution_hz)) + 1


def feature_config_signature(cfg: FeatureConfig) -> str:
    """Stable hash used to invalidate incompatible cached spectrograms."""
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class MultiResolutionSTFT(nn.Module):
    """Three-channel native-aspect spectrogram front end.

    Output channels are short-window log power, long-window log power, and
    local spectral contrast.  The two STFT branches are mapped to a common
    physical grid, so target rasterization and frequency-error metrics use the
    same coordinates.  Input shape may be ``[B,N]`` or ``[B,C_audio,N]``;
    multichannel power is max-pooled to avoid phase cancellation.
    """

    def __init__(self, cfg: FeatureConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "short_window", torch.hamming_window(cfg.win_length_short, periodic=True), persistent=False
        )
        self.register_buffer(
            "long_window", torch.hamming_window(cfg.win_length_long, periodic=True), persistent=False
        )
        if cfg.normalization == "global_zscore":
            if cfg.channel_means is None or cfg.channel_stds is None:
                raise ValueError("global_zscore requires channel_means and channel_stds from training only")
            self.register_buffer("means", torch.tensor(cfg.channel_means).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("stds", torch.tensor(cfg.channel_stds).view(1, 3, 1, 1), persistent=False)
        else:
            self.register_buffer("means", torch.zeros(1, 3, 1, 1), persistent=False)
            self.register_buffer("stds", torch.ones(1, 3, 1, 1), persistent=False)

    def frequency_axis(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.cfg.f_min_hz + torch.arange(
            self.cfg.target_frequency_bins, device=device, dtype=torch.float32
        ) * self.cfg.frequency_resolution_hz

    def time_axis(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.arange(self.cfg.target_frames, device=device, dtype=torch.float32) * (
            self.cfg.hop_length / self.cfg.sample_rate
        )

    def grid(self, device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.frequency_axis(device), self.time_axis(device)

    @staticmethod
    def _as_bcn(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            return waveform[:, None, :]
        if waveform.ndim == 3:
            return waveform
        raise ValueError(f"waveform must be [B,N] or [B,C,N], got {tuple(waveform.shape)}")

    def _power(
        self,
        waveform: torch.Tensor,
        *,
        n_fft: int,
        win_length: int,
        window: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, samples = waveform.shape
        flattened = waveform.reshape(batch * channels, samples)
        spectrum = torch.stft(
            flattened,
            n_fft=n_fft,
            hop_length=self.cfg.hop_length,
            win_length=win_length,
            window=window.to(flattened.device, flattened.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spectrum.abs().square().reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])
        power = power.amax(dim=1)
        if power.shape[-1] < self.cfg.target_frames:
            power = F.pad(power, (0, self.cfg.target_frames - power.shape[-1]))
        return power[..., : self.cfg.target_frames]

    def _select_band_and_resize(self, power: torch.Tensor, n_fft: int) -> torch.Tensor:
        """Interpolate power onto the exact physical frequency grid.

        Index-based image resizing can shift frequencies because the first FFT
        bin inside a band is rarely exactly ``f_min_hz``. This implementation
        interpolates using FFT-bin frequencies, preserving Hz coordinates for
        contour rasterization and frequency-error reporting.
        """
        source = torch.fft.rfftfreq(
            n_fft, d=1.0 / self.cfg.sample_rate, device=power.device
        )
        target = self.frequency_axis(power.device).to(source.dtype)
        upper = torch.searchsorted(source, target, right=False).clamp(1, source.numel() - 1)
        lower = upper - 1
        lower_frequency = source[lower]
        upper_frequency = source[upper]
        weight = (target - lower_frequency) / (upper_frequency - lower_frequency).clamp_min(1e-12)
        lower_power = power.index_select(1, lower)
        upper_power = power.index_select(1, upper)
        return lower_power + weight.view(1, -1, 1) * (upper_power - lower_power)

    def _db(self, power: torch.Tensor) -> torch.Tensor:
        return (10.0 * torch.log10(power.clamp_min(self.cfg.eps))).clamp_min(self.cfg.db_floor)

    def _local_contrast(self, db: torch.Tensor) -> torch.Tensor:
        kf, kt = self.cfg.local_kernel_freq, self.cfg.local_kernel_time
        if kf % 2 == 0 or kt % 2 == 0:
            raise ValueError("local contrast kernels must be odd")
        x = db[:, None]
        mean = F.avg_pool2d(x, (kf, kt), stride=1, padding=(kf // 2, kt // 2))
        mean_square = F.avg_pool2d(x.square(), (kf, kt), stride=1, padding=(kf // 2, kt // 2))
        std = (mean_square - mean.square()).clamp_min(1e-4).sqrt()
        return ((x - mean) / std).squeeze(1).clamp(-6.0, 6.0)

    @staticmethod
    def _robust_scale(channel: torch.Tensor) -> torch.Tensor:
        flattened = channel.flatten(1)
        median = torch.quantile(flattened, 0.50, dim=1, keepdim=True).view(-1, 1, 1)
        upper = torch.quantile(flattened, 0.95, dim=1, keepdim=True).view(-1, 1, 1)
        return ((channel - median) / (upper - median).clamp_min(1.0)).clamp(-5.0, 5.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._as_bcn(waveform).float()
        short = self._select_band_and_resize(
            self._power(
                waveform,
                n_fft=self.cfg.n_fft_short,
                win_length=self.cfg.win_length_short,
                window=self.short_window,
            ),
            self.cfg.n_fft_short,
        )
        long = self._select_band_and_resize(
            self._power(
                waveform,
                n_fft=self.cfg.n_fft_long,
                win_length=self.cfg.win_length_long,
                window=self.long_window,
            ),
            self.cfg.n_fft_long,
        )
        short_db, long_db = self._db(short), self._db(long)
        contrast = self._local_contrast(short_db)
        if self.cfg.normalization == "robust_window":
            features = torch.stack(
                [self._robust_scale(short_db), self._robust_scale(long_db), contrast], dim=1
            )
        else:
            features = torch.stack([short_db, long_db, contrast], dim=1)
            if self.cfg.normalization == "global_zscore":
                features = (features - self.means.to(features)) / self.stds.to(features).clamp_min(1e-5)
        return features
