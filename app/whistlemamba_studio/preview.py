"""Display-only preview grid: independent of the detector's four-second inputs."""
from math import ceil, isfinite
import numpy as np
import soundfile as sf
from whistlemamba.spectrogram import SpectrogramSettings, SpectrogramTile, compute_spectrogram_tile

VIEW_SECONDS = 15.

def viewport(seconds, start):
    if not isfinite(start):
        raise ValueError("Invalid preview position.")
    span = min(VIEW_SECONDS, seconds)
    return max(0., min(start, seconds-span)), span

def display_tile(path, start, span, settings, max_bins=1500):
    """Pool consecutive STFT columns, preserving the last partial display bin.

    Read overlapping blocks on one global hop grid so channel count or sample
    rate limits memory use, not how many seconds the viewer can see.
    """
    info = sf.info(path)
    rate = info.samplerate
    first = round(start*rate)
    samples = min(info.frames-first, round(span*rate))
    if samples < 2:
        raise ValueError("The selected audio is too short to display.")
    win = min(samples, settings.win_length)
    hop = min(win, settings.hop_length)
    frames = 1+(samples-win)//hop
    pool = max(1, ceil(frames/max_bins))
    columns = ceil(frames/pool)
    batch = max(1, min(256, (500000//info.channels-win)//hop+1))
    out = None
    for offset in range(0, frames, batch):
        count = min(batch, frames-offset)
        tile = compute_spectrogram_tile(path, start_s=(first+offset*hop)/rate,
            duration_s=(win+(count-1)*hop)/rate, settings=settings, maximum_time_bins=count)
        if out is None:
            freq = tile.frequency_hz
            out = np.full((len(freq),columns), -np.inf, dtype=np.float32)
        # Peak power per display column, across every real frame (including the tail).
        bins = np.arange(offset, offset+count)//pool
        begins = np.r_[0, np.flatnonzero(np.diff(bins))+1]
        block = np.maximum.reduceat(tile.magnitude_db, begins, axis=1)
        targets = bins[begins]
        out[:, targets] = np.maximum(out[:, targets], block)
    centers = np.arange(columns)*pool + (np.minimum(pool, frames-np.arange(columns)*pool)-1)/2
    times = (first+win/2+centers*hop)/rate
    peak = float(np.percentile(out,99.5)) if out.size else 0.
    return SpectrogramTile(start, start+span, freq, times, out, peak-settings.dynamic_range_db, peak)
