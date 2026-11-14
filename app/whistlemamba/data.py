from pathlib import Path
import math
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly

def load_audio_window(
    path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    target_sample_rate: int,
) -> torch.Tensor:
    path = Path(path)
    with sf.SoundFile(path) as handle:
        source_rate = int(handle.samplerate)
        start_frame = max(0, int(round(start_s * source_rate)))
        frames = int(math.ceil(duration_s * source_rate))
        handle.seek(min(start_frame, len(handle)))
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
    audio = np.asarray(audio, dtype=np.float32).T
    if source_rate != target_sample_rate:
        divisor = math.gcd(source_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            up=target_sample_rate // divisor,
            down=source_rate // divisor,
            axis=-1,
        ).astype(np.float32, copy=False)
    expected = int(round(duration_s * target_sample_rate))
    if audio.shape[-1] < expected:
        audio = np.pad(audio, ((0, 0), (0, expected - audio.shape[-1])))
    return torch.from_numpy(audio[..., :expected])
