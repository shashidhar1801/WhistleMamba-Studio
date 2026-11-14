"""Small display tiles in recording time. These are not inference windows."""
import base64
import io
import math
from functools import lru_cache
import numpy as np
import soundfile as sf
from PIL import Image
from matplotlib import colormaps
from whistlemamba.spectrogram import SpectrogramSettings
from .preview import display_tile

TILE = 5
DB_MIN, DB_MAX = -110, -10


@lru_cache(maxsize=1)
def display_palettes():
    """Exact 256-entry RGB tables, shipped locally; never applied to model data."""
    return {name: base64.b64encode(colormaps[name](np.arange(256),bytes=True)[:,:3].tobytes()).decode("ascii")
            for name in ("magma","viridis","cividis","inferno","gray")}


def tile(row, index):
    info = row["project"].audio
    if type(index) is not int or not 0 <= index < math.ceil(info.duration_s/TILE):
        raise ValueError("Outside the recording.")
    rate = info.sample_rate_hz
    start = index*TILE
    samples = min(TILE*rate, info.frames-start*rate)
    span = samples/rate
    hi = min(96000, rate/2)
    lo = 4000 if hi > 4100 else 0
    # Fixed physical color scale: no gain jumps when moving onto another tile.
    if samples >= 2:
        t = display_tile(row["path"], start, span, SpectrogramSettings(f_min_hz=lo, f_max_hz=hi), max_bins=480)
        level = np.rint(255*np.clip((t.magnitude_db[::-1]-DB_MIN)/(DB_MAX-DB_MIN),0,1)).astype(np.uint8)
        im = Image.fromarray(level).resize((min(480,level.shape[1]),240),Image.Resampling.BOX)
    else:
        im = Image.new("L",(1,240),0)
    b = io.BytesIO(); im.save(b,format="PNG")
    # Envelope retains positive and negative peaks from all channels; bounded reads.
    step=max(1, math.ceil(samples/480)); env=[]
    with sf.SoundFile(row["path"]) as src:
        src.seek(start*rate)
        remaining=samples
        while remaining:
            take=min(remaining,step*max(1, min(32,100000//(step*info.channels))))
            a=src.read(take,dtype="float32",always_2d=True)
            for i in range(0,len(a),step):
                part=a[i:i+step]
                env.append([round(float(part.min()),5),round(float(part.max()),5)])
            remaining-=take
    return dict(start=start,end=start+span,low=lo,high=hi,step=step/rate,wave=env,
                png=base64.b64encode(b.getvalue()).decode("ascii"))
