import numpy as np
import pytest
import soundfile as sf
from whistlemamba.spectrogram import SpectrogramSettings, compute_spectrogram_tile
from whistlemamba_studio.preview import display_tile, viewport

def test_viewports():
    assert viewport(22.18,0)==(0,15)
    assert viewport(22.18,40)==pytest.approx((7.18,15))
    assert viewport(4,7)==(0,4)
    with pytest.raises(ValueError): viewport(10,float("nan"))

def test_blocks_keep_full_span_and_last_frame(tmp_path):
    rate=96000; path=tmp_path/"stereo.wav"
    x=np.random.default_rng(17).normal(0,.1,(rate*17,2)).astype("float32")
    sf.write(path,x,rate,subtype="FLOAT")
    s=SpectrogramSettings(f_max_hz=48000)
    t=display_tile(path,2.,15.,s,max_bins=325)
    baseline=compute_spectrogram_tile(path,start_s=2.,duration_s=15.,settings=s,maximum_time_bins=10000)
    n=baseline.magnitude_db.shape[1];stride=int(np.ceil(n/325))
    np.testing.assert_allclose(t.magnitude_db,np.maximum.reduceat(baseline.magnitude_db,np.arange(0,n,stride),axis=1),atol=1e-4)
    assert t.start_s==2 and t.end_s==17 and t.time_s[-1]>16.9
