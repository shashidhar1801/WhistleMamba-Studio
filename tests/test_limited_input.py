import numpy as np
import torch
from whistlemamba.input_mode import rate_supported, compatibility_metadata
from whistlemamba.inference import _bandlimit_outputs

def test_source_rate_contract():
    assert [rate_supported(v) for v in (48000,95999,96000,96001,176400,192000,384000)] == [False,False,True,False,False,True,True]
    assert compatibility_metadata(192000)=={}
    assert compatibility_metadata(96000)["detection_band_hz"] == [4000,48000]

def test_bandlimit_before_node_ranking():
    grid=np.arange(4000,96001,125)
    stop=int(np.searchsorted(grid,48000,side="right"))
    prob=torch.zeros(2,1,len(grid),11);prob[:,:,stop-1]=1;prob[:,:,stop:]=3
    presence=torch.zeros(2)
    keys=("centerline_logits","tangent","embeddings","endpoint_logits","junction_logits")
    out=_bandlimit_outputs({**dict.fromkeys(keys,prob),"presence_logit":presence},stop)
    for key in keys:
        assert out[key].shape==(2,1,353,11) and out[key].max().item()==1
        assert grid[out[key][0,0,:,0].argmax().item()]==48000
    assert out["presence_logit"] is presence and prob.shape[2]==737
