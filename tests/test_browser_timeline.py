import io
import base64
import threading
import concurrent.futures
import numpy as np
import soundfile as sf
import pytest
from whistlemamba_studio import web
from whistlemamba_studio.web_timeline import display_palettes
from PIL import Image


def recording(tmp_path, frames=960001):
    stream=io.BytesIO()
    sf.write(stream,np.zeros(frames,dtype="float32"),96000,format="WAV")
    n=stream.tell();stream.seek(0)
    ws=web.Workspace(tmp_path,None)
    return ws,ws.receive(stream,n,"generated.wav")


def test_tiles_are_adjacent_and_reach_last_sample(tmp_path):
    ws,k=recording(tmp_path)
    a,b,c=[ws.render_tile(k,i) for i in range(3)]
    assert [(x["start"],x["end"]) for x in (a,b,c)]==[(0,5),(5,10),(10,960001/96000)]
    assert all(x["low"]==4000 and x["high"]==48000 and x["png"] and x["wave"] for x in (a,b,c))
    with pytest.raises(ValueError):ws.render_tile(k,3)


def test_display_is_scalar_silent_and_palette_independent(tmp_path):
    ws,k=recording(tmp_path,96000)
    data=ws.render_tile(k,0)
    im=Image.open(io.BytesIO(base64.b64decode(data["png"])))
    assert im.mode=="L" and im.getextrema()==(0,0)
    pals=display_palettes()
    assert set(pals)=={"magma","viridis","cividis","inferno","gray"}
    assert len(set(pals.values()))==5
    for b in pals.values():assert len(base64.b64decode(b))==256*3
    gray=np.frombuffer(base64.b64decode(pals["gray"]),dtype=np.uint8).reshape(256,3)
    assert np.all(gray==gray[:,:1])
    assert gray[0,0]==0 and gray[-1,0]==255 and np.all(np.diff(gray[:,0].astype(int))>=0)
    assert ws.snapshot(k)["phase"]=="ready" and ws.snapshot(k)["tracks"]==[]


def test_remove_inflight_tile_keeps_reader_alive(tmp_path,monkeypatch):
    ws,k=recording(tmp_path,10000)
    file=ws.rows[k]["path"]
    entered=threading.Event();release=threading.Event()
    def read(row,index):
        entered.set();assert release.wait(10);return row["path"].is_file()
    monkeypatch.setattr(web,"tile",read)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        f=pool.submit(ws.render_tile,k,0)
        assert entered.wait(10)
        try:
            ws.remove(k)
            assert k not in ws.rows and file.is_file() and ws.reserved>0
        finally:release.set()
        assert f.result(timeout=10)
    assert not file.exists() and ws.reserved==0
