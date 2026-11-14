"""Bounded tests. Set WHISTLEMAMBA_CHECKPOINT only to an authorized local SafeTensors asset."""
from pathlib import Path
import json, os
import pytest
import numpy as np
import soundfile as sf
from whistlemamba.inference import InferenceSettings,WhistleMambaPredictor,InferenceCancelled
from whistlemamba.registry import IDENTITY
from whistlemamba.project import AnalysisProject,sha256_file,ContourPoint,Contour
from whistlemamba.editing import ContourEditor
from whistlemamba.export import export_bundle

demo=None

asset=os.getenv("WHISTLEMAMBA_CHECKPOINT")
def test_reject_missing_and_corrupt(tmp_path):
    with pytest.raises(FileNotFoundError):WhistleMambaPredictor(checkpoint_path=tmp_path/"missing")
    bad=tmp_path/"bad";bad.write_bytes(b"not weights")
    with pytest.raises(ValueError,match="SHA-256"):WhistleMambaPredictor(checkpoint_path=bad)
def test_grid_centres():
    from whistlemamba_studio.spectrogram_view import grid_rect
    t=np.arange(2000)*0.002+2.125;f=4000+np.arange(737)*125
    rect=grid_rect(t,f)
    np.testing.assert_allclose(rect.x()+(np.arange(len(t))+.5)*rect.width()/len(t),t)
    np.testing.assert_allclose(rect.y()+(np.arange(len(f))+.5)*rect.height()/len(f),f)
def test_edit_history_and_input_protection(tmp_path):
    p=AnalysisProject.create(demo); e=ContourEditor(p)
    c=Contour.create([ContourPoint(0,4000),ContourPoint(.1,4200)],source="manual")
    e.add_contour(c);e.move_point(c.contour_id,1,ContourPoint(.1,4300))
    assert e.undo() and p.edited_contours[0].points[-1].frequency_hz==4200
    assert e.redo() and p.edited_contours[0].points[-1].frequency_hz==4300
    assert p.original_contours==[] and p.verify_audio()
    with pytest.raises(ValueError):p.save(demo)
    assert AnalysisProject.load(p.save(tmp_path/"edit.json")).edited_contours==p.edited_contours
@pytest.mark.skipif(not asset,reason="requires authorized local inference weights")
def test_resume_and_identity(tmp_path):
    from whistlemamba.registry import config_resource
    bad=tmp_path/"config.json";bad.write_text(config_resource().read_text()+" ")
    with pytest.raises(ValueError,match="configuration"):WhistleMambaPredictor(config_path=bad,checkpoint_path=asset)
    engine=WhistleMambaPredictor(checkpoint_path=asset,settings=InferenceSettings(device="cuda"))
    raw,sr=sf.read(demo,dtype="float32");wav=tmp_path/"synthetic-six.wav"
    sf.write(wav,np.concatenate([raw,raw[:2*sr]]),sr,subtype="PCM_16")
    original=sha256_file(wav);resume=tmp_path/"resume.json"; progress=[]
    unsupported=tmp_path/"short.wav";sf.write(unsupported,np.zeros(1000),sr)
    with pytest.raises(ValueError,match="Supported input"):engine.analyze_file(unsupported)
    with pytest.raises(InferenceCancelled):
        engine.analyze_file(wav,resume_path=resume,progress=progress.append,cancelled=lambda:bool(progress))
    state=json.loads(resume.read_text());assert not state["complete"] and len(state["completed_window_starts_s"])==1
    state["identity"]["deployment"]["model_id"]="different-model"
    wrong=tmp_path/"wrong.json";wrong.write_text(json.dumps(state))
    with pytest.raises(ValueError,match="different audio, model"):engine.analyze_file(wav,resume_path=wrong)
    result=engine.analyze_file(wav,resume_path=resume)
    assert result.windows==2 and result.resumed_windows==1 and sha256_file(wav)==original
    p=result.project;assert p.model.model_id==IDENTITY["model_id"] and not p.inference_settings["presence_gating"]
    x=p.to_dict();x["model"]["normalization_identity_sha256"]="bad"
    wrong.write_text(json.dumps(x))
    with pytest.raises(ValueError,match="not bound"):AnalysisProject.load(wrong)
    p.model=None
    with pytest.raises(ValueError,match="not bound"):export_bundle(p,tmp_path/"unexpected")
