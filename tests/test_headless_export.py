"""Regression: file-only exports in a request thread must never start a GUI."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def test_worker_export_does_not_select_or_start_gui_backend(tmp_path, synthetic_wav):
    import matplotlib
    from whistlemamba.export import export_figure
    from whistlemamba.project import AnalysisProject, ModelIdentity
    from whistlemamba.registry import IDENTITY

    # An unimportable GUI backend makes the old pyplot path fail safely, even
    # on display-equipped machines. Do not mask the bug by forcing Agg globally.
    unavailable = "module://studio_gui_backend_must_not_be_loaded"
    wav = synthetic_wav
    project = AnalysisProject.create(wav)
    keys = ("model_id", "checkpoint_sha256", "config_sha256",
            "canonical_state_dict_sha256", "normalization_identity_sha256")
    # Export-only fixture, deliberately no inference or fabricated detections.
    project.model = ModelIdentity(**{k: IDENTITY[k] for k in keys},
                                  backend="mamba2", kernel="cuda_fused", device="cuda")
    project.analysis_complete = True
    original = matplotlib.get_backend(auto_select=False)
    try:
        matplotlib.rcParams["backend"] = unavailable
        with ThreadPoolExecutor(max_workers=1) as pool:
            for kind, magic in (("png", b"\x89PNG"), ("pdf", b"%PDF-")):
                target = tmp_path / f"overlay.{kind}"
                pool.submit(export_figure, project, target, dpi=72).result(timeout=45)
                assert target.read_bytes().startswith(magic)
        assert matplotlib.get_backend(auto_select=False) == unavailable
    finally:
        if original is None:
            from matplotlib import rcsetup
            matplotlib.rcParams["backend"] = rcsetup._auto_backend_sentinel
        else:
            matplotlib.rcParams["backend"] = original
