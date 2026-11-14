"""Pipe bridge to the unchanged local predictor. Only invoked by Run analysis."""
import json
import sys
from pathlib import Path

import numpy as np
from whistlemamba.inference import InferenceSettings, InferenceCancelled, WhistleMambaPredictor
from whistlemamba.project import AnalysisProject


def run(directory):
    root = Path(directory)
    data = json.loads((root / "request.json").read_text(encoding="utf-8"))
    (root / "input.json").write_text(json.dumps(data["project"]), encoding="utf-8")
    project = AnalysisProject.load(root / "input.json")
    p = data["paths"]
    def send(event, **kw):
        print("STUDIO:" + json.dumps(dict(event=event, **kw), allow_nan=False), flush=True)
    def cancelled():
        return (root / "cancel").exists()
    def progress(e):
        send("progress", value=dict(completed_windows=e.completed_windows, total_windows=e.total_windows,
                                    fraction=e.fraction, window_start_s=e.window_start_s, phase=e.phase))
    index = 0
    def overlay(o):
        nonlocal index
        index += 1
        np.savez(root / f"overlay-{index}.npz", **vars(o))
        send("overlay", index=index)
    try:
        if cancelled():
            raise InferenceCancelled()
        predictor = WhistleMambaPredictor(config_path=p["config"], checkpoint_path=p["checkpoint"],
                                          settings=InferenceSettings(**data["settings"]))
        r = predictor.analyze_file(p["audio"], project=project, progress=progress, overlay=overlay,
                                   cancelled=cancelled, resume_path=p["resume"])
    except InferenceCancelled:
        send("cancelled")
        return
    r.project.save(root / "result.json")
    send("completed", counts=dict(learned_contours=r.learned_contours, heuristic_contours=r.heuristic_contours))


if __name__ == "__main__":
    run(sys.argv[1])
