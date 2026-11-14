"""No recording is opened. The production predictor must accept the asset and GPU."""
import json, os, subprocess, sys
from pathlib import Path

if __name__=='__main__':
    mode, chosen, model=sys.argv[1:]
    if mode=='locate':
        interpreter=Path.home()/'.local/share/whistlemamba-studio/csr01-e12/venv/bin/python' if chosen=='auto' else Path(chosen)
        if not interpreter.is_absolute() or not interpreter.is_file():sys.exit('No detector Python at '+str(interpreter)+'. Install the runtime or specify an existing WSL Python.')
        os.execv(str(interpreter),[str(interpreter),'-B','-u',__file__,'verify',str(interpreter),model])
    if mode!='verify':sys.exit('Unknown configuration action.')
    from whistlemamba.inference import InferenceSettings,WhistleMambaPredictor
    import torch
    detector=WhistleMambaPredictor(checkpoint_path=model,settings=InferenceSettings(device='cuda',tracker_mode='learned',batch_size=1))
    assert detector.runtime_kernel=='cuda_fused' and detector.model.effective_backend=='mamba2'
    print('STUDIO_CONFIG_OK '+json.dumps({'python':chosen,'weights':detector.checkpoint_sha256,'gpu':torch.cuda.get_device_name(0)}),flush=True)
