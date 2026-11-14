"""Single private CSR01 deployment; no network acquisition or legacy fallback."""
from importlib.resources import files
from hashlib import sha256
import json
from types import MappingProxyType

def config_resource():
    return files("whistlemamba.resources").joinpath("csr01_inference.json")

_bytes=files("whistlemamba.resources").joinpath("csr01_identity.json").read_bytes()
IDENTITY=MappingProxyType(json.loads(_bytes))
if IDENTITY["canonical_state_dict_sha256"] != "c08762340840e7eaaddc7ea5a3a45693e823916748f4b7bc68e10b5c8b228cc5":
    raise RuntimeError("Unrecognized corrected tensor identity")

def validate_project_identity(model):
    if not model or any(getattr(model,k,None)!=IDENTITY[k] for k in (
        "model_id","checkpoint_sha256","config_sha256","canonical_state_dict_sha256","normalization_identity_sha256"
    )) or (model.backend,model.kernel,model.device)!=("mamba2","cuda_fused","cuda"):
        raise ValueError("Project is not bound to this corrected model/configuration/backend")
