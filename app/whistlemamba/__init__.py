"""WhistleMamba corrected-split inference library."""
import os
import sys

# The CSR01 launcher set these *before* importing Mamba/Triton. The extension's
# decorators select their kernel configurations at import, not at model load.
for _name, _value in {"MAMBA_DETERMINISTIC":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"}.items():
    _existing=os.environ.get(_name)
    if _existing not in (None,_value) or (_existing is None and
        ("mamba_ssm" in sys.modules or ("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()))):
        raise RuntimeError(f"Start a fresh CSR01 process with {_name}={_value} before importing CUDA/Mamba")
    os.environ[_name]=_value
if os.environ.get("TRITON_CACHE_AUTOTUNING")=="1" or any(k.startswith("TRITON_AUTOTUNE_BLOCK_SIZE_") for k in os.environ):
    raise RuntimeError("CSR01 does not permit alternate Triton autotuning settings")
__version__ = "0.2.0b2"
