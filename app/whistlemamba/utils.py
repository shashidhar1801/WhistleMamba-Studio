import contextlib
import torch

def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "float32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
