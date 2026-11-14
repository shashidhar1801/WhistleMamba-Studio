from __future__ import annotations
import struct, hashlib
from typing import Any, Mapping
import torch

def _length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)

def canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted tensor identity and content, independent of torch serialization.

    Each entry contributes its UTF-8 name, dtype string, rank and little-endian
    int64 shape, followed by the raw bytes of a contiguous CPU tensor. Length
    prefixes make the stream unambiguous. Sparse and quantized tensors are
    rejected because the Milestone I model state is expected to be dense.
    """
    digest = hashlib.sha256()
    digest.update(b"whistlemamba-canonical-state-dict-v1\0")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor: {type(tensor)!r}")
        if tensor.is_sparse or tensor.is_quantized:
            raise TypeError(f"state entry {name!r} must be a dense, non-quantized tensor")
        value = tensor.detach().cpu().contiguous()
        _length_prefixed(digest, name.encode("utf-8"))
        _length_prefixed(digest, str(value.dtype).encode("ascii"))
        digest.update(struct.pack("<Q", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<q", int(dimension)))
        raw = value.view(torch.uint8).numpy().tobytes(order="C")
        _length_prefixed(digest, raw)
    return digest.hexdigest()
