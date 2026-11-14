# Third-party components

| Component | Role | Upstream license |
|---|---|---|
| Python | Runtime | PSF-2.0 |
| NumPy | arrays/numerics | BSD-3-Clause |
| SciPy | signal processing/assignment | BSD-3-Clause |
| SoundFile / libsndfile | WAV I/O | BSD-3-Clause / LGPL-2.1-or-later |
| PyYAML | configuration | MIT |
| PyTorch | tensor/model runtime | BSD-style |
| safetensors | inference-only model load | Apache-2.0 |
| einops | tensor rearrangement dependency | MIT |
| Matplotlib | static exports | Matplotlib license (PSF-compatible) |
| PySide6 / Qt for Python | desktop bindings | LGPL-3.0/GPL-3.0 or commercial |
| pyqtgraph | interactive scientific graphics | MIT |
| mamba-ssm | fused Mamba-2 CUDA implementation | Apache-2.0 |
| causal-conv1d | fused causal convolution | BSD-3-Clause |

Third-party components retain their respective licenses. The Windows installer uses the versions and hashes in [the runtime lock](runtime/runtime.lock.json).
