from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:  # Optional production dependency.
    from mamba_ssm import Mamba2 as OfficialMamba2
except Exception:  # pragma: no cover - depends on the CUDA environment
    OfficialMamba2 = None


@dataclass(frozen=True)
class WhistleMambaConfig:
    variant_name: str = "WhistleMamba-Track-Base"
    in_channels: int = 3
    channels: tuple[int, int, int, int] = (48, 96, 192, 320)
    depths: tuple[int, int, int, int] = (1, 2, 4, 2)
    d_state: tuple[int, int, int, int] = (32, 32, 64, 64)
    detail_channels: int = 16
    decoder_channels: int = 32
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    embedding_dim: int = 8
    drop_path_rate: float = 0.10
    use_frequency_scan_from_stage: int = 1
    use_local_attention: bool = True
    attention_stages: tuple[int, ...] = (3,)
    attention_window: tuple[int, int] = (8, 16)
    attention_heads: int = 8
    backend: Literal["auto", "mamba2", "torch_surrogate"] = "auto"
    axis_chunk_size: int = 0
    checkpoint_blocks: bool = True
    directional_fusion: Literal["stack", "sequential"] = "stack"
    checkpoint_directional_branches: bool = False
    checkpoint_decoder: bool = False
    checkpoint_heads: bool = False


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device, dtype=x.dtype) < keep).to(x.dtype)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (y.shape[-1],), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2).contiguous()


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        groups: int = 1,
        activate: bool = True,
    ):
        padding = (kernel[0] // 2, kernel[1] // 2)
        modules: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
        ]
        if activate:
            modules.append(nn.GELU())
        super().__init__(*modules)


class ConvNeXtBlock2d(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.contract = nn.Conv2d(4 * channels, channels, 1)
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.contract(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        y = y * self.scale.view(1, -1, 1, 1)
        return x + self.drop_path(y)


class TorchSequenceSurrogate(nn.Module):
    """Fast CPU-only sequence surrogate for unit tests.

    It has the same ``[B,L,D] -> [B,L,D]`` contract as Mamba-2 but is not a
    selective state-space model.  Manuscript experiments must set
    ``backend=mamba2`` and record the backend in every result artifact.
    """

    def __init__(self, d_model: int, d_conv: int = 4):
        super().__init__()
        self.input_projection = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, max(1, d_conv), padding=max(1, d_conv) - 1, groups=d_model
        )
        self.context_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input_projection(x).chunk(2, dim=-1)
        value = self.depthwise(value.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        value = F.silu(value)
        denominator = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[None, :, None]
        prefix = torch.cumsum(value, dim=1) / denominator
        return self.output_projection((value + self.context_projection(prefix)) * torch.sigmoid(gate))


class SequenceMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        backend: str,
    ):
        super().__init__()
        use_mamba = backend == "mamba2" or (
            backend == "auto" and OfficialMamba2 is not None and torch.cuda.is_available()
        )
        if use_mamba:
            if OfficialMamba2 is None:
                raise RuntimeError("mamba-ssm is not installed, but backend='mamba2' was requested")
            if d_model * expand % headdim:
                raise ValueError("d_model * expand must be divisible by headdim")
            kwargs = {
                "d_model": d_model,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
                "headdim": headdim,
            }
            signature = inspect.signature(OfficialMamba2)
            if "use_mem_eff_path" in signature.parameters:
                kwargs["use_mem_eff_path"] = True
            self.module = OfficialMamba2(**kwargs)
            self.backend = "mamba2"
        else:
            self.module = TorchSequenceSurrogate(d_model, d_conv=d_conv)
            self.backend = "torch_surrogate"
            if backend == "auto":
                warnings.warn(
                    "Using torch_surrogate because a CUDA Mamba-2 backend is unavailable. "
                    "Do not report these results as WhistleMamba-Track.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


def _run_chunks(sequence: torch.Tensor, mixer: nn.Module, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or sequence.shape[0] <= chunk_size:
        return mixer(sequence)
    return torch.cat([mixer(part) for part in sequence.split(chunk_size, dim=0)], dim=0)


class RectangularWindowAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        window: tuple[int, int],
        drop_path: float,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.window = window
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Linear(channels, channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        wh, ww = self.window
        pad_h, pad_w = (-height) % wh, (-width) % ww
        padded = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = padded.shape[-2:]
        windows = (
            padded.view(batch, channels, hp // wh, wh, wp // ww, ww)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, wh * ww, channels)
        )
        query = self.norm(windows)
        attended, _ = self.attention(query, query, query, need_weights=False)
        attended = self.output(attended)
        attended = (
            attended.view(batch, hp // wh, wp // ww, wh, ww, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, hp, wp)
        )[:, :, :height, :width]
        return x + self.drop_path(attended)


class AnisotropicBiAxialMambaBlock(nn.Module):
    """Local ridge branch + bidirectional temporal/frequency Mamba-2 scans."""

    def __init__(
        self,
        channels: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        frequency_scan: bool,
        backend: str,
        axis_chunk_size: int,
        drop_path: float,
        directional_fusion: str,
        checkpoint_directional_branches: bool,
    ):
        super().__init__()
        self.frequency_scan = frequency_scan
        self.axis_chunk_size = axis_chunk_size
        self.directional_fusion = directional_fusion
        self.checkpoint_directional_branches = checkpoint_directional_branches
        if directional_fusion not in {"stack", "sequential"}:
            raise ValueError("directional_fusion must be 'stack' or 'sequential'")
        self.norm = LayerNorm2d(channels)
        mixer_kwargs = {
            "d_model": channels,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "backend": backend,
        }
        self.time_forward = SequenceMixer(**mixer_kwargs)
        self.time_backward = SequenceMixer(**mixer_kwargs)
        self.frequency_up = SequenceMixer(**mixer_kwargs) if frequency_scan else None
        self.frequency_down = SequenceMixer(**mixer_kwargs) if frequency_scan else None
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, (3, 9), padding=(1, 4), groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        branch_count = 5 if frequency_scan else 3
        self.gate = nn.Sequential(
            nn.Conv2d(channels, max(8, channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), branch_count, 1),
        )
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.drop_path = DropPath(drop_path)
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, 1),
            nn.GELU(),
            nn.Conv2d(4 * channels, channels, 1),
        )
        self.ffn_scale = nn.Parameter(torch.full((channels,), 1e-6))

    def _time(self, x: torch.Tensor, mixer: nn.Module, reverse: bool) -> torch.Tensor:
        batch, channels, height, width = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        if reverse:
            sequence = sequence.flip(1)
        sequence = _run_chunks(sequence, mixer, self.axis_chunk_size)
        if reverse:
            sequence = sequence.flip(1)
        return sequence.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _frequency(self, x: torch.Tensor, mixer: nn.Module, reverse: bool) -> torch.Tensor:
        batch, channels, height, width = x.shape
        sequence = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        if reverse:
            sequence = sequence.flip(1)
        sequence = _run_chunks(sequence, mixer, self.axis_chunk_size)
        if reverse:
            sequence = sequence.flip(1)
        return sequence.view(batch, width, height, channels).permute(0, 3, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        gates = self.gate(normalized).softmax(dim=1)
        if self.directional_fusion == "stack":
            branches = [
                self.local(normalized),
                self._time(normalized, self.time_forward, False),
                self._time(normalized, self.time_backward, True),
            ]
            if self.frequency_scan:
                assert self.frequency_up is not None and self.frequency_down is not None
                branches.extend(
                    [
                        self._frequency(normalized, self.frequency_up, False),
                        self._frequency(normalized, self.frequency_down, True),
                    ]
                )
            fused = (torch.stack(branches, dim=1) * gates[:, :, None]).sum(dim=1)
        else:
            # Weight and release one branch at a time. Including the gate in the
            # checkpointed function prevents autograd from retaining a full branch
            # solely to compute gate gradients.
            def local_branch(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                return self.local(value) * gate

            def time_forward(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                return self._time(value, self.time_forward, False) * gate

            def time_backward(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                return self._time(value, self.time_backward, True) * gate

            use_checkpoint = self.training and self.checkpoint_directional_branches

            def run(function, gate_index: int) -> torch.Tensor:
                arguments = (normalized, gates[:, gate_index : gate_index + 1])
                return (
                    checkpoint(function, *arguments, use_reentrant=False)
                    if use_checkpoint
                    else function(*arguments)
                )

            fused = run(local_branch, 0)
            fused = fused + run(time_forward, 1)
            fused = fused + run(time_backward, 2)
            if self.frequency_scan:
                assert self.frequency_up is not None and self.frequency_down is not None

                def frequency_up(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                    return self._frequency(value, self.frequency_up, False) * gate

                def frequency_down(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
                    return self._frequency(value, self.frequency_down, True) * gate

                fused = fused + run(frequency_up, 3)
                fused = fused + run(frequency_down, 4)
        x = x + self.drop_path(self.output_projection(fused))
        update = self.ffn(self.ffn_norm(x)) * self.ffn_scale.view(1, -1, 1, 1)
        return x + self.drop_path(update)


class EncoderStage(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        stage_index: int,
        cfg: WhistleMambaConfig,
        drop_paths: Sequence[float],
    ):
        super().__init__()
        modules: list[nn.Module] = []
        for index in range(depth):
            modules.append(
                AnisotropicBiAxialMambaBlock(
                    channels,
                    d_state=cfg.d_state[stage_index],
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    headdim=cfg.headdim,
                    frequency_scan=stage_index >= cfg.use_frequency_scan_from_stage,
                    backend=cfg.backend,
                    axis_chunk_size=cfg.axis_chunk_size,
                    drop_path=drop_paths[index],
                    directional_fusion=cfg.directional_fusion,
                    checkpoint_directional_branches=cfg.checkpoint_directional_branches,
                )
            )
            if cfg.use_local_attention and stage_index in cfg.attention_stages:
                modules.append(
                    RectangularWindowAttention(
                        channels,
                        heads=cfg.attention_heads,
                        window=cfg.attention_window,
                        drop_path=drop_paths[index],
                    )
                )
        self.modules_list = nn.ModuleList(modules)
        self.use_checkpoint = cfg.checkpoint_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules_list:
            x = checkpoint(module, x, use_reentrant=False) if self.training and self.use_checkpoint else module(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels, (3, 7)),
            ConvNeXtBlock2d(out_channels),
            ConvNeXtBlock2d(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class PredictionHead(nn.Sequential):
    def __init__(self, channels: int, outputs: int):
        super().__init__(
            ConvNormAct(channels, channels, (3, 7)),
            nn.Conv2d(channels, outputs, 1),
        )


class EdgeScorer(nn.Module):
    FEATURE_DIMENSION = 12

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.FEATURE_DIMENSION, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.FEATURE_DIMENSION:
            raise ValueError(
                f"edge features must have {self.FEATURE_DIMENSION} values, got {features.shape[-1]}"
            )
        return self.network(features).squeeze(-1)


class EncoderPresenceHead(nn.Module):
    """Detect sparse window-level evidence from pooled deepest-encoder features."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.network = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        mean = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        rms = F.adaptive_avg_pool2d(encoded.square(), 1).flatten(1).add(1e-6).sqrt()
        return self.network(torch.cat([mean, rms], dim=1))


class WhistleMambaTrack(nn.Module):
    """Native-aspect, multi-task whistle contour detector and learned tracker."""

    def __init__(self, cfg: Optional[WhistleMambaConfig] = None):
        super().__init__()
        self.cfg = cfg or WhistleMambaConfig()
        if len(self.cfg.channels) != 4 or len(self.cfg.depths) != 4 or len(self.cfg.d_state) != 4:
            raise ValueError("channels, depths, and d_state must each contain four stages")
        c0, c1, c2, c3 = self.cfg.channels
        self.detail_stem = nn.Sequential(
            ConvNormAct(self.cfg.in_channels, self.cfg.detail_channels, (3, 9)),
            ConvNeXtBlock2d(self.cfg.detail_channels),
        )
        self.patch_stem = nn.Sequential(
            ConvNormAct(self.cfg.detail_channels, c0, (5, 9), stride=(2, 4)),
            ConvNeXtBlock2d(c0),
        )
        drop_paths = torch.linspace(0, self.cfg.drop_path_rate, sum(self.cfg.depths)).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_index, (channels, depth) in enumerate(zip(self.cfg.channels, self.cfg.depths)):
            self.stages.append(
                EncoderStage(
                    channels,
                    depth,
                    stage_index,
                    self.cfg,
                    drop_paths[cursor : cursor + depth],
                )
            )
            cursor += depth
            if stage_index < 3:
                self.downsamples.append(Downsample(channels, self.cfg.channels[stage_index + 1]))

        self.decode2 = DecoderBlock(c3, c2, c2)
        self.decode1 = DecoderBlock(c2, c1, c1)
        self.decode0 = DecoderBlock(c1, c0, c0)
        self.decode_detail = DecoderBlock(c0, self.cfg.detail_channels, self.cfg.decoder_channels)
        self.refine = nn.Sequential(
            ConvNeXtBlock2d(self.cfg.decoder_channels),
            ConvNeXtBlock2d(self.cfg.decoder_channels),
        )
        d = self.cfg.decoder_channels
        self.centerline_head = PredictionHead(d, 1)
        self.tangent_head = PredictionHead(d, 2)
        self.endpoint_head = PredictionHead(d, 2)  # start, end
        self.junction_head = PredictionHead(d, 1)
        self.embedding_head = PredictionHead(d, self.cfg.embedding_dim)
        self.uncertainty_head = PredictionHead(d, 1)
        self.presence_head = EncoderPresenceHead(c3)
        self.edge_scorer = EdgeScorer()
        self._initialize_task_heads()

    def _initialize_task_heads(self) -> None:
        # Never recursively reinitialize Mamba's A/delta parameters.
        task_modules = [
            self.centerline_head,
            self.tangent_head,
            self.endpoint_head,
            self.junction_head,
            self.embedding_head,
            self.uncertainty_head,
            self.presence_head,
            self.edge_scorer,
        ]
        for task_module in task_modules:
            for module in task_module.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        # The positive class is sparse. A modest negative centerline bias avoids
        # saturated false positives at initialization without making recall impossible.
        final = self.centerline_head[-1]
        if isinstance(final, nn.Conv2d) and final.bias is not None:
            nn.init.constant_(final.bias, -2.0)

    @property
    def effective_backend(self) -> str:
        for module in self.modules():
            if isinstance(module, SequenceMixer):
                return module.backend
        return "unknown"

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        detail = self.detail_stem(x)
        encoded = self.patch_stem(detail)
        skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            encoded = stage(encoded)
            skips.append(encoded)
            if index < len(self.downsamples):
                encoded = self.downsamples[index](encoded)
        return detail, skips

    def decode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        input_size = x.shape[-2:]
        detail, skips = self.forward_features(x)
        use_checkpoint = self.training and self.cfg.checkpoint_decoder

        def run(module: nn.Module, *arguments: torch.Tensor) -> torch.Tensor:
            return (
                checkpoint(module, *arguments, use_reentrant=False)
                if use_checkpoint
                else module(*arguments)
            )

        decoded = run(self.decode2, skips[3], skips[2])
        decoded = run(self.decode1, decoded, skips[1])
        decoded = run(self.decode0, decoded, skips[0])
        decoded = run(self.decode_detail, decoded, detail)
        decoded = run(self.refine, decoded)
        if decoded.shape[-2:] != input_size:
            decoded = F.interpolate(decoded, size=input_size, mode="bilinear", align_corners=False)
        return decoded, skips

    def forward(
        self,
        x: torch.Tensor,
        enabled_heads: Optional[set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        decoded, skips = self.decode_features(x)
        selected = enabled_heads if enabled_heads is not None else {
            "centerline",
            "tangent",
            "endpoint",
            "junction",
            "embedding",
            "uncertainty",
            "presence",
        }
        unknown = selected - {
            "centerline",
            "tangent",
            "endpoint",
            "junction",
            "embedding",
            "uncertainty",
            "presence",
        }
        if unknown:
            raise ValueError(f"unknown output heads: {sorted(unknown)}")
        use_checkpoint = self.training and self.cfg.checkpoint_heads

        def run(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
            return (
                checkpoint(module, value, use_reentrant=False)
                if use_checkpoint
                else module(value)
            )

        outputs: Dict[str, torch.Tensor] = {}
        if "centerline" in selected:
            outputs["centerline_logits"] = run(self.centerline_head, decoded)
        if "tangent" in selected:
            outputs["tangent"] = F.normalize(
                run(self.tangent_head, decoded), p=2, dim=1, eps=1e-6
            )
        if "endpoint" in selected:
            outputs["endpoint_logits"] = run(self.endpoint_head, decoded)
        if "junction" in selected:
            outputs["junction_logits"] = run(self.junction_head, decoded)
        if "embedding" in selected:
            outputs["embeddings"] = run(self.embedding_head, decoded)
        if "uncertainty" in selected:
            outputs["log_variance"] = run(self.uncertainty_head, decoded).clamp(-6.0, 6.0)
        if "presence" in selected:
            outputs["presence_logit"] = run(self.presence_head, skips[-1]).squeeze(-1)
        return outputs

    def score_edges(self, features: torch.Tensor) -> torch.Tensor:
        return self.edge_scorer(features)

    def parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        tracker_ids = {id(parameter) for parameter in self.edge_scorer.parameters()}
        head_modules = [
            self.centerline_head,
            self.tangent_head,
            self.endpoint_head,
            self.junction_head,
            self.embedding_head,
            self.uncertainty_head,
            self.presence_head,
        ]
        head_ids = {id(parameter) for module in head_modules for parameter in module.parameters()}
        return {
            "tracker": [parameter for parameter in self.parameters() if id(parameter) in tracker_ids],
            "heads": [parameter for parameter in self.parameters() if id(parameter) in head_ids],
            "backbone": [
                parameter
                for parameter in self.parameters()
                if id(parameter) not in tracker_ids and id(parameter) not in head_ids
            ],
        }
