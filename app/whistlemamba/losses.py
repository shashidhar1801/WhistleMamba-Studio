import torch
import torch.nn.functional as F

def build_edge_features(
    p_i: torch.Tensor,
    p_j: torch.Tensor,
    embedding_i: torch.Tensor,
    embedding_j: torch.Tensor,
    tangent_i: torch.Tensor,
    tangent_j: torch.Tensor,
    end_i: torch.Tensor,
    start_j: torch.Tensor,
    junction_midpoint: torch.Tensor,
    delta_time: torch.Tensor,
    delta_frequency: torch.Tensor,
    *,
    max_gap: float,
    max_jump: float,
) -> torch.Tensor:
    normalized_time = delta_time / max(max_gap, 1.0)
    normalized_frequency = delta_frequency / max(max_jump, 1.0)
    displacement = F.normalize(
        torch.stack([delta_frequency, delta_time], dim=-1), p=2, dim=-1, eps=1e-6
    )
    # Tangents use the orientation-safe double-angle representation. Convert
    # the candidate displacement to the same representation before comparing.
    displacement_orientation = torch.stack(
        [
            displacement[..., 1].square() - displacement[..., 0].square(),
            2.0 * displacement[..., 0] * displacement[..., 1],
        ],
        dim=-1,
    )
    alignment_i = (tangent_i * displacement_orientation).sum(dim=-1)
    alignment_j = (tangent_j * displacement_orientation).sum(dim=-1)
    embedding_cosine = (
        F.normalize(embedding_i, p=2, dim=-1, eps=1e-6)
        * F.normalize(embedding_j, p=2, dim=-1, eps=1e-6)
    ).sum(dim=-1)
    normalized_slope = (delta_frequency / delta_time.clamp_min(1.0)) / max(
        max_jump / max(max_gap, 1.0), 1.0
    )
    return torch.stack(
        [
            p_i,
            p_j,
            normalized_time,
            normalized_frequency,
            normalized_frequency.abs(),
            normalized_slope,
            alignment_i,
            alignment_j,
            embedding_cosine,
            junction_midpoint,
            end_i,
            start_j,
        ],
        dim=-1,
    )
