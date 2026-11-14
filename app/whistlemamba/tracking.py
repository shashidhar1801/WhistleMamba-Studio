from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .losses import build_edge_features


@dataclass(frozen=True)
class TrackerConfig:
    node_threshold: float = 0.25
    start_threshold: float = 0.35
    edge_threshold: float = 0.50
    endpoint_stop_threshold: float = 0.65
    frequency_nms_bins: int = 3
    max_nodes_per_frame: int = 24
    max_gap_frames: int = 8
    max_frequency_jump_bins: int = 24
    min_duration_frames: int = 75
    unmatched_cost: float = 6.0
    detection_cost_weight: float = 0.25


@dataclass
class TrackNode:
    time_index: int
    frequency_index: int
    confidence: float
    start_confidence: float
    end_confidence: float
    tangent: torch.Tensor
    embedding: torch.Tensor


@dataclass
class ActiveTrack:
    nodes: list[TrackNode] = field(default_factory=list)
    gaps: int = 0
    edge_probabilities: list[float] = field(default_factory=list)

    @property
    def last(self) -> TrackNode:
        return self.nodes[-1]

    @property
    def duration(self) -> int:
        return self.nodes[-1].time_index - self.nodes[0].time_index + 1


@dataclass
class WhistleTrack:
    time_indices: np.ndarray
    frequency_indices: np.ndarray
    node_confidences: np.ndarray
    mean_confidence: float
    mean_edge_probability: float

    def physical(
        self,
        time_axis_s: Sequence[float],
        frequency_axis_hz: Sequence[float],
        time_offset_s: float = 0.0,
    ) -> Dict[str, np.ndarray | float]:
        time_axis = np.asarray(time_axis_s)
        frequency_axis = np.asarray(frequency_axis_hz)
        return {
            "time_s": time_axis[self.time_indices] + time_offset_s,
            "frequency_hz": frequency_axis[self.frequency_indices],
            "node_confidence": self.node_confidences,
            "mean_confidence": self.mean_confidence,
            "mean_edge_probability": self.mean_edge_probability,
        }


def _candidates(
    probability: torch.Tensor,
    tangent: torch.Tensor,
    embedding: torch.Tensor,
    endpoints: torch.Tensor,
    cfg: TrackerConfig,
) -> list[list[TrackNode]]:
    if cfg.frequency_nms_bins % 2 == 0:
        raise ValueError("frequency_nms_bins must be odd")
    maxima = F.max_pool2d(
        probability[None, None],
        (cfg.frequency_nms_bins, 1),
        stride=1,
        padding=(cfg.frequency_nms_bins // 2, 0),
    )[0, 0]
    keep = (probability >= cfg.node_threshold) & (probability >= maxima - 1e-8)
    result: list[list[TrackNode]] = []
    for time_index in range(probability.shape[1]):
        frequencies = torch.nonzero(keep[:, time_index], as_tuple=False).flatten()
        if frequencies.numel() > cfg.max_nodes_per_frame:
            order = torch.argsort(
                probability[frequencies, time_index], descending=True, stable=True
            )[: cfg.max_nodes_per_frame]
            frequencies = frequencies[order]
        result.append(
            [
                TrackNode(
                    time_index=time_index,
                    frequency_index=int(frequency.item()),
                    confidence=float(probability[frequency, time_index].item()),
                    start_confidence=float(endpoints[0, frequency, time_index].item()),
                    end_confidence=float(endpoints[1, frequency, time_index].item()),
                    tangent=tangent[:, frequency, time_index].detach(),
                    embedding=embedding[:, frequency, time_index].detach(),
                )
                for frequency in frequencies
            ]
        )
    return result


class LearnedContourTracker:
    """Online multi-whistle assignment using the learned edge-scoring MLP."""

    def __init__(self, edge_scorer: torch.nn.Module, cfg: TrackerConfig = TrackerConfig()):
        self.edge_scorer = edge_scorer
        self.cfg = cfg
        self.last_diagnostics: Dict[str, float | int] = {}

    @torch.no_grad()
    def extract(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch_index: int = 0,
    ) -> list[WhistleTrack]:
        started = perf_counter()
        probability = torch.sigmoid(outputs["centerline_logits"])[batch_index, 0]
        tangent = outputs["tangent"][batch_index]
        embedding = outputs["embeddings"][batch_index]
        endpoints = torch.sigmoid(outputs["endpoint_logits"])[batch_index]
        junctions = torch.sigmoid(outputs["junction_logits"])[batch_index, 0]
        frames = _candidates(probability, tangent, embedding, endpoints, self.cfg)
        active: list[ActiveTrack] = []
        finished: list[ActiveTrack] = []
        device = probability.device
        candidate_nodes = sum(len(nodes) for nodes in frames)
        candidate_pairs_considered = 0
        candidate_edges = 0
        assigned_edges = 0
        gap_links = 0
        births = 0
        endpoint_terminations = 0
        maximum_active_tracks = 0

        for frame_nodes in frames:
            for track in active:
                track.gaps += 1
            assigned_nodes: set[int] = set()
            if active and frame_nodes:
                cost = torch.full(
                    (len(active), len(frame_nodes)),
                    self.cfg.unmatched_cost,
                    dtype=torch.float32,
                    device=device,
                )
                edge_probability = torch.zeros_like(cost)
                edge_features: list[torch.Tensor] = []
                edge_pairs: list[tuple[int, int]] = []
                for track_index, track in enumerate(active):
                    source = track.last
                    for node_index, destination in enumerate(frame_nodes):
                        candidate_pairs_considered += 1
                        delta_time = destination.time_index - source.time_index
                        delta_frequency = destination.frequency_index - source.frequency_index
                        if not 0 < delta_time <= self.cfg.max_gap_frames:
                            continue
                        if abs(delta_frequency) > self.cfg.max_frequency_jump_bins:
                            continue
                        candidate_edges += 1
                        midpoint_frequency = int(
                            round((source.frequency_index + destination.frequency_index) / 2)
                        )
                        midpoint_time = int(
                            round((source.time_index + destination.time_index) / 2)
                        )
                        midpoint_frequency = min(
                            max(midpoint_frequency, 0), junctions.shape[0] - 1
                        )
                        midpoint_time = min(max(midpoint_time, 0), junctions.shape[1] - 1)
                        edge_features.append(
                            build_edge_features(
                                probability.new_tensor([source.confidence]),
                                probability.new_tensor([destination.confidence]),
                                source.embedding[None].to(device),
                                destination.embedding[None].to(device),
                                source.tangent[None].to(device),
                                destination.tangent[None].to(device),
                                probability.new_tensor([source.end_confidence]),
                                probability.new_tensor([destination.start_confidence]),
                                junctions[midpoint_frequency, midpoint_time][None],
                                probability.new_tensor([float(delta_time)]),
                                probability.new_tensor([float(delta_frequency)]),
                                max_gap=float(self.cfg.max_gap_frames),
                                max_jump=float(self.cfg.max_frequency_jump_bins),
                            ).squeeze(0)
                        )
                        edge_pairs.append((track_index, node_index))
                if edge_features:
                    scores = torch.sigmoid(self.edge_scorer(torch.stack(edge_features)))
                    for score, (track_index, node_index) in zip(scores, edge_pairs):
                        edge_probability[track_index, node_index] = score
                        detection = max(frame_nodes[node_index].confidence, 1e-6)
                        cost[track_index, node_index] = -torch.log(score.clamp_min(1e-6)) - (
                            self.cfg.detection_cost_weight * np.log(detection)
                        )
                rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
                for track_index, node_index in zip(rows.tolist(), columns.tolist()):
                    score = float(edge_probability[track_index, node_index].item())
                    if score < self.cfg.edge_threshold:
                        continue
                    active[track_index].nodes.append(frame_nodes[node_index])
                    active[track_index].edge_probabilities.append(score)
                    active[track_index].gaps = 0
                    assigned_nodes.add(node_index)
                    assigned_edges += 1
                    if (
                        frame_nodes[node_index].time_index
                        - active[track_index].nodes[-2].time_index
                        > 1
                    ):
                        gap_links += 1

            for node_index, node in enumerate(frame_nodes):
                start_score = 0.75 * node.confidence + 0.25 * node.start_confidence
                if node_index not in assigned_nodes and start_score >= self.cfg.start_threshold:
                    active.append(ActiveTrack(nodes=[node]))
                    births += 1

            retained: list[ActiveTrack] = []
            for track in active:
                should_finish = track.gaps > self.cfg.max_gap_frames or (
                    track.gaps > 0
                    and track.last.end_confidence >= self.cfg.endpoint_stop_threshold
                )
                if should_finish:
                    finished.append(track)
                    if (
                        track.gaps > 0
                        and track.last.end_confidence >= self.cfg.endpoint_stop_threshold
                    ):
                        endpoint_terminations += 1
                else:
                    retained.append(track)
            active = retained
            maximum_active_tracks = max(maximum_active_tracks, len(active))

        finished.extend(active)
        tracks: list[WhistleTrack] = []
        for track in finished:
            if len(track.nodes) < 2 or track.duration < self.cfg.min_duration_frames:
                continue
            confidence = np.asarray([node.confidence for node in track.nodes], dtype=np.float32)
            tracks.append(
                WhistleTrack(
                    time_indices=np.asarray(
                        [node.time_index for node in track.nodes], dtype=np.int64
                    ),
                    frequency_indices=np.asarray(
                        [node.frequency_index for node in track.nodes], dtype=np.int64
                    ),
                    node_confidences=confidence,
                    mean_confidence=float(confidence.mean()),
                    mean_edge_probability=float(np.mean(track.edge_probabilities))
                    if track.edge_probabilities
                    else 0.0,
                )
            )
        elapsed = perf_counter() - started
        self.last_diagnostics = {
            "frames": len(frames),
            "candidate_nodes": candidate_nodes,
            "candidate_pairs_considered": candidate_pairs_considered,
            "candidate_edges": candidate_edges,
            "assigned_edges": assigned_edges,
            "gap_links": gap_links,
            "births": births,
            "deaths": len(finished),
            "endpoint_terminations": endpoint_terminations,
            "maximum_active_tracks": maximum_active_tracks,
            "output_tracks": len(tracks),
            "edge_feature_width": 12,
            "estimated_edge_feature_bytes": candidate_edges * 12 * 4,
            "runtime_seconds": elapsed,
        }
        return tracks


def _overlap_match_cost(
    previous: Mapping[str, np.ndarray | float],
    current: Mapping[str, np.ndarray | float],
    *,
    minimum_overlap_s: float,
    minimum_overlap_fraction: float,
    overlap_frequency_tolerance_hz: float,
) -> float | None:
    previous_time = np.asarray(previous["time_s"], dtype=np.float64)
    previous_frequency = np.asarray(previous["frequency_hz"], dtype=np.float64)
    current_time = np.asarray(current["time_s"], dtype=np.float64)
    current_frequency = np.asarray(current["frequency_hz"], dtype=np.float64)
    overlap_start = max(float(previous_time.min()), float(current_time.min()))
    overlap_end = min(float(previous_time.max()), float(current_time.max()))
    overlap = overlap_end - overlap_start
    if overlap < minimum_overlap_s:
        return None
    previous_duration = max(float(previous_time.max() - previous_time.min()), 1e-9)
    current_duration = max(float(current_time.max() - current_time.min()), 1e-9)
    fraction = overlap / min(previous_duration, current_duration)
    if fraction < minimum_overlap_fraction:
        return None
    samples = np.unique(
        np.concatenate(
            [
                previous_time[(previous_time >= overlap_start) & (previous_time <= overlap_end)],
                current_time[(current_time >= overlap_start) & (current_time <= overlap_end)],
            ]
        )
    )
    if samples.size < 2:
        return None
    previous_interp = np.interp(samples, previous_time, previous_frequency)
    current_interp = np.interp(samples, current_time, current_frequency)
    median_error = float(np.median(np.abs(previous_interp - current_interp)))
    if median_error > overlap_frequency_tolerance_hz:
        return None
    return median_error / max(overlap_frequency_tolerance_hz, 1e-9) + (1.0 - fraction)


def _coalesce_track(track: Mapping[str, np.ndarray | float]) -> Dict[str, np.ndarray | float]:
    """Sort a physical track and merge duplicate timestamps by confidence weighting."""
    result: Dict[str, np.ndarray | float] = dict(track)
    time = np.asarray(track["time_s"], dtype=np.float64)
    frequency = np.asarray(track["frequency_hz"], dtype=np.float64)
    confidence = np.asarray(
        track.get("node_confidence", np.ones_like(time)), dtype=np.float64
    )
    if not (len(time) == len(frequency) == len(confidence)):
        raise ValueError("time, frequency, and node_confidence must have equal lengths")
    order = np.argsort(time, kind="stable")
    time, frequency, confidence = time[order], frequency[order], confidence[order]
    unique, inverse = np.unique(time, return_inverse=True)
    weight_sum = np.zeros_like(unique, dtype=np.float64)
    frequency_sum = np.zeros_like(unique, dtype=np.float64)
    confidence_max = np.zeros_like(unique, dtype=np.float64)
    weights = np.maximum(confidence, 1e-6)
    np.add.at(weight_sum, inverse, weights)
    np.add.at(frequency_sum, inverse, frequency * weights)
    np.maximum.at(confidence_max, inverse, confidence)
    result["time_s"] = unique
    result["frequency_hz"] = frequency_sum / np.maximum(weight_sum, 1e-12)
    result["node_confidence"] = confidence_max
    result["mean_confidence"] = float(confidence_max.mean()) if unique.size else 0.0
    return result


def stitch_overlapping_tracks(
    tracks: Sequence[Dict[str, np.ndarray | float]],
    *,
    max_time_gap_s: float = 0.020,
    max_frequency_gap_hz: float = 1_000.0,
    minimum_overlap_s: float = 0.050,
    minimum_overlap_fraction: float = 0.20,
    overlap_frequency_tolerance_hz: float = 750.0,
) -> list[Dict[str, np.ndarray | float]]:
    """Merge duplicate/adjacent contours produced by overlapping windows.

    Two tracks are eligible when they either (a) overlap in time and have a
    small median frequency discrepancy over the shared interval, or (b) are
    adjacent within the endpoint time/frequency gates. Thresholds must be tuned
    on validation only and frozen for final evaluation.
    """
    pending = sorted(tracks, key=lambda track: float(np.asarray(track["time_s"])[0]))
    merged: list[Dict[str, np.ndarray | float]] = []
    for track in pending:
        time = np.asarray(track["time_s"], dtype=np.float64)
        frequency = np.asarray(track["frequency_hz"], dtype=np.float64)
        best_index, best_cost = None, float("inf")
        for index, previous in enumerate(merged):
            overlap_cost = _overlap_match_cost(
                previous,
                track,
                minimum_overlap_s=minimum_overlap_s,
                minimum_overlap_fraction=minimum_overlap_fraction,
                overlap_frequency_tolerance_hz=overlap_frequency_tolerance_hz,
            )
            if overlap_cost is not None and overlap_cost < best_cost:
                best_index, best_cost = index, overlap_cost
                continue
            previous_time = np.asarray(previous["time_s"], dtype=np.float64)
            previous_frequency = np.asarray(previous["frequency_hz"], dtype=np.float64)
            delta_time = float(time[0] - previous_time[-1])
            delta_frequency = float(abs(frequency[0] - previous_frequency[-1]))
            if 0.0 <= delta_time <= max_time_gap_s and delta_frequency <= max_frequency_gap_hz:
                cost = 2.0 + delta_time / max(max_time_gap_s, 1e-9) + (
                    delta_frequency / max(max_frequency_gap_hz, 1e-9)
                )
                if cost < best_cost:
                    best_index, best_cost = index, cost
        if best_index is None:
            merged.append(_coalesce_track(track))
            continue
        previous = merged[best_index]
        combined: Dict[str, np.ndarray | float] = dict(previous)
        combined["time_s"] = np.concatenate([np.asarray(previous["time_s"]), time])
        combined["frequency_hz"] = np.concatenate(
            [np.asarray(previous["frequency_hz"]), frequency]
        )
        previous_confidence = np.asarray(
            previous.get("node_confidence", np.ones_like(np.asarray(previous["time_s"])))
        )
        current_confidence = np.asarray(
            track.get("node_confidence", np.ones_like(time))
        )
        combined["node_confidence"] = np.concatenate(
            [previous_confidence, current_confidence]
        )
        merged[best_index] = _coalesce_track(combined)
    return merged


class HeuristicContourTracker:
    """Shared geometry-only postprocessor for dense segmentation baselines.

    It deliberately contains no learned edge parameters. Use it for harmonized
    baseline comparisons, while also reporting each published method's native
    postprocessor as a separate result.
    """

    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg

    @torch.no_grad()
    def extract(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch_index: int = 0,
    ) -> list[WhistleTrack]:
        probability = torch.sigmoid(outputs["centerline_logits"])[batch_index, 0]
        # Dummy geometry is only used by candidate construction.
        tangent = torch.zeros((2, *probability.shape), device=probability.device)
        tangent[1] = 1.0
        embedding = torch.zeros((1, *probability.shape), device=probability.device)
        endpoints = torch.zeros((2, *probability.shape), device=probability.device)
        frames = _candidates(probability, tangent, embedding, endpoints, self.cfg)
        active: list[ActiveTrack] = []
        finished: list[ActiveTrack] = []
        for frame_nodes in frames:
            for track in active:
                track.gaps += 1
            assigned: set[int] = set()
            if active and frame_nodes:
                cost = np.full(
                    (len(active), len(frame_nodes)), self.cfg.unmatched_cost, dtype=np.float64
                )
                for track_index, track in enumerate(active):
                    source = track.last
                    if len(track.nodes) >= 2:
                        previous = track.nodes[-2]
                        slope = (source.frequency_index - previous.frequency_index) / max(
                            source.time_index - previous.time_index, 1
                        )
                    else:
                        slope = 0.0
                    for node_index, destination in enumerate(frame_nodes):
                        dt = destination.time_index - source.time_index
                        if not 0 < dt <= self.cfg.max_gap_frames:
                            continue
                        predicted_frequency = source.frequency_index + slope * dt
                        residual = abs(destination.frequency_index - predicted_frequency)
                        if residual > self.cfg.max_frequency_jump_bins:
                            continue
                        cost[track_index, node_index] = (
                            residual / max(self.cfg.max_frequency_jump_bins, 1)
                            - self.cfg.detection_cost_weight
                            * np.log(max(destination.confidence, 1e-6))
                            + 0.05 * (dt - 1)
                        )
                rows, columns = linear_sum_assignment(cost)
                for track_index, node_index in zip(rows.tolist(), columns.tolist()):
                    if cost[track_index, node_index] >= self.cfg.unmatched_cost:
                        continue
                    active[track_index].nodes.append(frame_nodes[node_index])
                    active[track_index].edge_probabilities.append(
                        float(np.exp(-max(cost[track_index, node_index], 0.0)))
                    )
                    active[track_index].gaps = 0
                    assigned.add(node_index)
            for node_index, node in enumerate(frame_nodes):
                if node_index not in assigned and node.confidence >= self.cfg.start_threshold:
                    active.append(ActiveTrack(nodes=[node]))
            retained = []
            for track in active:
                if track.gaps > self.cfg.max_gap_frames:
                    finished.append(track)
                else:
                    retained.append(track)
            active = retained
        finished.extend(active)
        result: list[WhistleTrack] = []
        for track in finished:
            if len(track.nodes) < 2 or track.duration < self.cfg.min_duration_frames:
                continue
            confidence = np.asarray([node.confidence for node in track.nodes], dtype=np.float32)
            result.append(
                WhistleTrack(
                    time_indices=np.asarray([node.time_index for node in track.nodes]),
                    frequency_indices=np.asarray(
                        [node.frequency_index for node in track.nodes]
                    ),
                    node_confidences=confidence,
                    mean_confidence=float(confidence.mean()),
                    mean_edge_probability=float(np.mean(track.edge_probabilities))
                    if track.edge_probabilities
                    else 0.0,
                )
            )
        return result
