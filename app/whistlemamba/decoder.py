from __future__ import annotations
from dataclasses import asdict
from time import perf_counter
from typing import Literal,Mapping,Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .losses import build_edge_features
from .tracking import TrackerConfig
MINIMUM_CACHED_NODE_THRESHOLD=0.15

@torch.no_grad()
def inference_graph(
    outputs: Mapping[str, torch.Tensor],
    edge_scorer: torch.nn.Module,
    cfg: TrackerConfig,
    *,
    batch_index: int = 0,
    node_threshold: float = MINIMUM_CACHED_NODE_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Build a threshold-reusable sparse node/edge cache for one window."""
    if node_threshold > cfg.node_threshold:
        raise ValueError("cache node threshold must not exceed the configured tracker threshold")
    probability = torch.sigmoid(outputs["centerline_logits"])[batch_index, 0].float()
    tangent = outputs["tangent"][batch_index].float()
    embedding = outputs["embeddings"][batch_index].float()
    endpoints = torch.sigmoid(outputs["endpoint_logits"])[batch_index].float()
    junctions = torch.sigmoid(outputs["junction_logits"])[batch_index, 0].float()

    maxima = F.max_pool2d(
        probability[None, None],
        (cfg.frequency_nms_bins, 1),
        stride=1,
        padding=(cfg.frequency_nms_bins // 2, 0),
    )[0, 0]
    keep = (probability >= node_threshold) & (probability >= maxima - 1e-8)
    node_time: list[torch.Tensor] = []
    node_frequency: list[torch.Tensor] = []
    frame_offsets = [0]
    for time_index in range(probability.shape[1]):
        frequencies = torch.nonzero(keep[:, time_index], as_tuple=False).flatten()
        if frequencies.numel() > cfg.max_nodes_per_frame:
            order = torch.argsort(
                probability[frequencies, time_index], descending=True, stable=True
            )[: cfg.max_nodes_per_frame]
            frequencies = frequencies[order]
        node_frequency.append(frequencies)
        node_time.append(
            torch.full_like(frequencies, time_index, dtype=torch.int64)
        )
        frame_offsets.append(frame_offsets[-1] + frequencies.numel())
    time = torch.cat(node_time) if node_time else torch.empty(0, dtype=torch.int64, device=probability.device)
    frequency = (
        torch.cat(node_frequency)
        if node_frequency
        else torch.empty(0, dtype=torch.int64, device=probability.device)
    )
    score = probability[frequency, time]
    node_tangent = tangent[:, frequency, time].transpose(0, 1)
    node_endpoint = endpoints[:, frequency, time].transpose(0, 1)
    node_junction = junctions[frequency, time]
    node_embedding = embedding[:, frequency, time].transpose(0, 1)

    index_by_frame = torch.full(
        (probability.shape[1], cfg.max_nodes_per_frame),
        -1,
        dtype=torch.int64,
        device=probability.device,
    )
    for time_index, (start, stop) in enumerate(zip(frame_offsets[:-1], frame_offsets[1:])):
        if stop > start:
            index_by_frame[time_index, : stop - start] = torch.arange(
                start, stop, device=probability.device
            )
    edge_source: list[torch.Tensor] = []
    edge_destination: list[torch.Tensor] = []
    if frequency.numel():
        for delta_time in range(1, cfg.max_gap_frames + 1):
            source = index_by_frame[:-delta_time, :, None]
            destination = index_by_frame[delta_time:, None, :]
            valid = (source >= 0) & (destination >= 0)
            source_safe = source.clamp_min(0)
            destination_safe = destination.clamp_min(0)
            valid &= (
                frequency[source_safe] - frequency[destination_safe]
            ).abs() <= cfg.max_frequency_jump_bins
            edge_source.append(source.expand_as(valid)[valid])
            edge_destination.append(destination.expand_as(valid)[valid])
    source = torch.cat(edge_source) if edge_source else torch.empty(0, dtype=torch.int64, device=probability.device)
    destination = (
        torch.cat(edge_destination)
        if edge_destination
        else torch.empty(0, dtype=torch.int64, device=probability.device)
    )
    if source.numel():
        delta_time = (time[destination] - time[source]).float()
        delta_frequency = (frequency[destination] - frequency[source]).float()
        midpoint_frequency = torch.round(
            (frequency[source] + frequency[destination]).float() / 2
        ).long().clamp(0, junctions.shape[0] - 1)
        midpoint_time = torch.round(
            (time[source] + time[destination]).float() / 2
        ).long().clamp(0, junctions.shape[1] - 1)
        features = build_edge_features(
            score[source],
            score[destination],
            node_embedding[source],
            node_embedding[destination],
            node_tangent[source],
            node_tangent[destination],
            node_endpoint[source, 1],
            node_endpoint[destination, 0],
            junctions[midpoint_frequency, midpoint_time],
            delta_time,
            delta_frequency,
            max_gap=float(cfg.max_gap_frames),
            max_jump=float(cfg.max_frequency_jump_bins),
        )
        logits = []
        for start in range(0, len(features), 65_536):
            logits.append(edge_scorer(features[start : start + 65_536]).float())
        edge_logit = torch.cat(logits)
    else:
        features = probability.new_empty((0, 12))
        edge_logit = probability.new_empty((0,))

    result = {
        "frame_offsets": np.asarray(frame_offsets, dtype=np.int32),
        "node_time": time.detach().cpu().numpy().astype(np.int16, copy=False),
        "node_frequency": frequency.detach().cpu().numpy().astype(np.int16, copy=False),
        "node_score": score.detach().cpu().numpy().astype(np.float16, copy=False),
        "node_endpoint": node_endpoint.detach().cpu().numpy().astype(np.float16, copy=False),
        "edge_source": source.detach().cpu().numpy().astype(np.int32, copy=False),
        "edge_destination": destination.detach().cpu().numpy().astype(np.int32, copy=False),
        "edge_logit": edge_logit.detach().cpu().numpy().astype(np.float16, copy=False),
    }
    return result

def track_cached_window(
    graph: Mapping[str, np.ndarray],
    cfg: TrackerConfig,
    *,
    mode: Literal["learned", "heuristic"],
) -> tuple[list[dict[str, np.ndarray | float]], dict[str, float | int]]:
    """Track a cached window without neural-network or edge-MLP execution."""
    started = perf_counter()
    frame_offsets = np.asarray(graph["frame_offsets"], dtype=np.int64)
    node_time = np.asarray(graph["node_time"], dtype=np.int64)
    node_frequency = np.asarray(graph["node_frequency"], dtype=np.int64)
    node_score = np.asarray(graph["node_score"], dtype=np.float32)
    node_endpoint = np.asarray(graph["node_endpoint"], dtype=np.float32)
    source = np.asarray(graph["edge_source"], dtype=np.int64)
    destination = np.asarray(graph["edge_destination"], dtype=np.int64)
    edge_lookup: Mapping[tuple[int, int], float]
    if mode == "learned":
        cached_lookup = graph.get("_edge_lookup")
        if isinstance(cached_lookup, Mapping):
            edge_lookup = cached_lookup
        else:
            logits = np.asarray(graph["edge_logit"], dtype=np.float32)
            edge_probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            edge_lookup = {
                (int(left), int(right)): float(score)
                for left, right, score in zip(source, destination, edge_probability)
            }
    else:
        edge_lookup = {}
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    assigned_edges = gap_links = births = endpoint_terminations = 0
    maximum_active = 0
    candidate_nodes = 0
    candidate_edges = 0
    for frame in range(len(frame_offsets) - 1):
        candidates = np.arange(frame_offsets[frame], frame_offsets[frame + 1])
        candidates = candidates[node_score[candidates] >= cfg.node_threshold]
        candidate_nodes += len(candidates)
        for track in active:
            track["gaps"] += 1
        assigned: set[int] = set()
        if active and len(candidates):
            cost = np.full(
                (len(active), len(candidates)), cfg.unmatched_cost, dtype=np.float64
            )
            probability_matrix = np.zeros_like(cost)
            for track_index, track in enumerate(active):
                last = int(track["nodes"][-1])
                if mode == "heuristic" and len(track["nodes"]) >= 2:
                    previous = int(track["nodes"][-2])
                    slope = (node_frequency[last] - node_frequency[previous]) / max(
                        node_time[last] - node_time[previous], 1
                    )
                else:
                    slope = 0.0
                for candidate_index, node in enumerate(candidates):
                    node = int(node)
                    dt = node_time[node] - node_time[last]
                    if not 0 < dt <= cfg.max_gap_frames:
                        continue
                    if abs(node_frequency[node] - node_frequency[last]) > cfg.max_frequency_jump_bins:
                        continue
                    candidate_edges += 1
                    if mode == "learned":
                        score = edge_lookup.get((last, node), 0.0)
                        probability_matrix[track_index, candidate_index] = score
                        if score > 0:
                            cost[track_index, candidate_index] = -np.log(max(score, 1e-6)) - (
                                cfg.detection_cost_weight * np.log(max(node_score[node], 1e-6))
                            )
                    else:
                        expected_frequency = node_frequency[last] + slope * dt
                        residual = abs(node_frequency[node] - expected_frequency)
                        if residual <= cfg.max_frequency_jump_bins:
                            value = (
                                residual / max(cfg.max_frequency_jump_bins, 1)
                                - cfg.detection_cost_weight * np.log(max(node_score[node], 1e-6))
                                + 0.05 * (dt - 1)
                            )
                            cost[track_index, candidate_index] = value
                            probability_matrix[track_index, candidate_index] = np.exp(
                                -max(value, 0.0)
                            )
            rows, columns = linear_sum_assignment(cost)
            for track_index, candidate_index in zip(rows.tolist(), columns.tolist()):
                node = int(candidates[candidate_index])
                score = float(probability_matrix[track_index, candidate_index])
                if mode == "learned" and score < cfg.edge_threshold:
                    continue
                if mode == "heuristic" and cost[track_index, candidate_index] >= cfg.unmatched_cost:
                    continue
                previous = int(active[track_index]["nodes"][-1])
                active[track_index]["nodes"].append(node)
                active[track_index]["edges"].append(score)
                active[track_index]["gaps"] = 0
                assigned.add(node)
                assigned_edges += 1
                if node_time[node] - node_time[previous] > 1:
                    gap_links += 1
        for node in candidates:
            node = int(node)
            start_score = (
                0.75 * node_score[node] + 0.25 * node_endpoint[node, 0]
                if mode == "learned"
                else node_score[node]
            )
            if node not in assigned and start_score >= cfg.start_threshold:
                active.append({"nodes": [node], "edges": [], "gaps": 0})
                births += 1
        retained = []
        for track in active:
            last = int(track["nodes"][-1])
            endpoint_stop = (
                mode == "learned"
                and track["gaps"] > 0
                and node_endpoint[last, 1] >= cfg.endpoint_stop_threshold
            )
            if track["gaps"] > cfg.max_gap_frames or endpoint_stop:
                finished.append(track)
                endpoint_terminations += int(endpoint_stop)
            else:
                retained.append(track)
        active = retained
        maximum_active = max(maximum_active, len(active))
    finished.extend(active)
    tracks = []
    for track in finished:
        nodes = np.asarray(track["nodes"], dtype=np.int64)
        duration = node_time[nodes[-1]] - node_time[nodes[0]] + 1
        if len(nodes) < 2 or duration < cfg.min_duration_frames:
            continue
        confidence = node_score[nodes].astype(np.float32, copy=False)
        tracks.append(
            {
                "time_indices": node_time[nodes].copy(),
                "frequency_indices": node_frequency[nodes].copy(),
                "node_confidence": confidence.copy(),
                "mean_confidence": float(confidence.mean()),
                "mean_edge_probability": float(np.mean(track["edges"]))
                if track["edges"]
                else 0.0,
            }
        )
    return tracks, {
        "mode": mode,
        "frames": len(frame_offsets) - 1,
        "candidate_nodes": candidate_nodes,
        "candidate_edges_considered": candidate_edges,
        "cached_edges": len(source),
        "assigned_edges": assigned_edges,
        "gap_links": gap_links,
        "births": births,
        "deaths": len(finished),
        "endpoint_terminations": endpoint_terminations,
        "maximum_active_tracks": maximum_active,
        "output_tracks": len(tracks),
        "runtime_seconds": perf_counter() - started,
        "tracker_config": asdict(cfg),
    }

def _physical_track(
    track: Mapping[str, Any], *, window_start_s: float
) -> dict[str, np.ndarray | float]:
    return {
        "time_s": np.asarray(track["time_indices"], dtype=np.float64) * 0.002
        + window_start_s,
        "frequency_hz": 4_000.0
        + np.asarray(track["frequency_indices"], dtype=np.float64) * 125.0,
        "node_confidence": np.asarray(track["node_confidence"], dtype=np.float32),
        "mean_confidence": float(track["mean_confidence"]),
        "mean_edge_probability": float(track["mean_edge_probability"]),
    }
