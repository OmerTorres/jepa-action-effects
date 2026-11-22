"""Render-only PushT block translations with geometry and identity checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymunk
from shapely.geometry import LineString, Point, Polygon

from jepa_action_effects.latent_cache import array_digest
from jepa_action_effects.simulator_replay import InstrumentedPushT

WIDTH = 512.0
IMAGE_SIZE = 224
GEOMETRY_TOLERANCE = 1e-6


def directions(seed: int, identifier: str, label: str) -> np.ndarray:
    token = hashlib.sha256(f"{seed}:{identifier}:{label}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    origin = float(rng.uniform(0, 2 * math.pi))
    angles = origin + np.arange(8) * math.pi / 4
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def block_polygons(env: InstrumentedPushT) -> list[Polygon]:
    result = []
    for shape in env.block.shapes:
        if not isinstance(shape, pymunk.Poly):
            raise ValueError("Unexpected block collision shape")
        points = [env.block.local_to_world(vertex) for vertex in shape.get_vertices()]
        result.append(Polygon([(point.x, point.y) for point in points]))
    if len(result) != 2 or not all(poly.is_valid and poly.area > 0 for poly in result):
        raise ValueError("Invalid transformed T-block geometry")
    return result


def geometry(env: InstrumentedPushT) -> tuple[bool, str, float]:
    """Reject positive penetration of the held agent, walls, or image boundary."""
    polygons = block_polygons(env)
    agent_shapes = list(env.agent.shapes)
    if len(agent_shapes) != 1 or not isinstance(agent_shapes[0], pymunk.Circle):
        raise ValueError("Expected one circular agent collision shape")
    circle = agent_shapes[0]
    center = env.agent.local_to_world(circle.offset)
    point = Point(center.x, center.y)
    agent_clearance = min(poly.distance(point) - circle.radius for poly in polygons)
    wall_clearance = float("inf")
    for shape in env.space.static_body.shapes:
        if not isinstance(shape, pymunk.Segment):
            raise ValueError("Unexpected wall collision shape")
        a = shape.body.local_to_world(shape.a)
        b = shape.body.local_to_world(shape.b)
        segment = LineString(((a.x, a.y), (b.x, b.y)))
        wall_clearance = min(
            wall_clearance, *(poly.distance(segment) - shape.radius for poly in polygons)
        )
    bounds = [poly.bounds for poly in polygons]
    boundary_clearance = min(
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        WIDTH - max(item[2] for item in bounds),
        WIDTH - max(item[3] for item in bounds),
    )
    minimum = min(agent_clearance, wall_clearance, boundary_clearance)
    if agent_clearance < -GEOMETRY_TOLERANCE:
        return False, "agent_penetration", minimum
    if wall_clearance < -GEOMETRY_TOLERANCE:
        return False, "wall_penetration", minimum
    if boundary_clearance < -GEOMETRY_TOLERANCE:
        return False, "outside_workspace", minimum
    return True, "valid", minimum


def set_block_pose(env: InstrumentedPushT, angle: float, position: tuple[float, float]) -> None:
    """Pymunk angle assignment shifts the offset-center-of-mass T-block position."""
    env.block.angle = float(angle)
    env.block.position = tuple(float(value) for value in position)
    env.space.reindex_shapes_for_body(env.block)


def _footprint(polygons: list[Polygon]) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords, dtype=np.float64)
        xy[:, 0] *= IMAGE_SIZE / WIDTH
        xy[:, 1] *= IMAGE_SIZE / WIDTH
        cv2.fillPoly(mask, [np.rint(xy).astype(np.int32)], 1)
    return cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8))


def local_image_change(baseline: np.ndarray, changed: np.ndarray, before, after) -> bool:
    if baseline.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or changed.shape != baseline.shape:
        raise ValueError("Unexpected rendered frame size")
    permitted = _footprint(before) | _footprint(after)
    difference = np.any(baseline != changed, axis=2)
    return not bool(np.any(difference & ~permitted.astype(bool)))


def replay_endpoint(env: InstrumentedPushT, family: dict, index: int) -> np.ndarray:
    metadata = family["metadata"]
    env.reset(seed=metadata["seed"], options=metadata["reset_options"])
    for action in family["warmup_controls"]:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise ValueError("Warm-up ended before endpoint")
    for action in family["candidate_controls"][index]:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise ValueError("Candidate ended before endpoint")
    state = np.asarray(
        [*env.agent.position, *env.block.position, env.block.angle], dtype=np.float64
    )
    if not np.allclose(state, family["future_states"][index, -1], atol=1e-10, rtol=0):
        raise ValueError("Saved state and direct replay are misaligned")
    frame = env.render().copy()
    if not np.array_equal(frame, family["future_frames"][index, -1]):
        raise ValueError("Saved terminal RGB and direct replay differ")
    return frame


def _fixed_nuisance(env: InstrumentedPushT) -> tuple:
    return (
        tuple(env.agent.position),
        float(env.agent.angle),
        tuple(env.agent.velocity),
        tuple(np.asarray(env.goal_pose, dtype=np.float64)),
        len(env.physics_steps),
    )


def translated_frame(
    env: InstrumentedPushT,
    baseline: np.ndarray,
    original: tuple[tuple[float, float], float],
    offset: np.ndarray,
    before: list[Polygon],
) -> tuple[np.ndarray | None, str, float]:
    nuisance = _fixed_nuisance(env)
    target = tuple(np.asarray(original[0]) + np.asarray(offset, dtype=np.float64))
    try:
        set_block_pose(env, original[1], target)
        valid, reason, clearance = geometry(env)
        if not valid:
            return None, reason, clearance
        frame = env.render().copy()
        if not local_image_change(baseline, frame, before, block_polygons(env)):
            raise ValueError("Block intervention changed pixels outside block footprints")
        if np.array_equal(frame, baseline):
            return None, "identical_rgb", clearance
        return frame, "valid", clearance
    finally:
        set_block_pose(env, original[1], original[0])
        if _fixed_nuisance(env) != nuisance:
            raise ValueError("Renderer intervention changed nuisance or stepped physics")
        if not np.array_equal(env.render(), baseline):
            raise ValueError("Original RGB did not restore exactly")


def render_candidate(
    env: InstrumentedPushT, family: dict, index: int, identifier: str, seeds: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    baseline = replay_endpoint(env, family, index)
    original = (tuple(env.block.position), float(env.block.angle))
    before = block_polygons(env)
    original_footprints = [np.asarray(poly.exterior.coords).tolist() for poly in before]
    agent_circle = next(iter(env.agent.shapes))
    if not isinstance(agent_circle, pymunk.Circle):
        raise ValueError("Expected circular agent")
    _, _, baseline_clearance = geometry(env)
    images = {"true": baseline}
    meta = {
        "identifier": identifier,
        "candidate_id": family["metadata"]["candidate_ids"][index],
        "candidate_index": index,
        "baseline_clearance": baseline_clearance,
        "baseline_rgb_sha256": array_digest(baseline),
        "agent_center": list(env.agent.position),
        "agent_radius": float(agent_circle.radius),
        "footprints": {"true": original_footprints},
        "query_attempts": [],
        "scales": {},
    }
    for scale in (10.24, 20.48):
        label = f"negative_{scale:g}"
        attempts = []
        selected = []
        for attempt, direction in enumerate(
            directions(seeds["templates"], identifier, f"{index}:{label}")
        ):
            if len(selected) == 4:
                break
            offset = direction * scale
            frame, reason, clearance = translated_frame(env, baseline, original, offset, before)
            attempts.append(
                {
                    "attempt": attempt,
                    "offset": offset.tolist(),
                    "reason": reason,
                    "clearance": clearance,
                }
            )
            if frame is not None:
                key = f"{label}_{len(selected)}"
                images[key] = frame
                meta["footprints"][key] = [
                    (np.asarray(vertices) + offset).tolist() for vertices in original_footprints
                ]
                selected.append(key)
        meta["scales"][label] = {"selected": selected, "attempts": attempts}
    queries = []
    for attempt, direction in enumerate(directions(seeds["queries"], identifier, f"{index}:query")):
        if len(queries) == 2:
            break
        offset = direction * 1.28
        frame, reason, clearance = translated_frame(env, baseline, original, offset, before)
        if frame is not None and any(np.array_equal(frame, images[key]) for key in queries):
            frame, reason = None, "duplicate_query_rgb"
        meta["query_attempts"].append(
            {
                "attempt": attempt,
                "offset": offset.tolist(),
                "reason": reason,
                "clearance": clearance,
            }
        )
        if frame is not None:
            key = f"query_{len(queries)}"
            images[key] = frame
            meta["footprints"][key] = [
                (np.asarray(vertices) + offset).tolist() for vertices in original_footprints
            ]
            queries.append(key)
    meta["queries"] = queries
    meta["images_sha256"] = {key: array_digest(value) for key, value in images.items()}
    return meta, images


def save_rendered(path: Path, meta: dict, images: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        np.savez_compressed(
            temporary, metadata=np.array(json.dumps(meta, sort_keys=True)), **images
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rendered(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"].item()))
            images = {key: archive[key] for key in archive.files if key != "metadata"}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid rendered gallery {path}") from error
    if set(images) != set(meta["images_sha256"]):
        raise ValueError("Rendered gallery image inventory mismatch")
    if any(array_digest(value) != meta["images_sha256"][key] for key, value in images.items()):
        raise ValueError("Rendered gallery image hash mismatch")
    return meta, images
