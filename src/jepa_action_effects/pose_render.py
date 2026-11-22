"""Render-only joint agent/block poses for MEAS-001 inversion (D032, D033).

`SceneContext` holds only what the observed scene determines: reset seed, reset options
(start state and goal) and executed warm-up controls. It has no future state, frame or
candidate label, so a search built on it cannot read the answer.

`SceneRenderer` replays the warm-up once, then renders arbitrary poses by assigning the
kinematic agent position and the block pose (angle before position, then reindex, as in
`gallery_render`). It never steps physics. After each batch it restores the observed pose
and verifies the baseline RGB and nuisance state bit-for-bit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymunk
from shapely.geometry import LineString, Point

from jepa_action_effects.gallery_render import WIDTH, block_polygons, set_block_pose
from jepa_action_effects.simulator_replay import InstrumentedPushT

PENETRATION_TOLERANCE = 1.0  # D032: uniform allowance for small contact overlaps
# The agent is a kinematic body: walls do not stop it, and real frames show it inside walls
# or entirely off-canvas. Checking it against walls would reject physically realized states,
# so by default only the block is checked against walls and workspace (proposed D033
# amendment; see MEAS-001_ENGINEERING.md).
AGENT_WALL_CHECKS = False


@dataclass(frozen=True)
class SceneContext:
    seed: int
    reset_options: Mapping[str, Any]
    warmup_controls: tuple[tuple[float, float], ...]
    scene: str

    @classmethod
    def from_family(cls, family: Mapping[str, Any], scene: str) -> SceneContext:
        """Read only observed-scene fields; future states and frames are never touched."""
        metadata = family["metadata"]
        controls = np.asarray(family["warmup_controls"], dtype=np.float64)
        return cls(
            seed=int(metadata["seed"]),
            reset_options={key: list(value) for key, value in metadata["reset_options"].items()},
            warmup_controls=tuple((float(a), float(b)) for a, b in controls),
            scene=scene,
        )


def pose_of(env: InstrumentedPushT) -> np.ndarray:
    return np.array(
        [*env.agent.position, *env.block.position, float(env.block.angle) % (2 * math.pi)],
        dtype=np.float64,
    )


def set_pose(env: InstrumentedPushT, pose: np.ndarray) -> None:
    pose = np.asarray(pose, dtype=np.float64)
    env.agent.position = (float(pose[0]), float(pose[1]))
    env.space.reindex_shapes_for_body(env.agent)
    set_block_pose(env, float(pose[4]), (float(pose[2]), float(pose[3])))


def joint_geometry(
    env: InstrumentedPushT, tolerance: float, agent_walls: bool = AGENT_WALL_CHECKS
) -> tuple[bool, str, float]:
    """Penetration checks: agent–block, block–wall and block-workspace always; agent–wall and
    agent-workspace only when `agent_walls` (the literal D032 reading)."""
    polygons = block_polygons(env)
    circle = next(iter(env.agent.shapes))
    if len(env.agent.shapes) != 1 or not isinstance(circle, pymunk.Circle):
        raise ValueError("Expected one circular agent collision shape")
    centre = env.agent.local_to_world(circle.offset)
    point = Point(centre.x, centre.y)
    agent_block = min(poly.distance(point) - circle.radius for poly in polygons)
    block_wall = agent_wall = math.inf
    for shape in env.space.static_body.shapes:
        if not isinstance(shape, pymunk.Segment):
            raise ValueError("Unexpected wall collision shape")
        a, b = shape.body.local_to_world(shape.a), shape.body.local_to_world(shape.b)
        segment = LineString(((a.x, a.y), (b.x, b.y)))
        block_wall = min(block_wall, *(poly.distance(segment) - shape.radius for poly in polygons))
        agent_wall = min(agent_wall, segment.distance(point) - shape.radius - circle.radius)
    bounds = [poly.bounds for poly in polygons]
    block_edge = min(
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        WIDTH - max(item[2] for item in bounds),
        WIDTH - max(item[3] for item in bounds),
    )
    agent_edge = min(centre.x, centre.y, WIDTH - centre.x, WIDTH - centre.y) - circle.radius
    checks = [
        ("agent_block_penetration", agent_block),
        ("block_wall_penetration", block_wall),
        ("block_outside_workspace", block_edge),
    ]
    if agent_walls:
        checks += [("agent_wall_penetration", agent_wall), ("agent_outside_workspace", agent_edge)]
    minimum = min(value for _, value in checks)
    for reason, value in checks:
        if value < -tolerance:
            return False, reason, minimum
    return True, "valid", minimum


class SceneRenderer:
    """Renders poses in one observed scene; restores and verifies after every batch."""

    def __init__(
        self,
        context: SceneContext,
        tolerance: float = PENETRATION_TOLERANCE,
        agent_walls: bool = AGENT_WALL_CHECKS,
    ):
        self.context = context
        self.tolerance = float(tolerance)
        self.agent_walls = bool(agent_walls)
        self.env = InstrumentedPushT()
        self.env.reset(seed=context.seed, options=dict(context.reset_options))
        for action in context.warmup_controls:
            _, _, terminated, truncated, _ = self.env.step(np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                raise ValueError("Warm-up ended before the observed frame")
        self.baseline_pose = pose_of(self.env)
        self.baseline_rgb = self.env.render().copy()
        self.invariants = self._invariants()
        self.renders = 0

    def _invariants(self) -> tuple:
        env = self.env
        return (
            tuple(np.asarray(env.goal_pose, dtype=np.float64)),
            len(env.physics_steps),
            tuple(env.agent.velocity),
            tuple(env.block.velocity),
            float(env.block.angular_velocity),
        )

    def restore(self) -> None:
        set_pose(self.env, self.baseline_pose)
        if self._invariants() != self.invariants:
            raise RuntimeError("Rendering changed nuisance state or stepped physics")
        if not np.array_equal(self.env.render(), self.baseline_rgb):
            raise RuntimeError("Observed RGB did not restore exactly")

    def check(self, poses: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Geometry validity per pose, without rendering."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 5)
        valid, reasons, clearances = np.zeros(len(poses), dtype=bool), [], np.zeros(len(poses))
        try:
            for index, pose in enumerate(poses):
                set_pose(self.env, pose)
                ok, reason, clearance = joint_geometry(self.env, self.tolerance, self.agent_walls)
                valid[index], clearances[index] = ok, clearance
                reasons.append(reason)
        finally:
            self.restore()
        return valid, reasons, clearances

    def render(self, poses: np.ndarray) -> np.ndarray:
        """uint8 frames (n, 224, 224, 3); callers must pass geometry-valid poses."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 5)
        frames = []
        try:
            for pose in poses:
                set_pose(self.env, pose)
                frames.append(self.env.render().copy())
        finally:
            self.restore()
        self.renders += len(poses)
        return np.stack(frames) if frames else np.zeros((0, 224, 224, 3), dtype=np.uint8)

    def close(self) -> None:
        self.env.close()
