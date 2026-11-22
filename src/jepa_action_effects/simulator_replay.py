"""Deterministic PushT replay branching with physics-substep contact records."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import distribution
from typing import Any

import numpy as np
import pymunk

from jepa_action_effects.interface_audit import ACTION_BLOCK, SWM_REVISION

STATE_ATOL = 1e-10


def _validated_controls(value: Any, *, lengths: tuple[int, ...], label: str) -> np.ndarray:
    controls = np.array(value, dtype=np.float32, copy=True)
    if controls.ndim != 2 or controls.shape[1] != 2 or len(controls) not in lengths:
        allowed = " or ".join(str(length) for length in lengths)
        raise ValueError(f"{label} must have shape ({allowed}, 2)")
    if not np.isfinite(controls).all():
        raise ValueError(f"{label} must be finite")
    if np.any((controls < -1) | (controls > 1)):
        raise ValueError(f"{label} must stay within [-1, 1]; actions are never clipped")
    controls.setflags(write=False)
    return controls


@dataclass(frozen=True)
class ReplayRecipe:
    """A complete recipe for recreating one observed branch point."""

    family_id: str
    seed: int
    warmup_controls: np.ndarray
    reset_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id must be nonempty")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        object.__setattr__(
            self,
            "warmup_controls",
            _validated_controls(
                self.warmup_controls, lengths=tuple(range(10, 10_001)), label="warmup"
            ),
        )
        object.__setattr__(self, "reset_options", copy.deepcopy(dict(self.reset_options)))


@dataclass(frozen=True)
class Candidate:
    """A stable candidate identifier and one or two native action blocks."""

    candidate_id: str
    controls: np.ndarray

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        object.__setattr__(
            self,
            "controls",
            _validated_controls(
                self.controls,
                lengths=(ACTION_BLOCK, 2 * ACTION_BLOCK, 4 * ACTION_BLOCK),
                label="candidate",
            ),
        )


@dataclass(frozen=True)
class ContactEvent:
    shape_a: str
    shape_b: str
    agent_block: bool
    contact_point_count: int
    normal: tuple[float, float]
    points: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PhysicsStep:
    phase: str
    control_index: int | None
    substep_index: int
    dt: float
    agent: dict[str, Any]
    block: dict[str, Any]
    events: tuple[ContactEvent, ...]

    @property
    def agent_block_contact(self) -> bool:
        return any(event.agent_block for event in self.events)


@dataclass(frozen=True)
class BranchResult:
    candidate_id: str
    history_frames: np.ndarray
    history_controls: np.ndarray
    future_frames: np.ndarray
    observations: np.ndarray
    rewards: np.ndarray
    upstream_contacts: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    physics_steps: tuple[PhysicsStep, ...]
    completed: bool
    completed_controls: int
    expected_controls: int

    @property
    def agent_block_contact_mask(self) -> np.ndarray:
        return np.array(
            [step.agent_block_contact for step in self.physics_steps if step.phase == "candidate"],
            dtype=bool,
        )


@dataclass(frozen=True)
class FamilyResult:
    family_id: str
    seed: int
    execution_order: tuple[str, ...]
    branches: dict[str, BranchResult]


def _xy(value: Any) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _body_state(body: pymunk.Body) -> dict[str, Any]:
    return {
        "position": _xy(body.position),
        "angle": float(body.angle),
        "velocity": _xy(body.velocity),
        "angular_velocity": float(body.angular_velocity),
        "force": _xy(body.force),
        "torque": float(body.torque),
        "sleeping": bool(body.is_sleeping),
    }


def _shape_key(shape: pymunk.Shape) -> tuple[Any, ...]:
    if isinstance(shape, pymunk.Segment):
        return ("segment", *_xy(shape.a), *_xy(shape.b), float(shape.radius))
    if isinstance(shape, pymunk.Circle):
        return ("circle", *_xy(shape.offset), float(shape.radius))
    if isinstance(shape, pymunk.Poly):
        vertices = tuple(
            coordinate for vertex in shape.get_vertices() for coordinate in _xy(vertex)
        )
        return ("poly", *vertices)
    return (type(shape).__name__,)


def _point_record(point: Any) -> dict[str, Any]:
    return {
        "point_a": _xy(point.point_a),
        "point_b": _xy(point.point_b),
        "distance": float(point.distance),
    }


def _jsonable_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(options, default=lambda value: np.asarray(value).tolist()))


def recipe_to_dict(recipe: ReplayRecipe) -> dict[str, Any]:
    return {
        "family_id": recipe.family_id,
        "seed": recipe.seed,
        "warmup_controls": recipe.warmup_controls.tolist(),
        "reset_options": _jsonable_options(recipe.reset_options),
    }


def branch_to_dict(branch: BranchResult) -> dict[str, Any]:
    """Convert the detailed trace to JSON-compatible values (frames intentionally omitted)."""
    payload = asdict(branch)
    payload.pop("history_frames")
    payload.pop("future_frames")
    for key in (
        "history_controls",
        "observations",
        "rewards",
        "upstream_contacts",
        "terminated",
        "truncated",
    ):
        payload[key] = payload[key].tolist()
    payload["agent_block_contact_mask"] = branch.agent_block_contact_mask.tolist()
    return payload


def _pinned_push_t_class():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from stable_worldmodel.envs.pusht.env import PushT

    provenance = json.loads(distribution("stable-worldmodel").read_text("direct_url.json"))
    if provenance.get("vcs_info", {}).get("commit_id") != SWM_REVISION:
        raise RuntimeError("Simulator revision differs from the audited contract")
    return PushT


class InstrumentedPushT(_pinned_push_t_class()):
    """Pinned PushT with observational hooks around each physics step."""

    def __init__(self) -> None:
        self.physics_steps: list[PhysicsStep] = []
        self._phase = "reset"
        self._control_index: int | None = None
        self._substep_index = 0
        self._pending_events: list[ContactEvent] = []
        self._shape_names: dict[pymunk.Shape, str] = {}
        super().__init__(resolution=224, relative=True, render_action=False, with_target=True)

    def reset(self, seed=None, options=None):
        self.physics_steps.clear()
        self._phase = "reset"
        self._control_index = None
        self._substep_index = 0
        self._pending_events = []
        return super().reset(seed=seed, options=copy.deepcopy(options))

    def _setup(self) -> None:
        super()._setup()
        names: dict[pymunk.Shape, str] = {}
        for prefix, body in (("agent", self.agent), ("block", self.block)):
            for index, shape in enumerate(sorted(body.shapes, key=_shape_key)):
                names[shape] = f"{prefix}:{index}"
        for index, shape in enumerate(sorted(self.space.static_body.shapes, key=_shape_key)):
            names[shape] = f"wall:{index}"
        self._shape_names = names

        original_step = self.space.step

        def recorded_step(dt: float) -> None:
            self._pending_events = []
            original_step(dt)
            self.physics_steps.append(
                PhysicsStep(
                    phase=self._phase,
                    control_index=self._control_index,
                    substep_index=self._substep_index,
                    dt=float(dt),
                    agent=_body_state(self.agent),
                    block=_body_state(self.block),
                    events=tuple(self._pending_events),
                )
            )
            self._substep_index += 1

        self.space.step = recorded_step

    def _handle_collision(self, arbiter, space, data):
        super()._handle_collision(arbiter, space, data)
        shape_a, shape_b = arbiter.shapes
        bodies = {shape_a.body, shape_b.body}
        contact_set = arbiter.contact_point_set
        self._pending_events.append(
            ContactEvent(
                shape_a=self._shape_names.get(shape_a, "other"),
                shape_b=self._shape_names.get(shape_b, "other"),
                agent_block=bodies == {self.agent, self.block},
                contact_point_count=len(contact_set.points),
                normal=_xy(contact_set.normal),
                points=tuple(_point_record(point) for point in contact_set.points),
            )
        )

    def set_record_context(self, phase: str, control_index: int | None) -> None:
        if phase not in {"reset", "warmup", "candidate"}:
            raise ValueError(f"Unknown replay phase: {phase}")
        self._phase = phase
        self._control_index = control_index
        self._substep_index = 0


def _run_control(env: InstrumentedPushT, phase: str, index: int, action: np.ndarray):
    env.set_record_context(phase, index)
    return env.step(action.copy())


def run_family(
    recipe: ReplayRecipe,
    candidates: Sequence[Candidate],
    execution_order: Sequence[str] | None = None,
) -> FamilyResult:
    """Replay reset and warm-up independently for every ordered candidate."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    order = tuple(execution_order) if execution_order is not None else tuple(by_id)
    if len(order) != len(by_id) or set(order) != set(by_id):
        raise ValueError("execution_order must contain every candidate ID exactly once")

    branches: dict[str, BranchResult] = {}
    env = InstrumentedPushT()
    try:
        for candidate_id in order:
            candidate = by_id[candidate_id]
            observation, _ = env.reset(seed=recipe.seed, options=recipe.reset_options)
            frames_by_control = {0: env.render().copy()}
            warmup_ended = False
            for index, action in enumerate(recipe.warmup_controls):
                observation, _, terminated, truncated, _ = _run_control(
                    env, "warmup", index, action
                )
                if (index + 1) % ACTION_BLOCK == 0 or index + 1 >= len(recipe.warmup_controls) - 10:
                    frames_by_control[index + 1] = env.render().copy()
                if terminated or truncated:
                    warmup_ended = True
                    break
            if warmup_ended:
                raise RuntimeError(
                    f"Warm-up ended before branch point for family {recipe.family_id}"
                )

            end = len(recipe.warmup_controls)
            history_indices = (end - 10, end - 5, end)
            for index in history_indices:
                if index not in frames_by_control:
                    raise RuntimeError(f"Missing history frame at warm-up control {index}")
            history_frames = np.stack([frames_by_control[index] for index in history_indices])
            history_controls = recipe.warmup_controls[-10:].copy()
            observations: list[np.ndarray] = []
            rewards: list[float] = []
            upstream_contacts: list[int] = []
            terminated_flags: list[bool] = []
            truncated_flags: list[bool] = []
            future_frames: list[np.ndarray] = []
            completed = True
            for index, action in enumerate(candidate.controls):
                observation, reward, terminated, truncated, info = _run_control(
                    env, "candidate", index, action
                )
                observations.append(observation["state"].copy())
                rewards.append(float(reward))
                upstream_contacts.append(int(info["n_contacts"]))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                if (index + 1) % ACTION_BLOCK == 0:
                    future_frames.append(env.render().copy())
                if terminated or truncated:
                    completed = False
                    break
            branches[candidate_id] = BranchResult(
                candidate_id=candidate_id,
                history_frames=history_frames,
                history_controls=history_controls,
                future_frames=np.stack(future_frames)
                if future_frames
                else np.empty((0, 224, 224, 3), np.uint8),
                observations=np.stack(observations),
                rewards=np.asarray(rewards, dtype=np.float64),
                upstream_contacts=np.asarray(upstream_contacts, dtype=np.int64),
                terminated=np.asarray(terminated_flags, dtype=bool),
                truncated=np.asarray(truncated_flags, dtype=bool),
                physics_steps=tuple(env.physics_steps),
                completed=completed and len(observations) == len(candidate.controls),
                completed_controls=len(observations),
                expected_controls=len(candidate.controls),
            )
    finally:
        env.close()
    return FamilyResult(recipe.family_id, recipe.seed, order, branches)


def assert_dynamic_states_equal(
    left: Sequence[PhysicsStep], right: Sequence[PhysicsStep], *, atol: float = STATE_ATOL
) -> None:
    """Require matching full observable dynamics, including velocity for pose matches."""
    if len(left) != len(right):
        raise AssertionError(f"Physics trace lengths differ: {len(left)} != {len(right)}")
    fields = ("position", "angle", "velocity", "angular_velocity", "force", "torque")
    for index, (actual, expected) in enumerate(zip(left, right, strict=True)):
        if (actual.phase, actual.control_index, actual.substep_index) != (
            expected.phase,
            expected.control_index,
            expected.substep_index,
        ):
            raise AssertionError(f"Physics timestamp differs at step {index}")
        for body in ("agent", "block"):
            for field_name in fields:
                np.testing.assert_allclose(
                    actual.__getattribute__(body)[field_name],
                    expected.__getattribute__(body)[field_name],
                    atol=atol,
                    rtol=0,
                    err_msg=f"{body}.{field_name} differs at physics step {index}",
                )
            if (
                actual.__getattribute__(body)["sleeping"]
                != expected.__getattribute__(body)["sleeping"]
            ):
                raise AssertionError(f"{body}.sleeping differs at physics step {index}")
