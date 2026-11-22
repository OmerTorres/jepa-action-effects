"""Model-free joint-pose search with abstention (MEAS-001; D032, D033).

A pose is (agent x, agent y, block x, block y, block angle). The search sees only an
`Objective`: a cheap geometry-validity check and a costly residual. It never sees a true
pose, so the same code serves analytic fixtures and the render-and-encode instrument.

Each query runs two independently seeded searches (scrambled Sobol proposals, then batched
compass refinement from distinct starts), followed by nuisance challenges around the best
pose. `classify` turns the evidence into one status:

1. `unsupported_geometry`: no geometry-valid candidate was found.
2. `optimizer_unresolved`: a search did not converge, or the two searches' best block
   centres differ by more than the restart tolerance.
3. `residual_reject`: the best residual exceeds τ.
4. `ambiguous`: candidates within τ of the best span block centres wider than the
   near-optimal diameter, including those found by nuisance challenges.
5. `accepted`: a block centre, not a unique physical state.

A finite search cannot prove that no alternative was missed; acceptance is empirical
search stability only. Inverting a latent mean need not give a physical mean (D031).

Version 2 (D037, after MEAS-REVIEW-001) changes three rules and leaves version 1 intact so
recorded v1 results reproduce:
- each search ends with an independent Nelder–Mead refinement from its own best pose;
- the reported pose is the best *independent search* result; a better nuisance basin far
  from it, or an unfinished nuisance challenge, is `optimizer_unresolved`;
- acceptance (τ_fit) and the near-optimal ambiguity band (τ_amb) are separate. τ_amb is
  label-free: the smallest residual increase among 1.28-unit block offsets of the reported
  pose (the least-sensitive direction), so a distant candidate counts as an alternative
  only if it fits at least as well as the cheapest nearby displacement.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc

POSE_DIM = 5
ANGLE = 4
BLOCK = slice(2, 4)
STATUSES = (
    "accepted",
    "residual_reject",
    "ambiguous",
    "optimizer_unresolved",
    "unsupported_geometry",
)
PHASES = ("coarse", "local", "nuisance", "refine")


@dataclass(frozen=True)
class SearchConfig:
    """Every search rule; frozen before calibration (D032)."""

    # Agent centres span every distinct rendering: beyond ±16 of the canvas the radius-15
    # agent is invisible, so further positions render identically. Block centres span the
    # walls; geometry checks reject invalid block poses.
    lower: tuple[float, ...] = (-16.0, -16.0, 7.0, 7.0, 0.0)
    upper: tuple[float, ...] = (528.0, 528.0, 504.0, 504.0, 2 * math.pi)
    searches: int = 2
    coarse: int = 256
    local_starts: int = 4
    local_evaluations: int = 160
    nuisance_evaluations: int = 256
    initial_step: tuple[float, ...] = (16.0, 16.0, 16.0, 16.0, 0.25)
    minimum_step: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25, 0.004)
    start_separation: float = 10.24
    attempt_factor: int = 8
    batch: int = 64
    nuisance_agent: float = 5.12
    nuisance_angle_degrees: float = 10.0
    nuisance_step: float = 4.0
    restart_tolerance: float = 1.28
    near_optimal_diameter: float = 2.56
    # Version 2 (D037). Defaults keep version-1 behaviour exactly.
    version: int = 1
    refine_evaluations: int = 0
    refine_simplex: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 0.05)
    refine_xatol: float = 0.1
    refine_fatol: float = 1e-7
    resolution_offset: float = 1.28
    resolution_directions: int = 8

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "initial_step", "minimum_step", "refine_simplex"):
            if len(getattr(self, name)) != POSE_DIM:
                raise ValueError(f"{name} needs {POSE_DIM} values")
        if self.batch & (self.batch - 1):
            raise ValueError("Sobol batches must be a power of two")
        if any(m <= 0 or m > i for m, i in zip(self.minimum_step, self.initial_step, strict=True)):
            raise ValueError("Minimum steps must be positive and at most the initial steps")

    @property
    def evaluation_cap(self) -> int:
        per_search = self.coarse + self.local_starts * self.local_evaluations
        per_search += self.refine_evaluations
        return self.searches * per_search + self.nuisance_evaluations

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evaluation_cap": self.evaluation_cap}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SearchConfig:
        values = {k: v for k, v in payload.items() if k != "evaluation_cap"}
        for key in ("lower", "upper", "initial_step", "minimum_step", "refine_simplex"):
            if key in values:
                values[key] = tuple(float(v) for v in values[key])
        return cls(**values)

    def legacy_dict(self) -> dict[str, Any]:
        """Version-1 serialization (no v2 keys), for comparing with v1 sealed specs."""
        payload = self.to_dict()
        if self.version == 1:
            for key in V2_FIELDS:
                payload.pop(key, None)
        return payload


V2_FIELDS = (
    "version",
    "refine_evaluations",
    "refine_simplex",
    "refine_xatol",
    "refine_fatol",
    "resolution_offset",
    "resolution_directions",
)


class Objective(Protocol):
    def valid(self, poses: np.ndarray) -> np.ndarray:
        """Boolean geometry validity per pose; cheap, never renders or encodes."""

    def evaluate(self, poses: np.ndarray) -> np.ndarray:
        """Residual per (valid) pose; the costly render-and-encode step."""


def seed_sequence(master: int, query_id: str, stream: int) -> np.random.SeedSequence:
    """Independent, order-free stream per query and purpose."""
    token = int.from_bytes(hashlib.sha256(query_id.encode()).digest()[:8], "big")
    return np.random.SeedSequence([int(master), token, int(stream)])


def canonical(poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=np.float64).reshape(-1, POSE_DIM)
    poses[:, ANGLE] = np.mod(poses[:, ANGLE], 2 * math.pi)
    return poses


@dataclass
class Trace:
    """Every attempted evaluation, including cached duplicates."""

    poses: list[np.ndarray] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    searches: list[int] = field(default_factory=list)
    cached: list[bool] = field(default_factory=list)
    evaluations: int = 0
    invalid_attempts: int = 0

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "poses": np.array(self.poses, dtype=np.float64).reshape(-1, POSE_DIM),
            "residuals": np.array(self.residuals, dtype=np.float64),
            "phases": np.array(self.phases, dtype=np.int8),
            "searches": np.array(self.searches, dtype=np.int8),
            "cached": np.array(self.cached, dtype=bool),
        }


class _Evaluator:
    """Charges actual objective calls against a cap; exact duplicates come from a cache."""

    def __init__(self, objective: Objective, trace: Trace, cap: int):
        self.objective = objective
        self.trace = trace
        self.cap = cap
        self.cache: dict[bytes, float] = {}

    @property
    def remaining(self) -> int:
        return self.cap - self.trace.evaluations

    def __call__(self, poses: np.ndarray, phase: str, search: int, limit: int) -> np.ndarray:
        """Evaluate up to `limit` new poses in order; returns residuals (NaN if unevaluated)."""
        poses = canonical(poses)
        residuals = np.full(len(poses), np.nan)
        fresh = [i for i in range(len(poses)) if poses[i].tobytes() not in self.cache]
        allowed = set(fresh[: max(0, min(limit, self.remaining))])
        computed = [i for i in fresh if i in allowed]
        if computed:
            values = np.asarray(self.objective.evaluate(poses[computed]), dtype=np.float64)
            if values.shape != (len(computed),):
                raise ValueError("Objective returned the wrong number of residuals")
            self.trace.evaluations += len(computed)
            for index, value in zip(computed, values, strict=True):
                self.cache[poses[index].tobytes()] = float(value)
        for index in range(len(poses)):
            key = poses[index].tobytes()
            if key not in self.cache:
                continue
            residuals[index] = self.cache[key]
            self.trace.poses.append(poses[index])
            self.trace.residuals.append(residuals[index])
            self.trace.phases.append(PHASES.index(phase))
            self.trace.searches.append(search)
            self.trace.cached.append(index not in computed)
        return residuals


def _order(poses: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Ascending residual, ties broken lexicographically by pose (deterministic)."""
    keys = [*(poses[:, d] for d in reversed(range(POSE_DIM))), residuals]
    return np.lexsort(keys)


def _valid(objective: Objective, poses: np.ndarray, trace: Trace) -> np.ndarray:
    valid = np.asarray(objective.valid(poses), dtype=bool)
    trace.invalid_attempts += int((~valid).sum())
    return valid


def _compass(
    objective: Objective,
    evaluate: _Evaluator,
    trace: Trace,
    start: np.ndarray,
    start_residual: float,
    axes: Sequence[int],
    step: np.ndarray,
    minimum: np.ndarray,
    budget: int,
    config: SearchConfig,
    phase: str,
    search: int,
) -> tuple[np.ndarray, float, bool]:
    """Batched coordinate search over `axes`.

    Each iteration evaluates ±step on every axis in one batch. Improving axes are combined
    into one extra candidate; the best of that and the best single move is taken. Axes that
    did not improve halve their step. The search has converged when an iteration with every
    step at its minimum finds no improvement; running out of budget first is non-convergence.
    """
    x, fx = canonical(start)[0], float(start_residual)
    step, minimum = np.array(step, dtype=float), np.array(minimum, dtype=float)
    lower, upper = np.array(config.lower), np.array(config.upper)
    used = 0

    def moved(point: np.ndarray, axis: int, delta: float) -> np.ndarray:
        candidate = point.copy()
        candidate[axis] += delta
        if axis != ANGLE:
            candidate[axis] = min(max(candidate[axis], lower[axis]), upper[axis])
        return candidate

    def run(points: list[np.ndarray]) -> tuple[np.ndarray, bool]:
        """Residuals (∞ if invalid) and whether the budget cut off any valid point."""
        nonlocal used
        values = np.full(len(points), np.inf)
        if not points:
            return values, False
        array = canonical(np.array(points))
        valid = _valid(objective, array, trace)
        if valid.any():
            before = trace.evaluations
            values[valid] = evaluate(array[valid], phase, search, budget - used)
            used += trace.evaluations - before
        incomplete = bool(np.isnan(values).any())
        return np.where(np.isnan(values), np.inf, values), incomplete

    while True:
        if used >= budget or evaluate.remaining <= 0:
            return x, fx, False
        moves = [(axis, sign) for axis in axes for sign in (1.0, -1.0)]
        points = [moved(x, axis, sign * step[axis]) for axis, sign in moves]
        values, incomplete = run(points)
        improving: dict[int, tuple[float, float]] = {}
        for (axis, sign), value in zip(moves, values, strict=True):
            if value < fx and (axis not in improving or value < improving[axis][1]):
                improving[axis] = (sign, value)
        if not improving:
            if incomplete:
                return x, fx, False
            if np.all(step[list(axes)] <= minimum[list(axes)] + 1e-12):
                return x, fx, True
            for axis in axes:
                step[axis] = max(step[axis] / 2.0, minimum[axis])
            continue
        best_index = int(np.argmin(values))
        options = [(float(values[best_index]), points[best_index])]
        if len(improving) > 1:
            combined = x.copy()
            for axis, (sign, _) in improving.items():
                combined = moved(combined, axis, sign * step[axis])
            options.append((float(run([combined])[0][0]), combined))
        value, point = min(options, key=lambda item: (item[0], tuple(canonical(item[1])[0])))
        x, fx = canonical(point)[0], value
        for axis in axes:
            if axis not in improving:
                step[axis] = max(step[axis] / 2.0, minimum[axis])


class _Stop(Exception):
    """The refinement budget or the shared evaluation cap ran out."""


def _refine(
    objective: Objective,
    evaluate: _Evaluator,
    trace: Trace,
    config: SearchConfig,
    start: np.ndarray,
    start_residual: float,
    search: int,
) -> tuple[np.ndarray, float, bool]:
    """Nelder–Mead from one search's own best pose, charged to that search (D037).

    Invalid poses get a large finite penalty. Running out of budget first is
    non-convergence; the best pose seen so far is still returned.
    """
    from scipy.optimize import minimize

    best = [canonical(start)[0], float(start_residual)]
    used = 0

    def fun(x: np.ndarray) -> float:
        nonlocal used
        pose = canonical(x)
        if not _valid(objective, pose, trace)[0]:
            return 1e6
        if used >= config.refine_evaluations or evaluate.remaining <= 0:
            raise _Stop
        before = trace.evaluations
        value = float(evaluate(pose, "refine", search, 1)[0])
        used += trace.evaluations - before
        if not np.isfinite(value):
            raise _Stop
        if value < best[1]:
            best[0], best[1] = pose[0], value
        return value

    x0 = canonical(start)[0]
    simplex = np.vstack([x0, x0 + np.diag(np.asarray(config.refine_simplex, float))])
    try:
        result = minimize(
            fun,
            x0,
            method="Nelder-Mead",
            options={
                "initial_simplex": simplex,
                "maxfev": 10 * config.refine_evaluations,  # cached repeats are free
                "xatol": config.refine_xatol,
                "fatol": config.refine_fatol,
            },
        )
        converged = bool(result.success)
    except _Stop:
        converged = False
    return best[0], best[1], converged


def resolution(objective: Objective, pose: np.ndarray, config: SearchConfig) -> dict[str, Any]:
    """Label-free τ_amb: least residual increase among block offsets of the reported pose.

    Probes are evaluated directly, outside the trace, so they never enter the
    near-optimal set they calibrate.
    """
    pose = canonical(pose)[0]
    angles = np.arange(config.resolution_directions) * 2 * math.pi / config.resolution_directions
    probes = np.repeat(pose[None], len(angles), axis=0)
    probes[:, 2] += config.resolution_offset * np.cos(angles)
    probes[:, 3] += config.resolution_offset * np.sin(angles)
    valid = np.asarray(objective.valid(probes), dtype=bool)
    if valid.sum() < config.resolution_directions // 2:
        return {"tau_amb": None, "valid_probes": int(valid.sum()), "increases": []}
    centre = float(np.asarray(objective.evaluate(pose[None]), float)[0])
    values = np.asarray(objective.evaluate(probes[valid]), float)
    increases = values - centre
    return {
        "tau_amb": float(max(np.min(increases), 0.0)),
        "median_increase": float(np.median(increases)),
        "valid_probes": int(valid.sum()),
        "increases": increases.tolist(),
        "centre_residual": centre,
    }


@dataclass
class SearchResult:
    best_pose: np.ndarray | None
    best_residual: float
    converged: bool
    coarse_valid: int


def _search(
    objective: Objective,
    evaluate: _Evaluator,
    trace: Trace,
    config: SearchConfig,
    seed: np.random.SeedSequence,
    search: int,
) -> SearchResult:
    lower, upper = np.array(config.lower), np.array(config.upper)
    sampler = qmc.Sobol(d=POSE_DIM, scramble=True, seed=np.random.default_rng(seed))
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < config.coarse and attempts < config.attempt_factor * config.coarse:
        proposals = canonical(lower + sampler.random(config.batch) * (upper - lower))
        attempts += len(proposals)
        accepted.extend(proposals[_valid(objective, proposals, trace)])
    accepted_array = np.array(accepted[: config.coarse]).reshape(-1, POSE_DIM)
    if not len(accepted_array):
        return SearchResult(None, math.inf, False, 0)
    residuals = np.concatenate(
        [
            evaluate(accepted_array[s : s + config.batch], "coarse", search, config.batch)
            for s in range(0, len(accepted_array), config.batch)
        ]
    )
    finite = np.isfinite(residuals)
    poses, residuals = accepted_array[finite], residuals[finite]
    starts: list[int] = []
    for index in _order(poses, residuals):
        centre = poses[index, BLOCK]
        if all(np.linalg.norm(centre - poses[s, BLOCK]) >= config.start_separation for s in starts):
            starts.append(int(index))
        if len(starts) == config.local_starts:
            break
    best_pose, best_residual, best_converged = None, math.inf, False
    for index in starts:
        pose, value, converged = _compass(
            objective,
            evaluate,
            trace,
            poses[index],
            residuals[index],
            range(POSE_DIM),
            np.array(config.initial_step),
            np.array(config.minimum_step),
            config.local_evaluations,
            config,
            "local",
            search,
        )
        if value < best_residual or (
            value == best_residual and best_pose is not None and tuple(pose) < tuple(best_pose)
        ):
            best_pose, best_residual, best_converged = pose, value, converged
    if config.version >= 2 and config.refine_evaluations > 0 and best_pose is not None:
        # D037: this search's own refinement; its convergence is the search's convergence.
        pose, value, converged = _refine(
            objective, evaluate, trace, config, best_pose, best_residual, search
        )
        if value <= best_residual:
            best_pose, best_residual = pose, value
        best_converged = converged
    return SearchResult(best_pose, best_residual, best_converged, len(accepted_array))


def _nuisance(
    objective: Objective,
    evaluate: _Evaluator,
    trace: Trace,
    config: SearchConfig,
    best: np.ndarray,
) -> list[dict[str, Any]]:
    """Displace the fitted agent and block angle, then refit block position only."""
    angle = math.radians(config.nuisance_angle_degrees)
    offsets = [
        (0, config.nuisance_agent),
        (0, -config.nuisance_agent),
        (1, config.nuisance_agent),
        (1, -config.nuisance_agent),
        (ANGLE, angle),
        (ANGLE, -angle),
    ]
    each = config.nuisance_evaluations // len(offsets)
    step = np.zeros(POSE_DIM)
    step[BLOCK] = config.nuisance_step
    minimum = np.array(config.minimum_step)
    records = []
    for axis, offset in offsets:
        start = best.copy()
        start[axis] += offset
        start = canonical(start)
        record: dict[str, Any] = {"axis": axis, "offset": offset}
        if not _valid(objective, start, trace)[0]:
            records.append({**record, "status": "invalid_start"})
            continue
        before = trace.evaluations
        value = evaluate(start, "nuisance", -1, each)[0]
        if not np.isfinite(value):
            records.append({**record, "status": "budget"})
            continue
        spent = trace.evaluations - before
        pose, fitted, converged = _compass(
            objective,
            evaluate,
            trace,
            start[0],
            value,
            (2, 3),
            step.copy(),
            minimum,
            max(0, each - spent),
            config,
            "nuisance",
            -1,
        )
        records.append(
            {
                **record,
                "status": "fitted",
                "converged": converged,
                "residual": fitted,
                "block": pose[BLOCK].tolist(),
            }
        )
    return records


@dataclass
class Inversion:
    """Search evidence for one query, before τ is applied."""

    searches: list[SearchResult]
    nuisance: list[dict[str, Any]]
    trace: Trace
    config: SearchConfig
    resolution: dict[str, Any] | None = None

    @property
    def reported(self) -> tuple[np.ndarray | None, float]:
        """Best independent-search result (v2 reported pose); deterministic ties."""
        found = [s for s in self.searches if s.best_pose is not None]
        if not found:
            return None, math.inf
        best = min(found, key=lambda s: (s.best_residual, tuple(canonical(s.best_pose)[0])))
        return canonical(best.best_pose)[0], float(best.best_residual)

    @property
    def best(self) -> tuple[np.ndarray | None, float]:
        arrays = self.trace.arrays()
        finite = np.isfinite(arrays["residuals"])
        if not finite.any():
            return None, math.inf
        index = _order(arrays["poses"][finite], arrays["residuals"][finite])[0]
        return arrays["poses"][finite][index], float(arrays["residuals"][finite][index])


def invert(objective: Objective, config: SearchConfig, master: int, query_id: str) -> Inversion:
    """Run both searches and the nuisance challenges for one target."""
    trace = Trace()
    evaluate = _Evaluator(objective, trace, config.evaluation_cap)
    results = [
        _search(objective, evaluate, trace, config, seed_sequence(master, query_id, s), s)
        for s in range(config.searches)
    ]
    inversion = Inversion(results, [], trace, config)
    best, _ = inversion.reported if config.version >= 2 else inversion.best
    if best is not None:
        inversion.nuisance = _nuisance(objective, evaluate, trace, config, best)
    if trace.evaluations > config.evaluation_cap:
        raise AssertionError("Evaluation cap exceeded")
    if config.version >= 2 and best is not None:
        inversion.resolution = resolution(objective, best, config)
    return inversion


def diameter(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    unique = np.unique(points, axis=0)
    return float(pdist(unique).max()) if len(unique) > 1 else 0.0


def classify(
    inversion: Inversion, tau: float, ambiguity_tau: float | None = None
) -> dict[str, Any]:
    """One status plus diagnostics; the best pose is kept even for abstentions.

    Version 1 uses τ for both acceptance and the near-optimal band. Version 2 (D037) uses
    τ for acceptance only and `ambiguity_tau` (default: the inversion's label-free
    resolution) for the band, and reports only an independently supported pose.
    """
    if inversion.config.version >= 2:
        return _classify_v2(inversion, tau, ambiguity_tau)
    config = inversion.config
    best, best_residual = inversion.best
    searches = inversion.searches
    arrays = inversion.trace.arrays()
    diagnostics: dict[str, Any] = {
        "tau": float(tau),
        "best_pose": None if best is None else best.tolist(),
        "best_residual": best_residual,
        "converged": [s.converged for s in searches],
        "search_residuals": [s.best_residual for s in searches],
        "evaluations": inversion.trace.evaluations,
        "attempts": len(arrays["residuals"]),
        "invalid_attempts": inversion.trace.invalid_attempts,
        "nuisance": inversion.nuisance,
    }
    if best is None or any(s.best_pose is None for s in searches):
        return {"status": "unsupported_geometry", **diagnostics}
    restart = diameter(np.array([s.best_pose[BLOCK] for s in searches]))
    diagnostics["restart_distance"] = restart
    within = np.isfinite(arrays["residuals"]) & (arrays["residuals"] <= best_residual + tau)
    spread = diameter(arrays["poses"][within][:, BLOCK])
    diagnostics.update({"near_optimal_count": int(within.sum()), "near_optimal_diameter": spread})
    if not all(s.converged for s in searches) or restart > config.restart_tolerance:
        return {"status": "optimizer_unresolved", **diagnostics}
    if best_residual > tau:
        return {"status": "residual_reject", **diagnostics}
    if spread > config.near_optimal_diameter:
        return {"status": "ambiguous", **diagnostics}
    return {"status": "accepted", "block_center": best[BLOCK].tolist(), **diagnostics}


def _classify_v2(inversion: Inversion, tau: float, ambiguity_tau: float | None) -> dict[str, Any]:
    config = inversion.config
    reported, reported_residual = inversion.reported
    searches = inversion.searches
    arrays = inversion.trace.arrays()
    resolution_payload = inversion.resolution or {}
    band = resolution_payload.get("tau_amb") if ambiguity_tau is None else ambiguity_tau
    diagnostics: dict[str, Any] = {
        "tau": float(tau),
        "tau_amb": band,
        "version": config.version,
        "best_pose": None if reported is None else reported.tolist(),
        "best_residual": reported_residual,
        "converged": [s.converged for s in searches],
        "search_residuals": [s.best_residual for s in searches],
        "evaluations": inversion.trace.evaluations,
        "attempts": len(arrays["residuals"]),
        "invalid_attempts": inversion.trace.invalid_attempts,
        "nuisance": inversion.nuisance,
        "reason": "",
    }
    if reported is None or any(s.best_pose is None for s in searches):
        return {"status": "unsupported_geometry", **diagnostics}
    restart = diameter(np.array([s.best_pose[BLOCK] for s in searches]))
    diagnostics["restart_distance"] = restart
    if not all(s.converged for s in searches) or restart > config.restart_tolerance:
        return {"status": "optimizer_unresolved", **diagnostics, "reason": "restart"}
    unfinished = [
        r
        for r in inversion.nuisance
        if r["status"] == "budget" or (r["status"] == "fitted" and not r["converged"])
    ]
    if unfinished:
        return {"status": "optimizer_unresolved", **diagnostics, "reason": "nuisance_incomplete"}
    finite = np.isfinite(arrays["residuals"])
    if finite.any():
        index = int(np.argmin(np.where(finite, arrays["residuals"], np.inf)))
        best_any = arrays["poses"][index]
        far = np.linalg.norm(best_any[BLOCK] - reported[BLOCK]) > config.restart_tolerance
        if arrays["residuals"][index] < reported_residual and far:
            diagnostics["unsupported_better_pose"] = best_any.tolist()
            return {
                "status": "optimizer_unresolved",
                **diagnostics,
                "reason": "nuisance_better_basin",
            }
    if reported_residual > tau:
        return {"status": "residual_reject", **diagnostics}
    if band is None:
        return {"status": "ambiguous", **diagnostics, "reason": "resolution_unavailable"}
    within = finite & (arrays["residuals"] <= reported_residual + band)
    spread = diameter(np.vstack([arrays["poses"][within][:, BLOCK], reported[BLOCK][None]]))
    diagnostics.update({"near_optimal_count": int(within.sum()), "near_optimal_diameter": spread})
    if spread > config.near_optimal_diameter:
        return {"status": "ambiguous", **diagnostics}
    return {"status": "accepted", "block_center": reported[BLOCK].tolist(), **diagnostics}


class CalibrationFailure(RuntimeError):
    """A calibration query had no valid finite candidate; assessment stays closed."""


def calibrate_tau(family_maxima: Mapping[str, Sequence[float | None]], epsilon: float) -> float:
    """D032: larger scene-specific 95th percentile of per-family maximum best residuals."""
    floor = max(float(epsilon), 1e-12)
    percentiles = []
    for scene, values in sorted(family_maxima.items()):
        values = [math.inf if v is None else float(v) for v in values]
        if not values or not all(math.isfinite(v) for v in values):
            raise CalibrationFailure(f"Scene {scene} has a query without a finite candidate")
        percentiles.append(float(np.percentile(values, 95)))
    if not percentiles:
        raise CalibrationFailure("No calibration families")
    return max(max(percentiles), floor)
