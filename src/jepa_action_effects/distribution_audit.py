"""DIST-001: action support, simulator identity and archive/simulator error gap.

Stages (run in order on the machine holding the extracted archive and checkpoint):
    python -m jepa_action_effects.distribution_audit keys
    python -m jepa_action_effects.distribution_audit actions
    python -m jepa_action_effects.distribution_audit identity
    python -m jepa_action_effects.distribution_audit gap
    python -m jepa_action_effects.distribution_audit report

Nothing here selects families or chooses thresholds from model errors. The identity
and gap episodes are drawn from a fixed seed before any rendering or inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.next_common import (
    ARCHIVE_H5,
    FRESH,
    MANIFESTS,
    PILOT,
    Stopwatch,
    collection_manifest,
    encode_frames,
    load_family,
    load_model,
    runtime_info,
    source_hashes,
    write_json,
)
from jepa_action_effects.next_metrics import archive_blocks, block_features, support_summary

ROOT = Path("outputs/dist001")
SEED = 20261001
IDENTITY_EPISODES = 20
IDENTITY_STEPS = 40
GAP_WINDOWS = 200
RENDER_MISMATCH_LIMIT = 1.0  # mean absolute uint8 difference on reset frames


def _h5(path: Path = ARCHIVE_H5):
    import h5py

    if not path.exists():
        raise FileNotFoundError(f"Extracted archive not found: {path}")
    return h5py.File(path, "r")


def _find(h5, names: tuple[str, ...], width: int | None = None) -> str | None:
    for name in names:
        if name in h5 and (width is None or h5[name].shape[1:] == (width,)):
            return name
    return None


def keys(root: Path = ROOT, path: Path = ARCHIVE_H5) -> dict:
    """List datasets so later stages rely on recorded, not assumed, field names."""
    import h5py

    with _h5(path) as h5:
        datasets = {
            name: {"shape": list(item.shape), "dtype": str(item.dtype)}
            for name, item in h5.items()
            if isinstance(item, h5py.Dataset)
        }
        attributes = {key: str(value) for key, value in h5.attrs.items()}
        state = _find(h5, ("state", "observation.state", "states"), 7)
        pixels = _find(h5, ("pixels", "observation.image", "image"))
        goal = _find(h5, ("goal_state", "goal"), 7)
    result = {
        "datasets": datasets,
        "attributes": attributes,
        "state_key": state,
        "pixels_key": pixels,
        "goal_key": goal,
        "archive_sha256_checked_by": "action_stats.py (MODEL-001)",
    }
    write_json(root / "keys.json", result)
    return result


def _episodes(h5) -> tuple[np.ndarray, np.ndarray]:
    return h5["ep_offset"][:].astype(np.int64), h5["ep_len"][:].astype(np.int64)


def _study_blocks() -> dict[str, np.ndarray]:
    """Candidate 5-control blocks of every pilot/READOUT family, by cohort."""
    blocks: dict[str, list[np.ndarray]] = {"challenge": [], "ordinary": []}
    for root in (PILOT, FRESH):
        manifest = collection_manifest(root)
        for family_id, record in manifest["family_files"].items():
            family = load_family(root / record["path"])
            cohort = "challenge" if family_id.startswith("challenge") else "ordinary"
            blocks[cohort].append(family["candidate_controls"].reshape(-1, 5, 2))
    return {cohort: np.concatenate(values) for cohort, values in blocks.items()}


def actions(root: Path = ROOT, path: Path = ARCHIVE_H5) -> dict:
    with _h5(path) as h5:
        offsets, lengths = _episodes(h5)
        archive = archive_blocks(h5["action"][:], offsets, lengths)
    by_cohort = _study_blocks()
    study = np.concatenate(list(by_cohort.values()))
    result = {
        "archive_block_definition": "non-overlapping 5-control blocks from each episode start",
        "all_study": support_summary(archive, study, SEED),
        "by_cohort": {
            cohort: {
                **support_summary(archive, blocks, SEED),
                "magnitude_mean": float(np.mean(block_features(blocks)["magnitude"])),
            }
            for cohort, blocks in by_cohort.items()
        },
    }
    write_json(root / "actions.json", result)
    return result


def _selected_episodes(offsets: np.ndarray, lengths: np.ndarray, minimum: int, count: int):
    eligible = np.flatnonzero(lengths >= minimum)
    rng = np.random.default_rng(SEED)
    return sorted(int(value) for value in rng.choice(eligible, size=count, replace=False))


def _replay(state0: np.ndarray, controls: np.ndarray) -> dict:
    """Reset the pinned simulator to an archive start state and replay its controls.

    Reset first applies `goal_state`, then `state`, without re-indexing collision shapes;
    a goal pose that overlaps a wall (such as FAR_GOAL) kicks the block by ~2 units.
    Using the start state as its own goal makes the reset reproduce the requested pose.
    The drawn goal overlay comes from the variation space, not from `goal_state`, and
    termination flags are ignored here.
    """
    from jepa_action_effects.simulator_replay import InstrumentedPushT

    env = InstrumentedPushT()
    try:
        start = np.asarray(state0, dtype=np.float64).tolist()
        options = {"state": start, "goal_state": start}
        observation, _ = env.reset(seed=0, options=options)
        frames = [env.render().copy()]
        states = [observation["state"].copy()]
        for control in controls:
            observation, *_ = env.step(np.asarray(control, dtype=np.float32))
            frames.append(env.render().copy())
            states.append(observation["state"].copy())
    finally:
        env.close()
    return {"frames": np.stack(frames), "states": np.stack(states)}


def identity(root: Path = ROOT, path: Path = ARCHIVE_H5) -> dict:
    found = sealed_keys(root)
    if found["state_key"] is None or found["pixels_key"] is None:
        result = {
            "status": "unavailable",
            "reason": "archive lacks a 7-D state or pixel dataset; compare static scene manually",
            "keys": found["datasets"],
        }
        write_json(root / "identity.json", result)
        return result
    rows = []
    examples = {}
    with _h5(path) as h5:
        offsets, lengths = _episodes(h5)
        chosen = _selected_episodes(offsets, lengths, IDENTITY_STEPS + 1, IDENTITY_EPISODES)
        for episode in chosen:
            start = int(offsets[episode])
            span = slice(start, start + IDENTITY_STEPS + 1)
            archive_frames = h5[found["pixels_key"]][span]
            archive_states = h5[found["state_key"]][span].astype(np.float64)
            controls = h5["action"][start : start + IDENTITY_STEPS]
            simulated = _replay(archive_states[0], controls)
            if archive_frames.shape != simulated["frames"].shape:
                raise ValueError(
                    f"Archive frame layout {archive_frames.shape} differs from simulator "
                    f"{simulated['frames'].shape}; record it in keys.json before comparing"
                )
            difference = np.abs(
                simulated["frames"].astype(np.int16) - archive_frames.astype(np.int16)
            )
            state_error = np.linalg.norm(simulated["states"][:, :4] - archive_states[:, :4], axis=1)
            rows.append(
                {
                    "episode": episode,
                    "reset_frame_mean_abs": float(difference[0].mean()),
                    "reset_frame_identical": bool(difference[0].max() == 0),
                    "frame_mean_abs_by_step": difference.mean(axis=(1, 2, 3)).tolist(),
                    "identical_fraction": float(np.mean(difference.max(axis=(1, 2, 3)) == 0)),
                    "state_position_error_by_step": state_error.tolist(),
                }
            )
            if len(examples) < 3:
                examples[f"episode_{episode}"] = np.stack(
                    (
                        archive_frames[0],
                        simulated["frames"][0],
                        archive_frames[-1],
                        simulated["frames"][-1],
                    )
                )
    np.savez_compressed(root / "identity-examples.npz", **examples)
    reset_error = float(np.mean([row["reset_frame_mean_abs"] for row in rows]))
    result = {
        "status": "complete",
        "episodes": chosen,
        "steps": IDENTITY_STEPS,
        "reset_goal": "start state reused as goal (kick-free reset; overlay uses env default)",
        "reset_frame_mean_abs": reset_error,
        "reset_frame_identical_fraction": float(
            np.mean([r["reset_frame_identical"] for r in rows])
        ),
        "final_step_mean_abs": float(np.mean([r["frame_mean_abs_by_step"][-1] for r in rows])),
        "final_step_position_error_mean": float(
            np.mean([r["state_position_error_by_step"][-1] for r in rows])
        ),
        "rendering_mismatch_flag": reset_error > RENDER_MISMATCH_LIMIT,
        "rendering_mismatch_limit": RENDER_MISMATCH_LIMIT,
        "rows": rows,
    }
    write_json(root / "identity.json", result)
    return result


def sealed_keys(root: Path) -> dict:
    path = root / "keys.json"
    if not path.exists():
        raise FileNotFoundError("Run the `keys` stage first")
    return json.loads(path.read_text())


def _latent_errors(model, scaler, frames: np.ndarray, controls: np.ndarray) -> dict[str, float]:
    """frames: 5 frames at t-10,t-5,t,t+5,t+10; controls: 20 controls from t-10."""
    from jepa_action_effects.interface_audit import preprocess_pixels
    from jepa_action_effects.model_inference import predict_future

    predicted = predict_future(
        model,
        preprocess_pixels(frames[:3]).unsqueeze(0),
        controls[None, :10],
        controls[None, None, 10:],
        scaler,
    ).numpy()[0, 0]
    real = encode_frames(model, frames)["projected"]
    targets = real[3:]
    return {
        "latent_mse": float(np.mean((predicted[-1] - targets[-1]) ** 2)),
        "persistence_mse": float(np.mean((real[2] - targets[-1]) ** 2)),
    }


def gap(root: Path = ROOT, path: Path = ARCHIVE_H5) -> dict:
    found = sealed_keys(root)
    if found["pixels_key"] is None:
        raise ValueError("Archive has no recognised pixel dataset; see keys.json")
    model, scaler, _ = load_model()
    rows = []
    skipped = {"out_of_bounds_controls": 0}
    rng = np.random.default_rng(SEED + 1)
    with _h5(path) as h5:
        offsets, lengths = _episodes(h5)
        eligible = np.flatnonzero(lengths >= 21)
        episodes = rng.choice(eligible, size=GAP_WINDOWS, replace=False)
        for episode in episodes:
            start = int(offsets[episode])
            t = int(rng.integers(10, int(lengths[episode]) - 10))
            indices = [start + t + k for k in (-10, -5, 0, 5, 10)]
            controls = h5["action"][start + t - 10 : start + t + 10].astype(np.float32)
            if np.any(np.abs(controls) > 1) or not np.isfinite(controls).all():
                skipped["out_of_bounds_controls"] += 1
                continue
            archive_frames = np.stack([h5[found["pixels_key"]][index] for index in indices])
            row = {
                "episode": int(episode),
                "t": t,
                "archive": _latent_errors(model, scaler, archive_frames, controls),
            }
            if found["state_key"] is not None:
                state0 = h5[found["state_key"]][start].astype(np.float64)
                replay = _replay(state0, h5["action"][start : start + t + 10])
                sim_frames = replay["frames"][[t - 10, t - 5, t, t + 5, t + 10]]
                row["simulator"] = _latent_errors(model, scaler, sim_frames, controls)
                row["frame_mean_abs"] = float(
                    np.abs(sim_frames.astype(np.int16) - archive_frames.astype(np.int16)).mean()
                )
            rows.append(row)

    def ratio(source: str) -> dict[str, float] | None:
        cells = [row[source] for row in rows if source in row]
        if not cells:
            return None
        latent = float(np.mean([cell["latent_mse"] for cell in cells]))
        persistence = float(np.mean([cell["persistence_mse"] for cell in cells]))
        return {
            "windows": len(cells),
            "latent_mse": latent,
            "persistence_mse": persistence,
            "ratio": latent / persistence,
        }

    result = {
        "windows_requested": GAP_WINDOWS,
        "skipped": skipped,
        "archive": ratio("archive"),
        "simulator": ratio("simulator"),
        "rows": rows,
    }
    write_json(root / "gap.json", result)
    return result


def report(root: Path = ROOT) -> dict:
    def load(name: str) -> Any:
        path = root / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def without_rows(value: dict | None) -> dict | None:
        return None if value is None else {k: v for k, v in value.items() if k != "rows"}

    identity_result = load("identity")
    summary: dict[str, Any] = {
        "task": "DIST-001",
        "runtime": runtime_info(),
        "sources": source_hashes(
            {"runner": Path(__file__), "metrics": Path(__file__).with_name("next_metrics.py")}
        ),
        "keys": load("keys"),
        "actions": load("actions"),
        "identity": without_rows(identity_result),
        "gap": without_rows(load("gap")),
        "stop_rule": (
            "PAUSE Phase 3: rendering mismatch"
            if identity_result and identity_result.get("rendering_mismatch_flag")
            else "no rendering mismatch flagged"
        ),
    }
    write_json(root / "report.json", summary)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFESTS / "dist001-runtime.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("keys", "actions", "identity", "gap", "report"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE_H5)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    stages = {"keys": keys, "actions": actions, "identity": identity, "gap": gap}
    watch = Stopwatch(ROOT)
    if args.stage == "report":
        result = watch.run("report", report, ROOT)
    else:
        result = watch.run(args.stage, stages[args.stage], ROOT, args.archive)
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
