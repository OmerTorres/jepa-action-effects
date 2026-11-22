"""Joint-pose renderer checks (simulator only; no model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_action_effects.pose_render import SceneContext, SceneRenderer, joint_geometry, set_pose

DEFAULT = SceneContext(seed=0, reset_options={}, warmup_controls=(), scene="test")


@pytest.fixture()
def renderer():
    scene = SceneRenderer(DEFAULT)
    yield scene
    scene.close()


def _clearance(scene: SceneRenderer, pose) -> float:
    set_pose(scene.env, np.asarray(pose, float))
    try:
        return joint_geometry(scene.env, 1.0, agent_walls=False)[2]
    finally:
        scene.restore()


def _agent_at_overlap(scene: SceneRenderer, overlap: float) -> np.ndarray:
    """Agent on the block's +x side, penetrating it by `overlap` units (bisection)."""
    low, high = 256.0, 480.0  # inside the block ... clear of it
    for _ in range(80):
        middle = (low + high) / 2
        value = _clearance(scene, [middle, 256.0, 256.0, 256.0, 0.0])
        low, high = (middle, high) if value < -overlap else (low, middle)
    return np.array([high, 256.0, 256.0, 256.0, 0.0])


def test_render_restores_rgb_and_never_steps_physics(renderer) -> None:
    steps = len(renderer.env.physics_steps)
    poses = np.array([[100.0, 100.0, 300.0, 300.0, 0.3], [400.0, 380.0, 200.0, 150.0, 2.0]])
    frames = renderer.render(poses)
    assert frames.shape == (2, 224, 224, 3) and frames.dtype == np.uint8
    assert not np.array_equal(frames[0], frames[1])
    assert np.array_equal(renderer.env.render(), renderer.baseline_rgb)
    assert len(renderer.env.physics_steps) == steps
    again = renderer.render(poses[:1])
    assert np.array_equal(again[0], frames[0])  # deterministic


def test_contact_overlap_tolerance(renderer) -> None:
    valid, reasons, _ = renderer.check(
        np.stack([_agent_at_overlap(renderer, 0.5), _agent_at_overlap(renderer, 1.5)])
    )
    assert valid.tolist() == [True, False]
    assert reasons[1] == "agent_block_penetration"


def test_agent_may_pass_walls_but_block_may_not(renderer) -> None:
    agent_in_wall = [3.0, 256.0, 256.0, 256.0, 0.0]
    agent_off_canvas = [-16.0, 256.0, 256.0, 256.0, 0.0]
    block_in_wall = [256.0, 400.0, 10.0, 256.0, 0.0]
    valid, reasons, _ = renderer.check(np.array([agent_in_wall, agent_off_canvas, block_in_wall]))
    assert valid.tolist() == [True, True, False]
    assert reasons[2] in {"block_wall_penetration", "block_outside_workspace"}
    strict = SceneRenderer(DEFAULT, agent_walls=True)
    try:
        assert strict.check(np.array([agent_in_wall]))[0].tolist() == [False]
    finally:
        strict.close()


def test_scene_context_reads_no_future_fields() -> None:
    class Guard(dict):
        def __getitem__(self, key):
            if key.startswith("future") or key in {"candidate_controls", "block_trajectories"}:
                raise AssertionError(f"SceneContext read label field {key}")
            return super().__getitem__(key)

    family = Guard(
        metadata={"seed": 3, "reset_options": {"state": [1, 2, 3, 4, 5, 0, 0]}},
        warmup_controls=np.zeros((10, 2)),
        future_states=None,
    )
    context = SceneContext.from_family(family, "challenge")
    assert context.seed == 3 and len(context.warmup_controls) == 10


GALLERY = Path("outputs/gallery001/config.json")
FRESH = Path("outputs/readout001/collection")


@pytest.mark.skipif(not GALLERY.exists(), reason="GALLERY-001 artifacts are local-only")
def test_replayed_start_matches_saved_frame() -> None:
    from jepa_action_effects.pilot_collection import load_family

    identifier = json.loads(GALLERY.read_text())["ids"][0].split(":", 1)[1]
    manifest = json.loads((FRESH / "collection-manifest.json").read_text())
    family = load_family(FRESH / manifest["family_files"][identifier]["path"])
    scene = SceneRenderer(SceneContext.from_family(family, "challenge"))
    try:
        assert np.array_equal(scene.baseline_rgb, family["history_frames"][-1])
    finally:
        scene.close()
