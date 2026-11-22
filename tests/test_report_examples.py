"""D039 report examples: outcome-only selection, exact T geometry, hash-checked inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jepa_action_effects import report_examples as ex
from jepa_action_effects.pilot_collection import sha256_file


def test_median_closest_uses_outcomes_and_breaks_ties_by_lowest_id():
    assert ex.median_closest({"f3": 10.0, "f1": 30.0, "f2": 21.0}) == "f2"  # median 21
    # Median 17.75: f1 and f9 are equally close (2.75), so the lower id wins.
    assert ex.median_closest({"f2": 25.0, "f1": 15.0, "f9": 20.5, "f0": 0.0}) == "f1"
    assert ex.median_closest({"b": 15.0, "a": 25.0}) == "a"
    with pytest.raises(ex.ExampleMismatch):
        ex.median_closest({})


def test_tee_polygons_follow_the_simulator_geometry():
    bar, stem = ex.tee_polygons(0.0, 0.0, 0.0)
    assert bar.tolist() == [list(v) for v in ex.BAR]
    assert stem.tolist() == [list(v) for v in ex.STEM]
    # A quarter turn maps the bar end (60, 0) to (0, 60): y points down, as rendered.
    bar, _ = ex.tee_polygons(100.0, 200.0, np.pi / 2)
    assert bar[2] == pytest.approx([100.0, 260.0])


def _family_file(tmp_path: Path, vertices=None) -> tuple[Path, dict]:
    metadata = {"candidate_ids": ["m0.30", "m0.60"]}
    if vertices is not None:
        metadata["actual_block_vertices"] = vertices
    history = np.zeros((3, 5))
    history[2] = [50.0, 60.0, 250.0, 240.0, 0.4]
    future = np.zeros((2, 2, 5))
    future[:, -1, 2:4] = [[250.0, 252.0], [256.0, 248.0]]
    path = tmp_path / "families" / "f.npz"
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        history_states=history,
        future_states=future,
    )
    return path, {"id": "f", "relative_path": "families/f.npz", "file_sha256": sha256_file(path)}


def test_true_displacement_is_read_from_saved_states(tmp_path):
    _, entry = _family_file(tmp_path)
    family = ex._verified_family(tmp_path, entry)
    assert ex._true_displacement(family, "m0.30") == pytest.approx(12.0)
    assert ex._true_displacement(family, "m0.60") == pytest.approx(10.0)


def test_changed_family_file_is_refused(tmp_path):
    _, entry = _family_file(tmp_path)
    with pytest.raises(ex.ExampleMismatch, match="changed"):
        ex._verified_family(tmp_path, {**entry, "file_sha256": "0" * 64})


def test_vertex_check_accepts_exact_corners_and_rejects_others(tmp_path):
    corners = np.concatenate(ex.tee_polygons(250.0, 240.0, 0.4))[::-1].tolist()  # any order
    _, entry = _family_file(tmp_path, corners)
    assert ex._vertex_error(ex._verified_family(tmp_path, entry)) < 1e-9
    other = tmp_path / "other"
    other.mkdir()
    _, entry = _family_file(other, (np.asarray(corners) + 2.0).tolist())
    with pytest.raises(ex.ExampleMismatch, match="T geometry"):
        ex._vertex_error(ex._verified_family(other, entry))
    third = tmp_path / "third"
    third.mkdir()
    _, entry = _family_file(third)
    assert ex._vertex_error(ex._verified_family(third, entry)) is None


@pytest.mark.skipif(not ex.OUT.exists(), reason="tracked report-examples manifest")
def test_tracked_manifest_frames_match_their_recorded_hashes():
    payload = json.loads(ex.OUT.read_text())
    for key in ("support001", "train002"):
        example = payload[key]
        assert example["chosen_true_displacement"] > 0
        for frame in example["frames"].values():
            assert sha256_file(Path(frame["path"])) == frame["sha256"]
    histogram = payload["demo_commands"]
    assert sum(histogram["counts"]) + histogram["above_last_edge"] == histogram["windows"]
