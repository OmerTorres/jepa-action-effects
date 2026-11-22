"""TRAIN-001 cumulative budget ledger: accumulation, crashes, unit refusal, locking."""

from __future__ import annotations

import pytest

from jepa_action_effects.train001_budget import Budget, BudgetLocked, BudgetStop


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def budget(root, clock, limit=100.0):
    return Budget(
        root, limit, shutdown_allowance=5.0, heartbeat_interval=2.0, clock=clock, watchdog=False
    )


def test_ledger_accumulates_across_run_identities(tmp_path):
    clock = Clock()
    first = budget(tmp_path, clock)
    first.start("collect", "outputs/train001")
    clock.now += 30
    first.finish("completed")
    second = budget(tmp_path, clock)
    second.start("train", "outputs/train001-v2")  # a new run identity shares the ledger
    clock.now += 10
    assert second.used() == pytest.approx(40.0)
    second.finish("completed")
    assert budget(tmp_path, clock).used() == pytest.approx(40.0)


def test_crashed_session_is_charged_to_last_heartbeat(tmp_path):
    clock = Clock()
    crashed = budget(tmp_path, clock)
    crashed.start("train", "outputs/train001")
    clock.now += 3
    crashed.heartbeat()
    clock.now += 50  # killed without finishing; nothing persisted after the heartbeat
    crashed._lock_handle.close()
    crashed._lock_handle = None
    recovered = budget(tmp_path, clock)
    recovered.start("train", "outputs/train001")
    ledger = recovered.load()
    assert ledger["sessions"][0]["status"] == "crashed"
    assert ledger["sessions"][0]["ended"] == pytest.approx(1003.0 + 2.0)
    assert recovered.used() == pytest.approx(5.0)
    recovered.finish("completed")


def test_unit_that_cannot_finish_is_refused(tmp_path):
    clock = Clock()
    ledger = budget(tmp_path, clock)
    ledger.start("train", "outputs/train001")
    clock.now += 80
    ledger.unit("update", 5.0)  # 80 + 5 + 5 <= 100
    ledger.unit("update", 15.0)  # exactly at the cap is allowed
    with pytest.raises(BudgetStop):
        ledger.unit("update", 16.0)
    ledger.record_unit("update", 16.0)
    with pytest.raises(BudgetStop):
        ledger.unit("update")  # uses the running mean estimate
    ledger.finish("budget_stop")
    assert ledger.load()["sessions"][0]["status"] == "budget_stop"


def test_exhausted_budget_refuses_to_start(tmp_path):
    clock = Clock()
    first = budget(tmp_path, clock)
    first.start("train", "outputs/train001")
    clock.now += 97
    first.finish("completed")
    with pytest.raises(BudgetStop):
        budget(tmp_path, clock).start("confirm", "outputs/train001")


def test_concurrent_stages_are_locked_out(tmp_path):
    clock = Clock()
    first = budget(tmp_path, clock)
    first.start("train", "outputs/train001")
    with pytest.raises(BudgetLocked):
        budget(tmp_path, clock).start("confirm", "outputs/train001")
    first.finish("completed")
    second = budget(tmp_path, clock)
    second.start("confirm", "outputs/train001")
    second.finish("completed")


def test_resume_after_budget_stop_keeps_accounting(tmp_path):
    clock = Clock()
    first = budget(tmp_path, clock)
    first.start("train", "outputs/train001")
    clock.now += 20
    first.finish("budget_stop")
    second = budget(tmp_path, clock)
    second.start("train", "outputs/train001")
    clock.now += 15
    second.finish("completed")
    statuses = [s["status"] for s in second.load()["sessions"]]
    assert statuses == ["budget_stop", "completed"]
    assert budget(tmp_path, clock).used() == pytest.approx(35.0)
