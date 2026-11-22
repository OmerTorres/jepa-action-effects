"""TRAIN-001 cumulative machine-time budget (D028).

One ledger, outside every run directory, accumulates the wall-clock time of every
experimental stage invocation, including failed, interrupted and killed ones. A new
run identity therefore never resets the 24-hour cap.

- Each invocation is a *session* with a persisted start time and heartbeat.
- A session that never recorded its end (killed process) is charged up to its last
  heartbeat plus one heartbeat interval when the next session starts.
- Before each bounded work unit a stage calls `unit(estimate)`; if the unit could not
  finish inside the cap (with a shutdown allowance) it raises `BudgetStop`, and the
  stage exits resumably.
- A watchdog interrupts the main thread at the hard deadline (a hung unit).
- An exclusive lock prevents concurrent experimental stages from racing the ledger.
"""

from __future__ import annotations

import _thread
import fcntl
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

LIMIT_SECONDS = 24 * 3600.0
SHUTDOWN_ALLOWANCE = 300.0
HEARTBEAT_INTERVAL = 60.0


class BudgetStop(RuntimeError):
    """The next work unit cannot finish inside the remaining budget."""


class BudgetLocked(RuntimeError):
    """Another experimental stage holds the budget lock."""


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def session_seconds(session: dict[str, Any], now: float | None = None) -> float:
    """Charged seconds for one session; open sessions are charged up to `now`."""
    end = session.get("ended")
    if end is None:
        end = now if now is not None else session["last_heartbeat"]
    return max(0.0, float(end) - float(session["started"]))


class Budget:
    def __init__(
        self,
        root: Path,
        limit_seconds: float = LIMIT_SECONDS,
        shutdown_allowance: float = SHUTDOWN_ALLOWANCE,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        clock: Callable[[], float] = time.time,
        watchdog: bool = True,
    ):
        self.root = root
        self.path = root / "ledger.json"
        self.limit = float(limit_seconds)
        self.shutdown = float(shutdown_allowance)
        self.interval = float(heartbeat_interval)
        self.clock = clock
        self.use_watchdog = watchdog
        self.session: dict[str, Any] | None = None
        self._lock_handle = None
        self._timer: threading.Timer | None = None
        self._unit_means: dict[str, tuple[float, int]] = {}

    # ------------------------------------------------------------------ ledger
    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"limit_seconds": self.limit, "sessions": []}

    def _save(self, ledger: dict[str, Any]) -> None:
        _atomic_write(self.path, ledger)

    def used(self) -> float:
        """Charged seconds so far, including the open session up to now."""
        ledger = self.load()
        now = self.clock()
        total = 0.0
        for session in ledger["sessions"]:
            active = self.session is not None and session["id"] == self.session["id"]
            total += session_seconds(session, now if active else None)
        return total

    def remaining(self) -> float:
        return self.limit - self.used()

    def _recover(self, ledger: dict[str, Any]) -> None:
        for session in ledger["sessions"]:
            if session.get("ended") is None:
                session["ended"] = float(session["last_heartbeat"]) + self.interval
                session["status"] = "crashed"

    # ------------------------------------------------------------------ sessions
    def start(self, stage: str, run_root: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root / "ledger.lock").open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise BudgetLocked("Another TRAIN-001 experimental stage is running") from error
        self._lock_handle = handle
        ledger = self.load()
        self._recover(ledger)
        used = sum(session_seconds(session) for session in ledger["sessions"])
        now = self.clock()
        self.session = {
            "id": f"{stage}-{len(ledger['sessions']):04d}",
            "stage": stage,
            "run_root": run_root,
            "started": now,
            "last_heartbeat": now,
            "ended": None,
            "status": "running",
        }
        ledger["sessions"].append(self.session)
        self._save(ledger)
        if used + self.shutdown >= self.limit:
            self.finish("budget_stop")
            raise BudgetStop(f"Budget exhausted: {used / 3600:.2f} h used")
        if self.use_watchdog:
            deadline = self.limit - used + self.shutdown
            self._timer = threading.Timer(deadline, _thread.interrupt_main)
            self._timer.daemon = True
            self._timer.start()

    def heartbeat(self, force: bool = False) -> None:
        if self.session is None:
            return
        now = self.clock()
        if not force and now - self.session["last_heartbeat"] < self.interval:
            return
        self.session["last_heartbeat"] = now
        self._update_session()

    def _update_session(self) -> None:
        ledger = self.load()
        for index, session in enumerate(ledger["sessions"]):
            if session["id"] == self.session["id"]:
                ledger["sessions"][index] = self.session
        self._save(ledger)

    def finish(self, status: str) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.session is not None:
            now = self.clock()
            self.session["last_heartbeat"] = now
            self.session["ended"] = now
            self.session["status"] = status
            self._update_session()
            self.session = None
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    # ------------------------------------------------------------------ units
    def unit(self, kind: str, estimate: float | None = None) -> None:
        """Refuse to start a unit that could not finish inside the cap."""
        if estimate is None:
            mean, _ = self._unit_means.get(kind, (0.0, 0))
            estimate = mean
        if self.used() + estimate + self.shutdown > self.limit:
            raise BudgetStop(f"Next {kind} unit ({estimate:.1f} s) would exceed the budget")
        self.heartbeat()

    def record_unit(self, kind: str, seconds: float) -> None:
        mean, count = self._unit_means.get(kind, (0.0, 0))
        self._unit_means[kind] = ((mean * count + seconds) / (count + 1), count + 1)

    def timed(self, kind: str, function: Callable, *args, **kwargs):
        """Run one bounded unit with a budget check and duration accounting."""
        self.unit(kind)
        started = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            self.record_unit(kind, time.monotonic() - started)
            self.heartbeat()


class NullBudget(Budget):
    """No-op budget for tests and pure computations."""

    def __init__(self):
        super().__init__(Path("."), watchdog=False)

    def unit(self, kind: str, estimate: float | None = None) -> None:
        return None

    def heartbeat(self, force: bool = False) -> None:
        return None

    def used(self) -> float:
        return 0.0
