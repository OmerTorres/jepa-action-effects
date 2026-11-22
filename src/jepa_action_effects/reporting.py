"""REPORT-001: numbers, claims ledger and figures from tracked manifests only.

    python -m jepa_action_effects.reporting            # writes outputs/report001/

Every number in the report is a LaTeX macro produced here from a manifest field; the
same registry writes `docs/REPORT-001_CLAIMS.md`. A missing field raises instead of
inserting a placeholder. No model, simulator or sealed file is touched.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa_action_effects.pilot_collection import sha256_file

MANIFESTS = Path("docs/manifests")
OUT = Path("outputs/report001")
CLAIMS = Path("docs/REPORT-001_CLAIMS.md")


class MissingField(KeyError):
    """A registry path does not exist in its manifest."""


@dataclass(frozen=True)
class Number:
    macro: str  # LaTeX macro name, letters only
    manifest: str  # file under docs/manifests
    path: tuple[str, ...]  # keys into the JSON
    digits: int = 2
    kind: str = "value"  # "value" | "ci" (estimate + ci/ci95) | "count" | "percent"
    claim: str = ""
    status: str = "confirmed"  # confirmed | exploratory | descriptive | not established


def _get(payload: Any, path: tuple[str, ...], where: str) -> Any:
    node = payload
    for key in path:
        if isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
            continue
        if not isinstance(node, dict) or key not in node:
            raise MissingField(f"{where}: missing {'/'.join(path)} (at {key!r})")
        node = node[key]
    return node


def _interval(leaf: dict[str, Any]) -> tuple[float, list[float] | None]:
    interval = leaf.get("ci95", leaf.get("ci"))
    return float(leaf["estimate"]), interval


def render(number: Number, payload: Any) -> str:
    leaf = _get(payload, number.path, number.manifest)
    d = number.digits
    if number.kind == "value":
        return f"{float(leaf):.{d}f}"
    if number.kind == "count":
        return f"{int(leaf):,}".replace(",", "{,}")
    if number.kind == "percent":
        return f"{100 * float(leaf):.{d}f}\\%"
    estimate, interval = _interval(leaf)
    if interval is None:
        return f"{estimate:.{d}f}"
    return f"{estimate:.{d}f}\\,[{interval[0]:.{d}f},\\,{interval[1]:.{d}f}]"


def _phys(*keys: str) -> tuple[str, ...]:
    return ("confirm", "reserved", "summary", *keys)


def _shift(key: str) -> tuple[str, ...]:
    return ("confirm", "summary", key)


def _robust(scene: str, instrument: str, stratum: str = "all") -> tuple[str, ...]:
    return ("verdicts", "contrasts", f"{scene}:energy_ratio:{stratum}", instrument)


TRAIN002_ARMS = (
    ("frozen", "Frozen"),
    ("archive_only", "ArchOnly"),
    ("archive_weighted", "ArchWeighted"),
    ("balanced", "Balanced"),
    ("dose_omitted", "Omitted"),
)

REGISTRY: tuple[Number, ...] = (
    # PHYS-002 (150 reserved families; 100 challenge)
    Number(
        "physSlopePred",
        "phys002-runtime.json",
        _phys("challenge", "slope_pred"),
        3,
        "ci",
        "Predicted projection slope on fresh challenge pushes",
    ),
    Number(
        "physSlopeReal",
        "phys002-runtime.json",
        _phys("challenge", "slope_real"),
        3,
        "ci",
        "Real-frame readout projection slope on the same families",
    ),
    Number(
        "physSlopeOrdinary",
        "phys002-runtime.json",
        _phys("ordinary", "slope_pred"),
        3,
        "ci",
        "Predicted projection slope on ordinary families",
    ),
    # SHIFT-002 (240 reserved families)
    Number(
        "shiftRchThirty",
        "shift002-runtime.json",
        _shift("R_challenge_m0.30_constant"),
        2,
        "ci",
        "Pred/real slope ratio, challenge push 0.30",
    ),
    Number(
        "shiftRchFortyfive",
        "shift002-runtime.json",
        _shift("R_challenge_m0.45_constant"),
        2,
        "ci",
        "Pred/real slope ratio, challenge push 0.45",
    ),
    Number(
        "shiftRchSixty",
        "shift002-runtime.json",
        _shift("R_challenge_m0.60_constant"),
        2,
        "ci",
        "Pred/real slope ratio, challenge push 0.60",
    ),
    Number(
        "shiftRarOne",
        "shift002-runtime.json",
        _shift("R_archive_x1.0"),
        2,
        "ci",
        "Pred/real slope ratio, archive controls x1",
    ),
    Number(
        "shiftRarOnefive",
        "shift002-runtime.json",
        _shift("R_archive_x1.5"),
        2,
        "ci",
        "Pred/real slope ratio, archive controls x1.5",
    ),
    Number(
        "shiftRarTwo",
        "shift002-runtime.json",
        _shift("R_archive_x2.0"),
        2,
        "ci",
        "Pred/real slope ratio, archive controls x2",
    ),
    Number(
        "shiftdMch",
        "shift002-runtime.json",
        _shift("dM_challenge"),
        2,
        "ci",
        "Magnitude contrast R(0.30) - R(0.60), challenge",
    ),
    Number(
        "shiftdMar",
        "shift002-runtime.json",
        _shift("dM_archive"),
        2,
        "ci",
        "Magnitude contrast R(x1) - R(x2), archive",
    ),
    Number(
        "shiftdSar",
        "shift002-runtime.json",
        _shift("dS_archive"),
        3,
        "ci",
        "Structure contrast (varying - constant), archive",
    ),
    # ROBUST-001 (exploratory, opened records)
    Number(
        "robustReproChecks",
        "robust001-runtime.json",
        ("reproduction", "checks"),
        0,
        "value",
        "CALIB-001 values reproduced before reporting",
        "exploratory",
    ),
    Number(
        "robustKnnError",
        "robust001-runtime.json",
        ("knn", "validation_mean_error", "8"),
        2,
        "value",
        "kNN readout validation block error (k = 8)",
        "exploratory",
    ),
    *(
        Number(
            f"robustFlag{scene.capitalize()}{level}",
            "robust001-runtime.json",
            ("descriptive", "shift", f"reserved:{scene}|{cell}", "offmanifold_rate"),
            2,
            "value",
            f"Nearest-neighbour off-manifold flag rate, SHIFT-002 reserved {scene} {cell}",
            "exploratory",
        )
        for scene, cells in (("challenge", ("m0.30", "m0.60")), ("archive", ("x1.0", "x2.0")))
        for cell, level in zip(cells, ("Low", "High"), strict=True)
    ),
    *(
        Number(
            f"robust{scene.capitalize()}{tag}",
            "robust001-runtime.json",
            _robust(scene, instrument),
            2,
            "ci",
            f"Energy-ratio dose contrast, {scene}, {instrument}",
            "exploratory",
        )
        for scene in ("challenge", "archive")
        for instrument, tag in (("mlp-20261005", "Mlp"), ("ridge", "Ridge"), ("knn", "Knn"))
    ),
    Number(
        "robustOnChallengeRidge",
        "robust001-runtime.json",
        _robust("challenge", "ridge", "on"),
        2,
        "ci",
        "Energy-ratio dose contrast, challenge, ridge, unflagged predictions only",
        "exploratory (inconclusive)",
    ),
    # TRAIN-001 (240 fresh families; descriptive physical, prespecified latent)
    Number(
        "trainBalMinusArch",
        "train001-followup.json",
        ("exact_numbers", "primary", "balanced_minus_archive"),
        2,
        "ci",
        "High-dose endpoint error, balanced - archive-weighted",
        "descriptive",
    ),
    Number(
        "trainBalMinusFrozen",
        "train001-followup.json",
        ("exact_numbers", "primary", "balanced_minus_frozen"),
        2,
        "ci",
        "High-dose endpoint error, balanced - frozen",
        "descriptive",
    ),
    Number(
        "trainLatentBalMinusArch",
        "train001-followup.json",
        ("exact_numbers", "latent", "balanced_minus_archive"),
        4,
        "ci",
        "Terminal latent MSE, balanced - archive-weighted (prespecified secondary)",
    ),
    Number(
        "trainOrdBalMinusFrozen",
        "train001-followup.json",
        ("exact_numbers", "ordinary", "balanced_minus_frozen"),
        2,
        "ci",
        "Ordinary-window error, balanced - frozen",
        "not established",
    ),
    # SUPPORT-001 (300 fresh six-dose families; D036)
    *(
        Number(
            f"supSlope{scene.capitalize()}{tag}",
            "support001-runtime.json",
            (
                "summary",
                "readouts",
                "mlp-20261005",
                "dose_response",
                "mlp-20261005:block",
                scene,
                label,
                "b_rev",
            ),
            2,
            "ci",
            f"Block projection slope, {scene}, command {label[1:]}",
        )
        for scene in ("challenge", "archive")
        for label, tag in (
            ("m0.30", "Thirty"),
            ("m0.60", "Sixty"),
            ("m0.80", "Eighty"),
            ("m1.00", "Hundred"),
        )
    ),
    *(
        Number(
            f"supAgent{scene.capitalize()}Sixty",
            "support001-runtime.json",
            (
                "summary",
                "readouts",
                "mlp-20261005",
                "agent_probe",
                "mlp-20261005:agent",
                scene,
                "m0.60",
                "b_rev",
            ),
            2,
            "ci",
            f"Agent projection slope, {scene}, command 0.60 (gated)",
            "descriptive",
        )
        for scene in ("challenge", "archive")
    ),
    Number(
        "supCminusA",
        "support001-runtime.json",
        ("summary", "readouts", "mlp-20261005", "over_range_bootstrap", "C_minus_A"),
        1,
        "ci",
        "Over-range squared error, clamp - attenuation",
        "not established (rule)",
    ),
    Number(
        "supCminusB",
        "support001-runtime.json",
        ("summary", "readouts", "mlp-20261005", "over_range_bootstrap", "C_minus_B"),
        0,
        "ci",
        "Over-range squared error, clamp - reversion",
        "not established (rule)",
    ),
    # Archive command-magnitude tail of the SUPPORT-001 map (derived; REPORT-FIX-001)
    Number(
        "supMapPninetynine",
        "support001-runtime.json",
        ("map", "command_magnitude_quantiles", "p999"),
        2,
        "value",
        "Archive 99.9th percentile of ten-control mean-command magnitude",
        "descriptive",
    ),
    Number(
        "supMapAbove",
        "support001-account-slopes.json",
        ("command_tail", "above_0.60"),
        0,
        "value",
        "Sampled archive windows with mean-command magnitude above 0.60",
        "descriptive",
    ),
    Number(
        "supMapMax",
        "support001-account-slopes.json",
        ("command_tail", "max"),
        2,
        "value",
        "Largest sampled archive mean-command magnitude",
        "descriptive",
    ),
    # TRAIN-002 (240 fresh families + 120 windows; D035)
    Number(
        "tdOmitMinusBal",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "bootstrap", "dose_omitted_minus_balanced"),
        2,
        "ci",
        "Error at held-out 0.60, dose-omitted - balanced (primary)",
    ),
    Number(
        "tdTransfer",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "transfer_fraction"),
        2,
        "ci",
        "Transfer fraction of the fine-tuning gain to the held-out dose",
        "descriptive",
    ),
    Number(
        "tdFrozenMinusArchOnly",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "bootstrap", "frozen_minus_archive_only"),
        2,
        "ci",
        "Error at 0.60, frozen - archive-only fine-tuning",
        "not established",
    ),
    Number(
        "tdFrozenMinusBal",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "bootstrap", "frozen_minus_balanced"),
        1,
        "ci",
        "Error at 0.60, frozen - balanced fine-tuning",
    ),
    Number(
        "tdOrdBalMinusFrozen",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "ordinary", "bootstrap", "balanced_minus_frozen"),
        2,
        "ci",
        "Ordinary-window error, balanced - frozen (replication)",
    ),
    Number(
        "tdArchOnlyMinusBal",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "bootstrap", "archive_only_minus_balanced"),
        1,
        "ci",
        "Error at 0.60, archive-only - balanced fine-tuning",
        "secondary",
    ),
    Number(
        "tdOmitMinusBalInSupport",
        "train002-runtime.json",
        (
            "summary",
            "readouts",
            "mlp-20261005",
            "bootstrap",
            "in_support:dose_omitted_minus_balanced",
        ),
        2,
        "ci",
        "Error at in-support 0.45, dose-omitted - balanced (manipulation check)",
        "manipulation check",
    ),
    # TRAIN-002 absolute errors and ratios per model (REPORT-002; seed-averaged means)
    *(
        Number(
            f"tdMean{tag}",
            "train002-runtime.json",
            ("summary", "readouts", "mlp-20261005", "means_0.60", arm),
            2,
            "value",
            f"Mean terminal block error at 0.60, {arm}",
            "descriptive",
        )
        for arm, tag in TRAIN002_ARMS
    ),
    *(
        Number(
            f"tdOrdMean{tag}",
            "train002-runtime.json",
            ("summary", "readouts", "mlp-20261005", "ordinary", "means", arm),
            2,
            "value",
            f"Mean ordinary-window block error, {arm}",
            "descriptive",
        )
        for arm, tag in (*TRAIN002_ARMS, ("real-frames", "Real"))
    ),
    *(
        Number(
            f"tdRatio{scene.capitalize()}{tag}",
            "train002-runtime.json",
            ("summary", "readouts", "mlp-20261005", "slope_ratios", arm, f"{scene}:m0.60", "ratio"),
            2,
            "value",
            f"Pred/real slope ratio at 0.60, {scene}, {arm}",
            "descriptive",
        )
        for scene in ("challenge", "archive")
        for arm, tag in TRAIN002_ARMS
    ),
    *(
        Number(
            f"tdGateMedian{scene.capitalize()}",
            "train002-runtime.json",
            ("gate", "cells", f"{scene}:m0.60", "median_moved_true_displacement"),
            1,
            "value",
            f"Median true block displacement, {scene} 0.60 (TRAIN-002 confirmation)",
            "descriptive",
        )
        for scene in ("challenge", "archive")
    ),
    # H3-PLAN-002 (148 fresh families)
    Number(
        "planJepaMinusReal",
        "planning002-runtime.json",
        ("confirm", "summary", "pooled", "jepa_minus_real_all"),
        2,
        "ci",
        "Regret, JEPA - real-latent chooser",
        "not established",
    ),
    Number(
        "planRealMinusRandom",
        "planning002-runtime.json",
        ("confirm", "summary", "pooled", "real_minus_random_all"),
        2,
        "ci",
        "Regret, real-latent - random (latent-cost check)",
    ),
    # REPORT-REDESIGN-001: data description, physics and examples (D039)
    Number(
        "demoEpisodes",
        "model001-runtime.json",
        ("dataset", "episodes"),
        0,
        "count",
        "Episodes in LeWM's PushT training demos",
        "descriptive",
    ),
    Number(
        "demoFrames",
        "model001-runtime.json",
        ("dataset", "rows"),
        0,
        "count",
        "Frames in LeWM's PushT training demos",
        "descriptive",
    ),
    Number(
        "demoWindows",
        "support001-runtime.json",
        ("map", "windows"),
        0,
        "count",
        "Sampled stride-5 demo windows in the support map",
        "descriptive",
    ),
    *(
        Number(
            f"demoCommand{tag}",
            "support001-runtime.json",
            ("map", "command_magnitude_quantiles", key),
            2,
            "value",
            f"Demo ten-control mean-command magnitude, {key}",
            "descriptive",
        )
        for key, tag in (("p50", "Median"), ("p90", "Ninety"), ("p99", "Ninetynine"))
    ),
    *(
        Number(
            f"supTrue{scene.capitalize()}{tag}",
            "support001-runtime.json",
            ("gate", f"{scene}:{label}", "median_moved_true_displacement"),
            1,
            "value",
            f"Median true block displacement, {scene} {label} (SUPPORT-001 gate cell)",
            "descriptive",
        )
        for scene in ("challenge", "archive")
        for label, tag in (("m0.60", "Sixty"), ("m1.00", "Hundred"))
    ),
    *(
        Number(
            f"supReal{scene.capitalize()}Hundred",
            "support001-runtime.json",
            (
                "summary",
                "readouts",
                "mlp-20261005",
                "dose_response",
                "mlp-20261005:block",
                scene,
                "m1.00",
                "slope_real_moved",
            ),
            2,
            "ci",
            f"Real-frame block projection slope, {scene}, command 1.00",
        )
        for scene in ("challenge", "archive")
    ),
    Number(
        "supAgentChallengeEighty",
        "support001-runtime.json",
        (
            "summary",
            "readouts",
            "mlp-20261005",
            "agent_probe",
            "mlp-20261005:agent",
            "challenge",
            "m0.80",
            "b_rev",
        ),
        2,
        "ci",
        "Agent projection slope, designed, command 0.80 (gated)",
        "descriptive",
    ),
    Number(
        "readoutBlockError",
        "readout002-runtime.json",
        ("candidates", "mlp-20261005", "validation", "block", "mean"),
        1,
        "value",
        "Selected readout, validation block-position error (units)",
        "descriptive",
    ),
    Number(
        "readoutRidgeError",
        "readout002-runtime.json",
        ("candidates", "ridge", "validation", "block", "mean"),
        1,
        "value",
        "Linear ridge readout, validation block-position error (units)",
        "descriptive",
    ),
    Number(
        "readoutTrainFrames",
        "readout002-runtime.json",
        ("frames", "train"),
        0,
        "count",
        "Readout training frames",
        "descriptive",
    ),
    Number(
        "readoutValFrames",
        "readout002-runtime.json",
        ("frames", "validation"),
        0,
        "count",
        "Readout validation frames",
        "descriptive",
    ),
    Number(
        "ftParameters",
        "train001-runtime.json",
        ("pilot", "implementation", "trainable_parameters"),
        0,
        "count",
        "Fine-tuned parameters (predictor, action encoder, prediction projector)",
        "descriptive",
    ),
    *(
        Number(
            f"inView{scene.capitalize()}{tag}",
            "report-examples.json",
            ("agent_in_view", scene, label),
            0,
            "percent",
            f"Share of SUPPORT-001 {scene} families whose agent ends inside the table, {label}",
            "descriptive",
        )
        for scene in ("challenge", "archive")
        for label, tag in (("m0.80", "Eighty"), ("m1.00", "Hundred"))
    ),
    *(
        Number(
            f"exSup{name}{tag}",
            "report-examples.json",
            ("support001", "summary", label, field),
            0 if field == "agent_travel" else 1,
            "value",
            f"Example SUPPORT-001 family, {label}: {field.replace('_', ' ')}",
            "descriptive (example)",
        )
        for label, tag in (("m0.30", "Thirty"), ("m0.60", "Sixty"), ("m1.00", "Hundred"))
        for field, name in (("true", "True"), ("agent_travel", "Travel"))
    ),
    *(
        Number(
            f"exSupGain{tag}",
            "report-examples.json",
            ("support001", "summary", label, "frozen", "gain"),
            2,
            "value",
            f"Example SUPPORT-001 family, {label}: released-model gain along the true motion",
            "descriptive (example)",
        )
        for label, tag in (("m0.30", "Thirty"), ("m0.60", "Sixty"), ("m1.00", "Hundred"))
    ),
    Number(
        "tdTransferPercent",
        "train002-runtime.json",
        ("summary", "readouts", "mlp-20261005", "transfer_fraction", "estimate"),
        0,
        "percent",
        "Transfer fraction as a percentage (point estimate)",
        "descriptive",
    ),
    Number(
        "tdArchOnlyBound",
        "train002-runtime.json",
        (
            "summary",
            "readouts",
            "mlp-20261005",
            "bootstrap",
            "frozen_minus_archive_only",
            "ci",
            "1",
        ),
        1,
        "value",
        "Upper 95% bound on the ordinary-clip fine-tuning gain at 0.60",
        "not established (bound)",
    ),
    Number(
        "planBound",
        "planning002-runtime.json",
        ("confirm", "summary", "pooled", "jepa_minus_real_all", "ci95", "1"),
        1,
        "value",
        "Upper 95% bound on the JEPA - real-latent regret difference",
        "not established (bound)",
    ),
    Number(
        "exTrainTrue",
        "report-examples.json",
        ("train002", "summary", "m0.60", "true"),
        1,
        "value",
        "Example TRAIN-002 family, 0.60: true block displacement",
        "descriptive (example)",
    ),
)


def load(manifest: str, root: Path = MANIFESTS) -> Any:
    return json.loads((root / manifest).read_text())


def numbers(registry=REGISTRY, root: Path = MANIFESTS) -> dict[str, str]:
    cache: dict[str, Any] = {}
    out = {}
    for number in registry:
        if number.macro in out:
            raise ValueError(f"Duplicate macro {number.macro}")
        if not number.macro.isalpha():
            raise ValueError(f"Macro names must be letters only: {number.macro}")
        payload = cache.setdefault(number.manifest, load(number.manifest, root))
        out[number.macro] = render(number, payload)
    return out


def macros_tex(values: dict[str, str]) -> str:
    """One macro per number, set in math mode so that negative values get a true minus sign;
    interval values also get a point-estimate twin `\\<name>Pt`."""
    lines = ["% Generated by python -m jepa_action_effects.reporting; do not edit."]
    for name, value in sorted(values.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{value}}}}}")
        if "\\,[" in value:
            point = value.split(chr(92) + ",[")[0]
            lines.append(f"\\newcommand{{\\{name}Pt}}{{\\ensuremath{{{point}}}}}")
    return "\n".join(lines) + "\n"


def claims_md(values: dict[str, str], registry=REGISTRY, root: Path = MANIFESTS) -> str:
    lines = [
        "# REPORT-001 claims ledger",
        "",
        "Generated by `python -m jepa_action_effects.reporting`; do not edit. Each report",
        "number is a macro below, read from the named manifest field.",
        "",
        "| Macro | Claim | Value | Status | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n in registry:
        value = values[n.macro].replace("\\,", " ")
        source = f"`{n.manifest}` → `{'/'.join(n.path)}`"
        lines.append(f"| `\\{n.macro}` | {n.claim} | {value} | {n.status} | {source} |")
    hashes = sorted({n.manifest for n in registry})
    lines += ["", "Manifest SHA-256 at generation:", ""]
    lines += [f"- `{m}`: `{sha256_file(root / m)[:16]}…`" for m in hashes]
    return "\n".join(lines) + "\n"


FIGURES: dict[str, Callable[[Path], Path]] = {}


def figure(name: str):
    def register(function):
        FIGURES[name] = function
        return function

    return register


# One visual system for every figure: print sizes for A4 with 1.65 cm margins and a
# 0.55 cm column gap (figures are included at scale 1, so text sizes are true), the body
# font (Times via STIX), and a semantic colour-blind-safe palette (Okabe-Ito).
COLUMN, FULL = 3.38, 6.97
INK, MUTED, FAINT, SHADE = "#1b1b1b", "#6b6963", "#c9c6bf", "#f1efe9"
TRUE_C, REAL_C, AGENT_C, BLOCK_FILL = "#111111", "#8f8f8f", "#cc79a7", "#dcd9d2"
RELEASED, FT_ALL, FT_HELD, FT_DEMO, FT_ORD = "#d55e00", "#0072b2", "#009e73", "#56b4e9", "#e69f00"
ARMS = {  # internal arm id: (label, colour, line style, marker)
    "frozen": ("released LeWM", RELEASED, (0, (4, 1.6)), "o"),
    "archive_only": ("FT ordinary", FT_ORD, "-", "v"),
    "archive_weighted": ("FT demo-weighted", FT_DEMO, "-", "s"),
    "dose_omitted": ("FT w/o 0.60", FT_HELD, "-", "D"),
    "balanced": ("FT all strengths", FT_ALL, "-", "o"),
}
SCENE_NAMES = {"challenge": "designed scenes", "archive": "demo scenes"}
TEE_OUTLINE = ((-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30))
DOSES = (0.15, 0.30, 0.45, 0.60, 0.80, 1.00)
BAND = (0.85, 1.15)
READOUT = "mlp-20261005"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#55534f",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": "#55534f",
            "ytick.color": "#55534f",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": True,
            "grid.color": "#ebe9e4",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    return plt


def _dose_label(dose: float) -> str:
    return f"m{dose:.2f}"


def _panel(ax, text: str, x: float = 0.0, y: float = 1.0) -> None:
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom"
    )


def tee_outline(x: float, y: float, angle: float):
    """T-block outline in world units (y down), from the simulator's `add_tee` geometry."""
    import numpy as np

    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(TEE_OUTLINE, float) @ rotation.T + (x, y)


def _arrow(ax, start, end, color, lw=1.1, style="-", z=4, scale=7.0):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="-|>,head_length=0.5,head_width=0.25",
            mutation_scale=scale,
            color=color,
            lw=lw,
            linestyle=style,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
            joinstyle="miter",
        )
    )


def _frame(entry: dict[str, str]):
    """A tracked example frame, checked against the hash recorded at export."""
    import numpy as np
    from PIL import Image

    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise MissingField(f"{path} does not match report-examples.json")
    return np.asarray(Image.open(path).convert("RGB"))


def support_curves(payload: dict, accounts: dict, readout: str = READOUT) -> dict[str, Any]:
    """Block, real-frame block and gated agent slopes per scene and dose.

    Account curves are the projection slopes of each fitted account's prediction on the
    same families (`support001_derived`), kept for the claims record; None where the cell
    has no true block displacement. Agent cells count only where the real-frame agent slope
    CI lies inside the band.
    """
    if accounts["readout"] != readout:
        raise ValueError(f"Account slopes use {accounts['readout']}, not {readout}")
    summary = payload["summary"]["readouts"][readout]
    block = summary["dose_response"][f"{readout}:block"]
    agent = summary["agent_probe"][f"{readout}:agent"]
    implied = accounts["account_slopes"]
    out: dict[str, Any] = {}
    for scene in ("challenge", "archive"):
        rows = []
        for dose in DOSES:
            label = _dose_label(dose)
            b, a = block[scene][label], agent[scene][label]
            real = a["slope_real_moved"]
            gated = (
                real["ci95"] is not None and BAND[0] < real["ci95"][0] and real["ci95"][1] < BAND[1]
            )
            rows.append(
                {
                    "dose": dose,
                    "block": b["b_rev"],
                    "real": b.get("slope_real_moved"),
                    "agent": a["b_rev"] if gated else None,
                    **{key: implied[scene][label][key] for key in ("A", "B", "C")},
                }
            )
        out[scene] = rows
    return out


def _points(rows: list[dict], key: str) -> tuple[list, list, list, list]:
    pts = [(r["dose"], r[key]) for r in rows if r[key] and r[key]["estimate"] is not None]
    x = [d for d, _ in pts]
    y = [v["estimate"] for _, v in pts]
    low = [v["ci95"][0] if v.get("ci95") else v["estimate"] for _, v in pts]
    high = [v["ci95"][1] if v.get("ci95") else v["estimate"] for _, v in pts]
    return x, y, low, high


def _start_frame(ax, example: dict) -> None:
    """The real start frame the model sees, cropped around agent and block, with the push."""
    import numpy as np

    history = np.asarray(example["history_states"])
    agent, pose = history[2, 0:2], history[2, 2:5]
    corners = np.vstack([tee_outline(*pose), [agent]])
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2
    half = (corners.max(axis=0) - corners.min(axis=0)).max() / 2 + 28
    crop, (x0, x1, y0, y1) = _crop(_frame(example["frames"]["history-2"]), centre, half)
    ax.imshow(crop, extent=(x0, x1, y1, y0), interpolation="bilinear")
    command = np.asarray(example["mean_command"][-1], float)
    direction = command / np.linalg.norm(command)
    tail = agent + 20 * direction
    _arrow(ax, tail, tail + 62 * direction, AGENT_C, lw=1.6, scale=9)
    ax.text(*(tail + 40 * direction + (-10, 0)), "push", fontsize=6.8, color=AGENT_C, ha="right")
    ax.text(*(agent + (20, -6)), "agent", fontsize=6.8, color=INK, ha="left", va="center")
    block = np.mean(tee_outline(*pose), axis=0)
    ax.text(*(block + (8, 78)), "block", fontsize=6.8, color=INK, ha="center", va="center")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # y down, as rendered
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b9b6ae")
        spine.set_linewidth(0.6)


def _example_bars(ax, example: dict, labels: tuple[str, ...]) -> None:
    """True vs imagined block motion along the true direction, per push strength (units)."""
    import numpy as np

    rows = example["summary"]
    x = np.arange(len(labels))
    true = np.array([rows[label]["true"] for label in labels])
    imagined = np.array([rows[label]["frozen"]["gain"] for label in labels]) * true
    real = np.array([rows[label]["real_frame"]["gain"] for label in labels]) * true
    width = 0.36
    ax.bar(x - width / 2, true, width, color=TRUE_C, zorder=3, label="true motion")
    ax.bar(x + width / 2, imagined, width, color=RELEASED, zorder=3, label="LeWM imagines")
    ax.plot(
        x - width / 2,
        real,
        ls="",
        marker="o",
        ms=4.4,
        mfc="white",
        mec=REAL_C,
        mew=1.1,
        zorder=4,
        label="read from real end frame",
    )
    for k in range(len(labels)):
        ax.text(
            x[k] - width / 2,
            1.2,
            f"{true[k]:.0f}",
            fontsize=6.6,
            ha="center",
            va="bottom",
            color="white",
            zorder=5,
        )
        below = imagined[k] < 0
        ax.text(
            x[k] + width / 2,
            imagined[k] + (-1.2 if below else 1.2),
            f"{imagined[k]:.0f}".replace("-", "\u2212"),
            fontsize=6.6,
            ha="center",
            va="top" if below else "bottom",
            color=RELEASED,
        )
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    ax.set_xticks(x, [f"{float(label[1:]):.2f}" for label in labels])
    ax.set_xlabel("push strength $m$", labelpad=1.5)
    ax.set_ylabel("block motion (units)")
    ax.set_ylim(-12, 32)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis="x", visible=False)


def _fig_label(fig, ax, text: str, y: float, dx: float = -0.012) -> None:
    """Panel letter at a common figure height, just left of the panel."""
    fig.text(ax.get_position().x0 + dx, y, text, fontsize=9, fontweight="bold", ha="right")


def _beyond(ax, threshold: float, right: float = 1.08) -> None:
    ax.axvspan(threshold + 0.012, right, color=SHADE, lw=0, zorder=0)


@figure("teaser")
def teaser(out: Path) -> Path:
    """Fig. 1: a real example, the gain-vs-strength curve and the fine-tuning result."""
    plt = _style()

    examples = load("report-examples.json")
    support = load("support001-runtime.json")
    train = load("train002-runtime.json")
    fig = plt.figure(figsize=(FULL, 2.35))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(3.15, 2.1, 1.45),
        wspace=0.34,
        left=0.012,
        right=0.99,
        top=0.86,
        bottom=0.24,
    )

    # (a) One designed SUPPORT-001 family (D039 rule): the real start frame, and how far the
    # block truly moves vs how far LeWM imagines it moving, at three push strengths.
    example = examples["support001"]
    labels = ("m0.30", "m0.60", "m1.00")
    grid = outer[0].subgridspec(1, 2, width_ratios=(1.12, 1.25), wspace=0.4)
    scene = fig.add_subplot(grid[0])
    _start_frame(scene, example)
    scene.set_title("one test family (real frame)", fontsize=7.5, pad=2.5)
    motion = fig.add_subplot(grid[1])
    _example_bars(motion, example, labels)
    motion.set_title("how far the block moves", fontsize=7.5, pad=2.5)
    scenes = [scene]
    handles, names = motion.get_legend_handles_labels()
    order = [names.index(n) for n in ("true motion", "LeWM imagines", "read from real end frame")]
    handles, names = [handles[k] for k in order], [names[k] for k in order]
    first, last = scene.get_position(), motion.get_position()
    fig.legend(
        handles,
        names,
        loc="lower center",
        bbox_to_anchor=((first.x0 + last.x1) / 2, 0.0),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.9,
        handletextpad=0.35,
        fontsize=6.6,
    )

    # (b) Projection gain vs push strength (SUPPORT-001, 300 fresh families).
    beyond = support["map"]["command_magnitude_quantiles"]["p999"]
    curves = support_curves(support, load("support001-account-slopes.json"))
    sub = outer[1].subgridspec(2, 1, height_ratios=(1, 0.19), hspace=0.06)
    ax = fig.add_subplot(sub[0])
    strip = fig.add_subplot(sub[1], sharex=ax)
    for target in (ax, strip):
        _beyond(target, beyond)
    for scene, style, marker in (("challenge", "-", "o"), ("archive", (0, (3, 1.4)), "s")):
        rows = curves[scene]
        x, y, low, high = _points(rows, "real")
        ax.fill_between(x, low, high, color=REAL_C, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, y, color=REAL_C, lw=0.9, ls=style, zorder=2)
        x, y, _, _ = _points(rows, "agent")
        ax.plot(x, y, color=AGENT_C, lw=1.1, ls=style, marker=marker, ms=2.6, mew=0, zorder=3)
        x, y, low, high = _points(rows, "block")
        ax.fill_between(x, low, high, color=RELEASED, alpha=0.16, lw=0, zorder=2)
        ax.plot(
            x,
            y,
            color=RELEASED,
            lw=1.5,
            ls=style,
            marker=marker,
            ms=3,
            mec="white",
            mew=0.5,
            zorder=4,
        )
        ax.text(
            x[-1] + 0.035,
            y[-1],
            "designed" if scene == "challenge" else "demo",
            fontsize=6.5,
            color=RELEASED,
            va="center",
        )
    ax.axhline(1.0, color=MUTED, lw=0.5, zorder=1)
    ax.text(1.035, 0.99, "real\nframes", fontsize=6.3, color=MUTED, va="center", linespacing=1.0)
    ax.text(0.83, 0.905, "agent", fontsize=6.5, color=AGENT_C, va="center")
    ax.text(0.16, 0.745, "block", fontsize=6.5, color=RELEASED, va="center")
    ax.text(
        (beyond + 1.08) / 2 + 0.006,
        1.205,
        "beyond 99.9%\nof demo clips",
        fontsize=6.3,
        color=MUTED,
        ha="center",
        va="top",
        linespacing=1.0,
    )
    ax.set_ylim(0.32, 1.22)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("gain along true motion, $b$")
    ax.tick_params(labelbottom=False)
    histogram = examples["demo_commands"]
    edges, counts = histogram["edges"], histogram["counts"]
    widths = [b - a for a, b in zip(edges, edges[1:], strict=False)]
    strip.bar(edges[:-1], counts, width=widths, align="edge", color=FAINT, lw=0)
    strip.set_yticks([])
    strip.spines["left"].set_visible(False)
    strip.grid(False)
    strip.text(0.36, max(counts) * 0.5, "training demos", fontsize=6.3, color=MUTED)
    strip.set_xlim(0.1, 1.17)
    strip.set_xticks([0.15, 0.3, 0.45, 0.6, 0.8, 1.0], ["0.15", "0.3", "0.45", "0.6", "0.8", "1"])
    strip.set_xlabel("push strength $m$", labelpad=1.5)

    # (c) Mean block error at strength 0.60 after each training condition (TRAIN-002).
    summary = train["summary"]["readouts"][READOUT]
    bars = fig.add_subplot(outer[2])
    order = ("frozen", "archive_only", "archive_weighted", "dose_omitted", "balanced")
    short = {
        "frozen": "released",
        "archive_only": "FT ordinary",
        "archive_weighted": "FT demo-wt.",
        "dose_omitted": "FT w/o 0.60",
        "balanced": "FT all",
    }
    values = [summary["means_0.60"][arm] for arm in order]
    ys = list(range(len(order)))
    bars.barh(
        ys,
        values,
        color=[ARMS[a][1] for a in order],
        height=0.62,
        hatch=[("////" if a == "dose_omitted" else "") for a in order],
        edgecolor="white",
        lw=0,
    )
    for y_, v in zip(ys, values, strict=True):
        bars.text(v + 0.7, y_, f"{v:.1f}", va="center", fontsize=6.8, color=INK)
    bars.axvline(values[0], color=RELEASED, lw=0.6, ls=(0, (1.5, 1.5)), zorder=0)
    bars.set_yticks(ys, [short[a] for a in order])
    bars.set_ylim(len(order) - 0.45, -0.55)
    bars.set_xlim(0, 34)
    bars.set_xticks([0, 10, 20, 30])
    bars.grid(axis="y", visible=False)
    bars.set_xlabel("mean block error at 0.60", labelpad=1.5)
    for target, letter in ((scenes[0], "(a)"), (ax, "(b)"), (bars, "(c)")):
        _fig_label(fig, target, letter, 0.94, dx=-0.004 if letter == "(a)" else -0.045)
    return _save(fig, out, "teaser")


def _crop(frame, centre, half: float):
    """Crop a 224-pixel frame to a world window (centre, half-width in world units)."""
    scale = frame.shape[0] / 512.0
    x0, x1 = (int(round((centre[0] - half) * scale)), int(round((centre[0] + half) * scale)))
    y0, y1 = (int(round((centre[1] - half) * scale)), int(round((centre[1] + half) * scale)))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    return frame[y0:y1, x0:x1], (x0 / scale, x1 / scale, y0 / scale, y1 / scale)


@figure("readout")
def readout(out: Path) -> Path:
    """Readout layer blocks, including the fitted input/output transformations."""
    plt = _style()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(COLUMN, 1.12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, COLUMN), ylim=(0, 1.12))
    ax.axis("off")

    def box(x, y, width, text, fill=SHADE, height=0.38):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                facecolor=fill,
                edgecolor="#8f9398",
                linewidth=0.65,
            )
        )
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=8, linespacing=1.15)

    def arrow(start, end):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8, "shrinkA": 1},
        )

    top, bottom = 0.84, 0.27
    box(0.02, top, 0.43, "Latent\n192", "white")
    box(0.65, top, 0.73, "Standardise\n192", "#e7edf2")
    box(1.60, top, 0.77, "Linear + ReLU\n192 → 256")
    box(2.59, top, 0.77, "Linear + ReLU\n256 → 256")
    box(2.59, bottom, 0.77, "Linear\n256 → 6")
    box(1.49, bottom, 0.88, "Undo target\nstandardisation", "#e7edf2")
    box(0.02, bottom, 1.25, "Block x, y\nsin φ, cos φ\nAgent x, y", "white", height=0.50)
    for left, right in ((0.45, 0.65), (1.38, 1.60), (2.37, 2.59)):
        arrow((left, top), (right, top))
    arrow((2.975, top - 0.205), (2.975, bottom + 0.205))
    arrow((2.59, bottom), (2.37, bottom))
    arrow((1.49, bottom), (1.27, bottom))
    return _save(fig, out, "readout")


@figure("method")
def method(out: Path) -> Path:
    """Fig. 2: real model inputs and outcomes, the measurement pipeline and the metric."""
    plt = _style()
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

    example = load("report-examples.json")["train002"]
    width, height = FULL, 2.82
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    def framed(x0, y0, size):
        ax.add_patch(Rectangle((x0, y0), size, size, fill=False, ec="#b9b6ae", lw=0.6, zorder=3))

    # (a) What the model sees, and what really happens at four push strengths.
    size, top = 0.78, 1.82
    ax.text(0.03, 2.74, "(a)", fontsize=9, fontweight="bold", va="center")
    for k in range(3):
        x0, y0 = 0.14 + 0.06 * k, top + 0.06 - 0.06 * k
        ax.imshow(_frame(example["frames"][f"history-{k}"]), extent=(x0, x0 + size, y0, y0 + size))
        framed(x0, y0, size)
    ax.text(0.56, top - 0.1, "3 past frames, 0.5 s apart", fontsize=6.8, ha="center", va="top")
    ax.text(0.56, top - 0.24, "(model input)", fontsize=6.6, ha="center", va="top", color=MUTED)
    ax.annotate(
        "",
        xy=(1.52, top + size / 2),
        xytext=(1.13, top + size / 2),
        arrowprops={"arrowstyle": "-|>", "color": AGENT_C, "lw": 1.0},
    )
    ax.text(1.32, top + size / 2 + 0.07, "push\n1 s", fontsize=6.6, ha="center", va="bottom")
    history = np.asarray(example["history_states"])
    future = np.asarray(example["future_states"])
    # Use one common crop containing every final block and the full agent disk.
    # A block-only crop hid the agent at m=0.60 despite it remaining on the table.
    agent_ends = future[:, -1, :2]
    corners = np.vstack(
        [tee_outline(*history[2, 2:5])]
        + [tee_outline(*state[2:5]) for state in future[:, -1]]
        + [agent_ends - 15.0, agent_ends + 15.0]
    )
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2
    half = max(118.0, float(np.max(np.ptp(corners, axis=0))) / 2 + 12.0)
    for k, label in enumerate(example["labels"]):
        x0 = 1.58 + k * 0.86
        crop, (wx0, wx1, wy0, wy1) = _crop(
            _frame(example["frames"][f"future-{label}"]), centre, half
        )
        ax.imshow(crop, extent=(x0, x0 + size, top, top + size), interpolation="bilinear")
        outline = tee_outline(*history[2, 2:5])
        shape = Polygon(
            np.column_stack(
                (
                    x0 + (outline[:, 0] - wx0) / (wx1 - wx0) * size,
                    top + size - (outline[:, 1] - wy0) / (wy1 - wy0) * size,
                )
            ),
            fill=False,
            ec=INK,
            lw=0.6,
            ls=(0, (1.6, 1.2)),
            zorder=4,
        )
        ax.add_patch(shape)
        shape.set_clip_path(Rectangle((x0, top), size, size, transform=ax.transData))
        framed(x0, top, size)
        summary = example["summary"][label]
        ax.text(
            x0 + size / 2,
            top + size + 0.05,
            f"$m = {float(label[1:]):.2f}$",
            fontsize=7.2,
            ha="center",
        )
        ax.text(
            x0 + size / 2,
            top - 0.1,
            f"block moves {summary['true']:.0f}",
            fontsize=6.8,
            ha="center",
            va="top",
        )
        ax.text(
            x0 + size / 2,
            top - 0.24,
            f"agent travels {summary['agent_travel']:.0f}",
            fontsize=6.6,
            ha="center",
            va="top",
            color=AGENT_C,
        )

    # (c) The metric, drawn as vectors.
    ax.text(5.18, 2.74, "(c)", fontsize=9, fontweight="bold", va="center")
    o = np.array([5.32, 2.12])
    t = np.array([1.42, 0.0])
    angle = np.deg2rad(26)
    p = 0.6 * 1.42 * np.array([np.cos(angle), np.sin(angle)])
    r = np.array([1.4, 0.07])
    foot = o + np.array([p @ t / (t @ t) * t[0], 0.0])
    ax.plot([o[0] + p[0], foot[0]], [o[1] + p[1], foot[1]], color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    for vec, colour, name, offset in (
        (t, TRUE_C, "$t$ true", (0.03, -0.13)),
        (r, REAL_C, "$r$ real frame", (0.03, 0.04)),
        (p, RELEASED, "$p$ imagined", (-0.12, 0.06)),
    ):
        _arrow(ax, o, o + vec, colour, lw=1.3, scale=8)
        ax.text(*(o + vec + offset), name, fontsize=7, color=colour, ha="left", va="bottom")
    ax.plot([o[0], foot[0]], [o[1] - 0.08, o[1] - 0.08], color=RELEASED, lw=2.4, alpha=0.45)
    ax.text(
        (o[0] + foot[0]) / 2,
        o[1] - 0.14,
        "part of $p$ along $t$",
        fontsize=6.5,
        ha="center",
        va="top",
        color=RELEASED,
    )
    ax.text(
        5.32,
        1.72,
        r"gain $b = \Sigma\, p\cdot t \,/\, \Sigma\, t\cdot t$ (1 = full motion)",
        fontsize=7,
        va="center",
    )
    ax.text(
        5.32,
        1.57,
        r"ratio $R = b\,/\,b_{\mathrm{real}}$ (1 = as good as real frames)",
        fontsize=7,
        va="center",
    )

    # (b) The measurement pipeline.
    ax.text(0.03, 1.3, "(b)", fontsize=9, fontweight="bold", va="center")
    frozen_fill, trained_fill, data_fill = "#e2e8ef", "#fbe3d6", "#f3f2ee"

    def box(xc, yc, text, fill, w=0.86, h=0.3, edge="#9a978f", lw=0.6, colour=INK):
        ax.add_patch(
            FancyBboxPatch(
                (xc - w / 2, yc - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.045",
                fc=fill,
                ec=edge,
                lw=lw,
                zorder=2,
            )
        )
        ax.text(xc, yc, text, ha="center", va="center", fontsize=6.9, color=colour, zorder=3)

    def link(x0, y0, x1, y1, dashed=False, colour="#55534f"):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.3,head_width=0.15",
                "color": colour,
                "lw": 0.7,
                "linestyle": (0, (2.5, 1.5)) if dashed else "-",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    xs = (0.62, 1.66, 2.76, 3.86, 4.86, 5.86)
    lanes = (0.98, 0.55, 0.14)
    w = 0.86
    box(xs[0], lanes[0], "3 past frames", data_fill)
    box(xs[1], lanes[0], "encoder $E$\n(frozen)", frozen_fill)
    box(xs[2], lanes[0], "predictor $P$\n2 steps, fed back", trained_fill, w=1.0)
    box(xs[3], lanes[0], r"imagined latent $\hat z$", data_fill)
    box(xs[4], lanes[0], "readout $g_{\\mathrm{B}}$\n− start reading", frozen_fill)
    box(xs[5], lanes[0], "imagined\nmotion $p$", "white", edge=RELEASED, lw=0.9, colour=RELEASED)
    box(xs[2], 1.29, "push strength $m$", data_fill, w=1.0, h=0.2)
    box(xs[0], lanes[1], "real end frame", data_fill)
    box(xs[1], lanes[1], "encoder $E$\n(same)", frozen_fill)
    box(xs[3], lanes[1], "real latent $z$", data_fill)
    box(xs[4], lanes[1], "readout $g_{\\mathrm{B}}$\n− start reading", frozen_fill)
    box(xs[5], lanes[1], "real-frame\nmotion $r$", "white", edge=REAL_C, lw=0.9, colour="#5f5f5f")
    box(1.15, lanes[2], "simulator: same start, same push", data_fill, w=1.92, h=0.22)
    box(xs[5], lanes[2], "true motion $t$", "white", edge=TRUE_C, lw=0.9, h=0.22)
    for lane, keys in ((lanes[0], (0, 1, 2, 3, 4, 5)), (lanes[1], (0, 1, 3, 4, 5))):
        for left, right in zip(keys, keys[1:], strict=False):
            left_w, right_w = (1.0 if left == 2 else w), (1.0 if right == 2 else w)
            link(xs[left] + left_w / 2, lane, xs[right] - right_w / 2, lane)
    link(xs[2], 1.19, xs[2], lanes[0] + 0.15)
    link(xs[0], lanes[2] + 0.11, xs[0], lanes[1] - 0.15)
    link(2.11, lanes[2], xs[5] - w / 2, lanes[2])
    link(xs[3], lanes[1] + 0.15, xs[3], lanes[0] - 0.15, dashed=True, colour=MUTED)
    ax.text(
        xs[3] + 0.06, 0.765, "training target (§5 only)", fontsize=6.3, color=MUTED, va="center"
    )
    ax.text(
        xs[2] + 0.56, 1.29, "never sees the real future", fontsize=6.3, color=MUTED, va="center"
    )
    box(6.62, 0.55, "compare\n$p$ and $r$\nwith $t$:\n$b$, $R$, error", "white", w=0.6, h=1.0)
    for lane in lanes:
        link(xs[5] + w / 2, lane, 6.32, lane)
    return _save(fig, out, "method")


@figure("robustness")
def robustness(out: Path) -> Path:
    """Energy-ratio dose contrast per readout (ROBUST-001, exploratory)."""
    plt = _style()
    payload = load("robust001-runtime.json")
    instruments = ("mlp-20261005", "mlp-20261004", "mlp-20261006", "ridge", "knn")
    labels = ("MLP (main)", "MLP, seed 2", "MLP, seed 3", "linear (ridge)", "nearest neighbour")
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.5), sharey=True, sharex=True)
    titles = ("designed: 0.60 vs 0.30", "demo: ×2 vs ×1")
    for ax, scene, title in zip(axes, ("challenge", "archive"), titles, strict=True):
        for row, instrument in enumerate(instruments):
            leaf = _get(payload, _robust(scene, instrument), "robust001-runtime.json")
            low, high = leaf["ci95"]
            ax.plot([low, high], [row, row], color=RELEASED, lw=1.8, solid_capstyle="round")
            ax.plot(leaf["estimate"], row, "o", color=RELEASED, ms=3.6, mec="white", mew=0.7)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(instruments)), labels)
    axes[0].set_ylim(len(instruments) - 0.5, -0.5)
    axes[0].set_xlim(-1.15, 0.2)
    axes[0].set_xticks([-1.0, -0.5, 0.0])
    fig.supxlabel("change in energy ratio, strong − weak push (95% CI)", fontsize=7.5, y=0.02)
    fig.tight_layout(w_pad=0.6, pad=0.3)
    return _save(fig, out, "robustness")


CURVE_ARMS = (
    ("balanced", FT_ALL, "FT: all strengths"),
    ("archive_weighted", FT_DEMO, "FT: demo-weighted"),
    ("dose_omitted", FT_HELD, "FT: without 0.60"),
    ("archive_only", FT_ORD, "FT: ordinary clips"),
)


def seed_mean_curves(payload: dict, key: str, smooth: int = 1) -> dict[str, tuple[list, list]]:
    """Seed-averaged curve per arm; optional trailing moving average over `smooth` points."""
    import numpy as np

    out = {}
    for arm, _, _ in CURVE_ARMS:
        runs = [r for r in payload["runs"].values() if r["arm"] == arm]
        if not runs:
            raise MissingField(f"training-curves.json: no runs for {arm}")
        updates = [u for u, _ in runs[0][key]]
        if any([u for u, _ in r[key]] != updates for r in runs):
            raise ValueError(f"{arm}: seeds logged {key} at different updates")
        values = np.mean([[v for _, v in r[key]] for r in runs], axis=0)
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values = np.convolve(values, kernel, mode="valid")
            updates = updates[smooth - 1 :]
        out[arm] = (list(updates), [float(v) for v in values])
    return out


@figure("finetuning")
def finetuning(out: Path) -> Path:
    """R by strength per training arm, a real example at 0.60, validation curves."""
    plt = _style()
    from matplotlib.lines import Line2D

    payload = load("train002-runtime.json")
    ratios = payload["summary"]["readouts"][READOUT]["slope_ratios"]
    example = load("report-examples.json")["train002"]
    curves = seed_mean_curves(load("training-curves.json"), "validation_curve")
    fig = plt.figure(figsize=(FULL, 2.0))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(2.85, 1.3, 1.75),
        wspace=0.36,
        left=0.065,
        right=0.99,
        top=0.79,
        bottom=0.2,
    )
    order = ("frozen", "archive_only", "archive_weighted", "dose_omitted", "balanced")
    doses = (0.30, 0.45, 0.60)
    grid = outer[0].subgridspec(1, 2, wspace=0.08)
    axes = [fig.add_subplot(grid[0])]
    axes.append(fig.add_subplot(grid[1], sharey=axes[0]))
    for ax, scene in zip(axes, ("challenge", "archive"), strict=True):
        ax.axvspan(0.525, 0.66, color=SHADE, lw=0, zorder=0)
        for arm in order:
            _, colour, style, marker = ARMS[arm]
            y = [ratios[arm][f"{scene}:{_dose_label(d)}"]["ratio"] for d in doses]
            ax.plot(
                doses,
                y,
                ls=style,
                marker=marker,
                color=colour,
                lw=1.5,
                ms=3.2,
                mec="white",
                mew=0.5,
                zorder=3,
            )
        ax.axhline(1.0, color=MUTED, lw=0.5)
        ax.set_title(SCENE_NAMES[scene], fontsize=7.5, pad=3)
        ax.set_xticks(doses, ["0.30", "0.45", "0.60"])
        ax.set_xlim(0.26, 0.66)
        ax.set_xlabel("push strength $m$", labelpad=1.5)
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("$R$ = gain / real-frame gain")
    axes[0].set_ylim(0.55, 1.06)
    axes[0].text(0.5925, 0.565, "held out\nfor FT w/o 0.60", fontsize=6, color=MUTED, ha="center")

    # (b) The TRAIN-002 example family at 0.60 (D039): motion along the true path.
    ax = fig.add_subplot(outer[1])
    row = example["summary"]["m0.60"]
    true = row["true"]
    rows = [
        ("true", true, TRUE_C, "s"),
        ("real frame", row["real_frame"]["gain"] * true, REAL_C, "s"),
    ] + [
        (short, row[arm]["gain"] * true, ARMS[arm][1], ARMS[arm][3])
        for arm, short in (
            ("frozen", "released"),
            ("archive_only", "FT ordinary"),
            ("dose_omitted", "FT w/o 0.60"),
            ("balanced", "FT all"),
        )
    ]
    for k, (_name, value, colour, marker) in enumerate(rows):
        ax.plot([0, value], [k, k], color=colour, lw=1.6, solid_capstyle="butt")
        ax.plot(value, k, marker, color=colour, ms=4, mec="white", mew=0.5)
    ax.axvline(true, color=TRUE_C, lw=0.6, ls=(0, (2, 1.5)), zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, 1.15 * max(r[1] for r in rows))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("block motion along its true path", labelpad=1.5)
    ax.set_title("one test family, push 0.60", fontsize=7.5, pad=3)

    # (c) Validation latent MSE on the 120 validation families (60 designed, 60 demo).
    curve_ax = fig.add_subplot(outer[2])
    for arm, colour, _ in CURVE_ARMS:
        curve_ax.plot(*curves[arm], "-o", color=colour, lw=1.3, ms=2.4, mec="white", mew=0.4)
    curve_ax.set_xticks([0, 1000, 2000])
    curve_ax.set_xlim(0, 2080)
    curve_ax.set_ylim(0.08, 0.2)
    curve_ax.set_xlabel("update", labelpad=1.5)
    curve_ax.set_ylabel("validation latent MSE")
    curve_ax.text(1040, 0.192, "ordinary clips: flat", fontsize=6.3, color=FT_ORD, ha="center")
    curve_ax.text(1040, 0.111, "push families", fontsize=6.3, color=FT_ALL, ha="center")

    handles = [
        Line2D([], [], color=ARMS[a][1], ls=ARMS[a][2], marker=ARMS[a][3], ms=3.2, lw=1.5)
        for a in order
    ]
    labels = [ARMS[a][0] for a in order]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        handlelength=2.0,
        columnspacing=0.8,
        handletextpad=0.4,
        fontsize=6.5,
    )
    for target, letter in ((axes[0], "(a)"), (ax, "(b)"), (curve_ax, "(c)")):
        _fig_label(fig, target, letter, 0.835, dx=-0.05 if letter != "(b)" else -0.075)
    return _save(fig, out, "finetuning")


FIGURE_ONLY_MANIFESTS = ("training-curves.json",)  # read only by figures


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return pdf


def build(out: Path = OUT, figures: bool = True) -> dict[str, Any]:
    values = numbers()
    (out / "pdf").mkdir(parents=True, exist_ok=True)  # Tectonic's --outdir must exist
    (out / "numbers.tex").write_text(macros_tex(values))
    CLAIMS.write_text(claims_md(values))
    made = [str(FIGURES[name](out / "figures")) for name in FIGURES] if figures else []
    used = {n.manifest for n in REGISTRY} | set(FIGURE_ONLY_MANIFESTS)
    inputs = {m: sha256_file(MANIFESTS / m) for m in sorted(used)}
    (out / "inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
    return {"numbers": len(values), "figures": made}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(figures=not args.no_figures)))


if __name__ == "__main__":
    main()
