"""TRAIN-001 report: compact tracked manifest, figures and an auto-generated table draft.

This module only formats sealed results; it performs no inference or statistics and
is therefore outside the sealed source set. Production output goes to
`docs/manifests/train001-runtime.json`, `docs/figures/train001/` and
`docs/TRAIN-001_RESULT_DRAFT.md`; smoke output stays under the smoke run root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jepa_action_effects.latent_audit import sealed, write_json
from jepa_action_effects.pilot_collection import sha256_file
from jepa_action_effects.train001_protocol import (
    CONDITIONS,
    MAGNITUDES,
    READOUTS,
    SCENES,
    Protocol,
    magnitude_label,
)

COLORS = {"frozen": "#2a78d6", "archive_weighted": "#eb6834", "balanced": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MAIN = ("balanced_minus_archive", "balanced_minus_frozen", "archive_minus_frozen")


def _outputs(protocol: Protocol, root: Path) -> tuple[Path, Path, Path]:
    if protocol.name == "train001":
        return (
            Path("docs/manifests/train001-runtime.json"),
            Path("docs/figures/train001"),
            Path("docs/TRAIN-001_RESULT_DRAFT.md"),
        )
    base = root / "report"
    return base / "train001-runtime.json", base / "figures", base / "RESULT_DRAFT.md"


def _compact_primary(block: dict) -> dict:
    boot = block["primary"]["bootstrap"]
    return {
        "verdict": block["primary"]["verdict"],
        "contrasts": {k: boot[k] for k in boot if not k.startswith("abs:") and "@" not in k},
        "per_seed": {k: boot[k] for k in boot if "@" in k},
        "absolute": {k: boot[k] for k in boot if k.startswith("abs:")},
        "seed_spread": block["primary"]["seed_spread"],
        "measurement_gates": block["measurement_gates"],
        "ordinary": {
            "outcome": block["ordinary"]["outcome"],
            "tolerance": block["ordinary"]["tolerance"],
            "windows": block["ordinary"]["windows"],
            "contrasts": {k: v for k, v in block["ordinary"]["bootstrap"].items() if "@" not in k},
        },
        "action_permuted": {k: v["estimate"] for k, v in block["action_permuted"].items()},
        "action_permuted_ci": {k: v["ci"] for k, v in block["action_permuted"].items()},
        "calibration_scale": block["calibration_scale"],
    }


def _condition_diagnostics(block: dict, seeds) -> dict[str, dict[str, dict[str, float | None]]]:
    """Seed-averaged per-cell point estimates for each condition."""
    diagnostics = block["diagnostics"]
    result: dict[str, dict] = {}
    groups = {"frozen": ["frozen"]} | {c: [f"{c}-{s}" for s in seeds] for c in CONDITIONS}
    for condition, members in groups.items():
        result[condition] = {}
        for cell in diagnostics["frozen"]:
            values = {}
            for key in (
                "endpoint_error_mean",
                "displacement_error_mean",
                "slope_pred",
                "slope_real",
                "slope_ratio_pred_over_real",
            ):
                numbers = [diagnostics[m][cell][key] for m in members]
                numbers = [n for n in numbers if n is not None]
                values[key] = float(np.mean(numbers)) if numbers else None
            values["moved"] = diagnostics["frozen"][cell]["moved"]
            values["median_true_displacement"] = diagnostics["frozen"][cell][
                "median_true_displacement"
            ]
            result[condition][cell] = values
    return result


def _physical_label(summary: dict) -> str:
    """Physical plots carry the pooled-eligibility qualification (D029)."""
    if summary["pooled_verdict"]["verdict"].startswith("not_assessable"):
        return (
            "DESCRIPTIVE ONLY: pooled physical hypothesis not assessable (development gate failed)"
        )
    if summary["pooled_verdict"]["verdict"].startswith("inconclusive"):
        return "DESCRIPTIVE ONLY: confirmation measurement gate failed"
    return "confirmatory primary readout"


def _figures(directory: Path, runs: dict, summary: dict, conditions: dict, seeds) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    written = []
    styles = {seed: style for seed, style in zip(seeds, ("-", "--", ":"), strict=False)}

    # 1. learning curves
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for _name, run in sorted(runs.items()):
        color = COLORS[run["condition"]]
        curve = np.asarray(run["curve"])
        axes[0].plot(
            curve[:, 0],
            curve[:, 1],
            styles[run["seed"]],
            color=color,
            lw=1.2,
            label=f"{run['condition']} seed {run['seed']}",
        )
        validation = np.asarray(run["validation_curve"])
        if len(validation):
            axes[1].plot(
                validation[:, 0],
                validation[:, 1],
                styles[run["seed"]],
                color=color,
                lw=2,
                marker="o",
                ms=4,
            )
    axes[0].set(
        xlabel="optimizer update",
        ylabel="weighted training loss (latent MSE)",
        title="Training objective",
        yscale="log",
    )
    axes[1].set(
        xlabel="optimizer update",
        ylabel="validation latent MSE (equal weights)",
        title="Development validation (diagnostic)",
    )
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    path = directory / "learning_curves.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(str(path))

    # 2. magnitude response (primary readout, seed-averaged per condition)
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for column, scene in enumerate(SCENES):
        for condition in ("frozen", *CONDITIONS):
            cells = [conditions[condition][f"{scene}:{magnitude_label(m)}"] for m in MAGNITUDES]
            error = [c["endpoint_error_mean"] for c in cells]
            ratio = [c["slope_ratio_pred_over_real"] for c in cells]
            label = condition.replace("_", "-")
            axes[0, column].plot(
                MAGNITUDES, error, "-o", color=COLORS[condition], lw=2, ms=5, label=label
            )
            axes[1, column].plot(
                MAGNITUDES,
                [np.nan if r is None else r for r in ratio],
                "-o",
                color=COLORS[condition],
                lw=2,
                ms=5,
                label=label,
            )
        axes[0, column].set(title=f"{scene} scenes", ylabel="endpoint error (workspace units)")
        axes[1, column].axhline(1.0, color=MUTED, lw=0.8)
        axes[1, column].set(xlabel="command magnitude", ylabel="pred/real projection slope")
        for row in (0, 1):
            axes[row, column].axvspan(0.4, 0.65, color=GRID, alpha=0.5, lw=0)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Confirmation magnitude response (decoded; shaded = primary branches)\n"
        + _physical_label(summary),
        fontsize=9,
    )
    figure.tight_layout()
    path = directory / "magnitude_response.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(str(path))

    # 3. paired contrasts (primary, secondary latent, ordinary)
    primary = summary["readouts"][READOUTS[0]]
    panels = [
        (
            "Physical, descriptive: 0.45/0.60 endpoint error\n(workspace units)",
            primary["primary"]["bootstrap"],
        ),
        (
            "Secondary (prespecified): terminal latent MSE",
            summary["secondary_terminal_latent_mse"]["contrasts"],
        ),
        (
            "Ordinary windows: endpoint error\n(workspace units; dashed = 1.28 tolerance)",
            primary["ordinary"]["bootstrap"],
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for axis, (title, boot) in zip(axes, panels, strict=True):
        rows = []
        for name in MAIN:
            rows.append((name.replace("_minus_", " − ").replace("_", "-"), boot[name], 2.2, INK))
            for seed in seeds:
                rows.append((f"   seed {seed}", boot[f"{name}@{seed}"], 1.0, MUTED))
        for y, (_label, entry, width, color) in enumerate(reversed(rows)):
            if entry["ci"] is not None:
                axis.plot(entry["ci"], [y, y], color=color, lw=width)
            axis.plot([entry["estimate"]], [y], "o", color=color, ms=4 if width < 2 else 6)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([r[0] for r in reversed(rows)], fontsize=7)
        axis.axvline(0.0, color=MUTED, lw=0.8)
        axis.set_title(title, fontsize=8)
    axes[2].axvline(1.28, color=COLORS["archive_weighted"], lw=0.8, ls="--")
    figure.suptitle(
        "Paired family-bootstrap contrasts (95% CI; seeds not resampled)\n"
        + _physical_label(summary),
        fontsize=9,
    )
    figure.tight_layout()
    path = directory / "paired_errors.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(str(path))
    return written


def _fmt(entry: dict | None, digits: int = 3) -> str:
    if entry is None:
        return "n/a"
    ci = entry.get("ci")
    ci_text = f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "n/a"
    return f"{entry['estimate']:.{digits}f} {ci_text}"


def _draft(
    summary: dict, development: dict, manifest: dict, seeds, sensitivity: dict | None = None
) -> str:
    primary = summary["readouts"][READOUTS[0]]
    lines = [
        "# TRAIN-001 auto-generated tables",
        "",
        "Generated by `python -m jepa_action_effects.train001 report` from sealed outputs.",
        "Interpretation lives in `TRAIN-001_RESULT.md`.",
        "",
        f"- Development eligibility: `{summary['development_eligibility']}`",
        f"- Pooled verdict: `{summary['pooled_verdict']['verdict']}`",
        f"- Ordinary-case outcome: `{summary['ordinary_outcome']}`",
        "- Readout sensitivity: see the contrast-level table below. The sealed verdict-string",
        "  flag cannot detect disagreement once development eligibility fails (D029).",
        f"- Physical tables: {_physical_label(summary)}.",
        f"- Selected learning rate: `{manifest['pilot']['selected_learning_rate']}`",
        "",
        "## Measurement gates (confirmation, primary readout)",
        "",
        "| Cell | Moved | slope_real [CI] | Floor | Passed |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for cell, gate in primary["measurement_gates"]["cells"].items():
        ci = gate["slope_real_ci"]
        slope = f"{gate['slope_real']:.3f}" if gate["slope_real"] is not None else "n/a"
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
        floor = f"{gate['floor']:.3f}" if gate["floor"] is not None else "n/a"
        lines.append(
            f"| {cell} | {gate['moved']} | {slope} {ci_text} | {floor} | {gate['passed']} |"
        )
    for title, boot, digits in (
        (
            "Physical contrasts (endpoint error, 0.45/0.60; descriptive if not assessable)",
            primary["primary"]["bootstrap"],
            3,
        ),
        (
            "Secondary: terminal latent MSE",
            summary["secondary_terminal_latent_mse"]["contrasts"],
            5,
        ),
        ("Ordinary windows (endpoint error)", primary["ordinary"]["bootstrap"], 3),
        (
            "Development validation (EXPLORATORY) primary contrasts",
            development["readouts"][READOUTS[0]]["primary"]["bootstrap"],
            3,
        ),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Quantity | Estimate [95% CI] | challenge | archive |",
            "| --- | --- | --- | --- |",
        ]
        for name, entry in boot.items():
            scenes = entry.get("by_scene", {})
            lines.append(
                f"| {name} | {_fmt(entry, digits)} | {_fmt(scenes.get('challenge'), digits)} | "
                f"{_fmt(scenes.get('archive'), digits)} |"
            )
    if sensitivity is not None:
        header = " | ".join(READOUTS)
        lines += [
            "",
            "## Readout sensitivity by contrast (interval sign per readout)",
            "",
            f"| Quantity | {header} | Signs agree |",
            "| --- |" + " --- |" * len(READOUTS) + " --- |",
        ]
        for name, entry in sensitivity["quantities"].items():
            cells = " | ".join(
                f"{_fmt(entry['per_readout'][r])} ({entry['per_readout'][r]['sign']})"
                for r in READOUTS
            )
            lines.append(f"| {name} | {cells} | {entry['signs_agree']} |")
        outcome = sensitivity["ordinary_outcome"]["per_readout"]
        lines += [
            "",
            f"Ordinary outcome per readout: `{outcome}`.",
            f"Confirmation gates agree across readouts: "
            f"`{sensitivity['confirmation_gates']['agree']}`.",
        ]
    return "\n".join(lines) + "\n"


def regenerate_presentation(protocol: Protocol, root: Path, sensitivity: dict) -> dict[str, Any]:
    """Rewrite figures and the table draft from sealed outputs; the manifest is untouched."""
    _, figures, draft_path = _outputs(protocol, root)
    summary = sealed(root / "confirm" / "summary.json")
    development = sealed(root / "development-eval.json")
    pilot = sealed(root / "pilot" / "pilot.json")
    seeds = protocol.training_seeds
    runs = {
        f"{c}-{s}": sealed(root / "runs" / f"{c}-{s}" / "run.json")
        for c in CONDITIONS
        for s in seeds
    }
    conditions = _condition_diagnostics(summary["readouts"][READOUTS[0]], seeds)
    written = _figures(figures, runs, summary, conditions, seeds)
    manifest = {"pilot": {"selected_learning_rate": pilot["selected_learning_rate"]}}
    draft_path.write_text(_draft(summary, development, manifest, seeds, sensitivity))
    return {"figures": written, "draft": str(draft_path)}


def write_report(protocol: Protocol, root: Path, budget) -> dict[str, Any]:
    manifest_path, figures, draft_path = _outputs(protocol, root)
    seal_payload = sealed(root / "seal.json")
    summary = sealed(root / "confirm" / "summary.json")
    development = sealed(root / "development-eval.json")
    pilot = sealed(root / "pilot" / "pilot.json")
    eligibility = sealed(root / "pilot" / "eligibility.json")
    collections = {p: sealed(root / p / "collection.json") for p in ("development", "confirmation")}
    audit = sealed(root / "audit.json")
    seeds = protocol.training_seeds
    runs = {
        f"{c}-{s}": sealed(root / "runs" / f"{c}-{s}" / "run.json")
        for c in CONDITIONS
        for s in seeds
    }
    primary = summary["readouts"][READOUTS[0]]
    conditions = _condition_diagnostics(primary, seeds)
    manifest = {
        "task": "TRAIN-001",
        "protocol_sha256": protocol.payload_sha256(),
        "seal_sha256": sha256_file(root / "seal.json"),
        "summary_sha256": sha256_file(root / "confirm" / "summary.json"),
        "git_at_seal": seal_payload.get("git"),
        "sources": seal_payload["sources"],
        "audit": {
            "exclusion_counts": audit["exclusion_counts"],
            "checkpoint": audit["checkpoint"],
            "archive": audit["archive"],
            "readouts": audit["readouts"],
        },
        "collection": {
            phase: {
                "counts": c["counts"],
                "rejections": c["rejections"],
                "seconds": c.get("seconds"),
                **(
                    {"weights": {k: c["weights"][k] for k in c["weights"] if k != "episodes"}}
                    if "weights" in c
                    else {}
                ),
            }
            for phase, c in collections.items()
        },
        "eligibility": {"status": eligibility["status"], "gates": eligibility["gates"]},
        "pilot": {
            key: pilot[key]
            for key in (
                "checks",
                "status",
                "implementation",
                "equivalence",
                "selected_learning_rate",
                "learning_rate_sweep",
                "throughput",
            )
        }
        | {
            "overfit": {
                k: pilot["overfit"][k] for k in ("reduction", "required", "passed", "curve")
            }
        },
        "runs": {
            name: {
                "condition": run["condition"],
                "seed": run["seed"],
                "learning_rate": run["learning_rate"],
                "updates": run["updates"],
                "weights": run["weights"],
                "checkpoints": run["checkpoints"],
                "final_train_loss": run["curve"][-1][1],
                "validation_curve": run["validation_curve"],
            }
            for name, run in runs.items()
        },
        "calibration": sealed(root / "calibration.json")["scales"],
        "confirmation": {
            "development_eligibility": summary["development_eligibility"],
            "pooled_verdict": summary["pooled_verdict"],
            "ordinary_outcome": summary["ordinary_outcome"],
            "readout_sensitivity": summary["readout_sensitivity"],
            "secondary_terminal_latent_mse": summary["secondary_terminal_latent_mse"],
            "readouts": {r: _compact_primary(b) for r, b in summary["readouts"].items()},
            "condition_diagnostics_primary_readout": conditions,
        },
        "development_exploratory": {
            "label": development["label"],
            "primary_readout": _compact_primary(development["readouts"][READOUTS[0]]),
            "secondary_terminal_latent_mse": development["secondary_terminal_latent_mse"][
                "contrasts"
            ],
        },
        "ledger": {"used_seconds": budget.used(), "sessions": budget.load()["sessions"]},
    }
    manifest["figures"] = _figures(figures, runs, summary, conditions, seeds)
    write_json(manifest_path, manifest)
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    from jepa_action_effects.train001_followup import contrast_sensitivity

    draft_path.write_text(
        _draft(summary, development, manifest, seeds, contrast_sensitivity(summary))
    )
    return {
        "manifest": str(manifest_path),
        "figures": manifest["figures"],
        "draft": str(draft_path),
    }
