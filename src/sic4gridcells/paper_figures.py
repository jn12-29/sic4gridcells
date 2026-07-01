from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sic4gridcells.figure_data import FigureDataBundle, load_figure_data, write_summary_tables
from sic4gridcells.runtime import atomic_json_dump


FIGURE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fig_grid_modules", ("unit_modules", "module_summary")),
    ("fig_arena_generalization", ("cross_arena_unit_metrics", "cross_arena_module_stability", "module_summary")),
    ("fig_path_invariance", ("path_invariance_summary", "path_invariance_bins")),
    ("fig_fourier_phase_state", ("fourier_lattice_vectors", "phase_tiling", "state_space_summary")),
    ("fig_ablations", ("module_summary", "path_invariance_summary")),
)


@dataclass(frozen=True)
class FigureBuildResult:
    suite_dir: Path
    output_dir: Path
    manifest_path: Path
    figures: dict[str, dict[str, str]]
    summary_tables_dir: Path


def build_paper_figures(
    suite_dir: str | Path,
    output_dir: str | Path,
) -> FigureBuildResult:
    """Render paper-result-style figures from existing suite analysis artifacts."""

    bundle = load_figure_data(suite_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_outputs = write_summary_tables(bundle, out_dir)

    plotters: dict[str, Callable[[FigureDataBundle], plt.Figure]] = {
        "fig_grid_modules": _plot_grid_modules,
        "fig_arena_generalization": _plot_arena_generalization,
        "fig_path_invariance": _plot_path_invariance,
        "fig_fourier_phase_state": _plot_fourier_phase_state,
        "fig_ablations": _plot_ablations,
    }
    figure_outputs: dict[str, dict[str, str]] = {}
    manifest_figures = []
    for figure_name, table_names in FIGURE_SPECS:
        fig = plotters[figure_name](bundle)
        png_path = out_dir / f"{figure_name}.png"
        pdf_path = out_dir / f"{figure_name}.pdf"
        fig.savefig(png_path, dpi=180)
        fig.savefig(pdf_path)
        plt.close(fig)
        figure_outputs[figure_name] = {"png": str(png_path), "pdf": str(pdf_path)}
        deps = sorted(
            {
                dependency
                for table_name in table_names
                for dependency in bundle.dependencies.get(table_name, [])
            }
        )
        manifest_figures.append(
            {
                "name": figure_name,
                "outputs": figure_outputs[figure_name],
                "dependencies": deps,
                "tables": list(table_names),
            }
        )

    manifest = {
        "suite_dir": str(Path(suite_dir)),
        "output_dir": str(out_dir),
        "figures": manifest_figures,
        "summary_tables": table_outputs,
        "runs": [
            {
                "run_id": run.run_id,
                "variant": run.variant,
                "seed": run.seed,
                "status": run.status,
                "diagnostic_only": run.diagnostic_only,
                "checkpoint_path": None if run.checkpoint_path is None else str(run.checkpoint_path),
                "eval_output_dir": None if run.eval_output_dir is None else str(run.eval_output_dir),
                "analysis_output_dir": (
                    None if run.analysis_output_dir is None else str(run.analysis_output_dir)
                ),
                "validation_report_path": (
                    None
                    if run.validation_report_path is None
                    else str(run.validation_report_path)
                ),
                "validation_passed": run.validation_passed,
            }
            for run in bundle.runs
        ],
        "diagnostic_runs": bundle.diagnostic_runs,
    }
    manifest_path = out_dir / "figure_manifest.json"
    atomic_json_dump(manifest, manifest_path)
    return FigureBuildResult(
        suite_dir=Path(suite_dir),
        output_dir=out_dir,
        manifest_path=manifest_path,
        figures=figure_outputs,
        summary_tables_dir=out_dir / "summary_tables",
    )


def _plot_grid_modules(bundle: FigureDataBundle) -> plt.Figure:
    rows = bundle.table("module_summary")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    fig.suptitle("Grid modules")
    if not rows:
        _no_data(axes[0])
        _no_data(axes[1])
        return fig
    variants = _variant_order(rows)
    colors = _variant_colors(variants)
    for variant in variants:
        subset = [row for row in rows if row.get("variant") == variant]
        x = [_float(row.get("arena_size")) for row in subset]
        y = [_float(row.get("mean_scale_meters")) for row in subset]
        sizes = [30 + 18 * (_float(row.get("unit_count")) or 0.0) for row in subset]
        axes[0].scatter(x, y, s=sizes, color=colors[variant], alpha=0.8, label=variant)
    axes[0].set_xlabel("arena size (m)")
    axes[0].set_ylabel("module scale (m)")
    axes[0].legend(fontsize=7)

    scores = [_float(row.get("mean_grid_score_60")) for row in rows]
    scores = [value for value in scores if value is not None]
    if scores:
        axes[1].hist(scores, bins=min(12, max(4, len(scores))), color="#4c78a8", edgecolor="white")
    else:
        _no_data(axes[1])
    axes[1].set_xlabel("module mean grid score 60")
    axes[1].set_ylabel("modules")
    return fig


def _plot_arena_generalization(bundle: FigureDataBundle) -> plt.Figure:
    cross_rows = bundle.table("cross_arena_unit_metrics")
    module_rows = bundle.table("module_summary")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    fig.suptitle("Arena generalization")
    if cross_rows:
        stable = [row for row in cross_rows if bool(row.get("stable"))]
        unstable = [row for row in cross_rows if not bool(row.get("stable"))]
        for label, subset, color in (
            ("stable", stable, "#59a14f"),
            ("unstable", unstable, "#e15759"),
        ):
            x = [_float(row.get("arena_size")) for row in subset]
            y = [_float(row.get("scale_ratio")) for row in subset]
            valid = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
            if valid:
                axes[0].scatter(
                    [item[0] for item in valid],
                    [item[1] for item in valid],
                    color=color,
                    alpha=0.75,
                    label=label,
                )
        axes[0].axhline(1.2, color="#6b6b6b", linestyle="--", linewidth=1.0)
        axes[0].set_xlabel("arena size (m)")
        axes[0].set_ylabel("scale ratio vs reference")
        axes[0].legend(fontsize=7)
    else:
        _no_data(axes[0])
    if module_rows:
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in module_rows:
            arena = _float(row.get("arena_size"))
            score = _float(row.get("mean_grid_score_60"))
            if arena is not None and score is not None:
                grouped[arena].append(score)
        arenas = sorted(grouped)
        if arenas:
            axes[1].plot(
                arenas,
                [float(np.mean(grouped[arena])) for arena in arenas],
                marker="o",
                color="#f28e2b",
            )
        else:
            _no_data(axes[1])
    else:
        _no_data(axes[1])
    axes[1].set_xlabel("arena size (m)")
    axes[1].set_ylabel("mean module grid score 60")
    return fig


def _plot_path_invariance(bundle: FigureDataBundle) -> plt.Figure:
    summary_rows = bundle.table("path_invariance_summary")
    bin_rows = bundle.table("path_invariance_bins")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    fig.suptitle("Path invariance")
    if summary_rows:
        labels = [
            f"{row.get('variant')}:{row.get('seed')}"
            for row in summary_rows
        ]
        scores = [_float(row.get("path_invariance_score")) for row in summary_rows]
        values = [0.0 if score is None else score for score in scores]
        axes[0].bar(range(len(values)), values, color="#4c78a8")
        axes[0].set_xticks(range(len(values)), labels, rotation=35, ha="right", fontsize=7)
        axes[0].set_ylim(0.0, 1.0)
    else:
        _no_data(axes[0])
    axes[0].set_ylabel("path invariance score")

    grouped_bins: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in bin_rows:
        variant = str(row.get("variant"))
        arena = _float(row.get("arena_size")) or 0.0
        grouped_bins[(variant, arena)].append(row)
    if grouped_bins:
        for (variant, arena), rows in sorted(grouped_bins.items()):
            rows = sorted(rows, key=lambda row: _float(row.get("bin_low")) or 0.0)
            xs = [_float(row.get("mean_spatial_distance")) for row in rows]
            ys = [_float(row.get("mean_neural_distance")) for row in rows]
            valid = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if valid:
                axes[1].plot(
                    [item[0] for item in valid],
                    [item[1] for item in valid],
                    marker="o",
                    linewidth=1.3,
                    label=f"{variant} {arena:g}m",
                )
        axes[1].legend(fontsize=7)
    else:
        _no_data(axes[1])
    axes[1].set_xlabel("spatial distance (m)")
    axes[1].set_ylabel("neural distance")
    return fig


def _plot_fourier_phase_state(bundle: FigureDataBundle) -> plt.Figure:
    fourier_rows = bundle.table("fourier_lattice_vectors")
    phase_rows = bundle.table("phase_tiling")
    state_rows = bundle.table("state_space_summary")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    fig.suptitle("Fourier, phase, and state space")
    if fourier_rows:
        for row in fourier_rows:
            x = _float(row.get("vector_x"))
            y = _float(row.get("vector_y"))
            if x is None or y is None:
                continue
            axes[0].arrow(0, 0, x, y, head_width=0.04, length_includes_head=True, alpha=0.65)
        axes[0].set_aspect("equal", adjustable="box")
    else:
        _no_data(axes[0])
    axes[0].set_xlabel("kx")
    axes[0].set_ylabel("ky")

    valid_phase = [
        (_float(row.get("phase_x")), _float(row.get("phase_y")))
        for row in phase_rows
        if _float(row.get("module_id")) is not None and int(row.get("module_id", -1)) >= 0
    ]
    valid_phase = [(x, y) for x, y in valid_phase if x is not None and y is not None]
    if valid_phase:
        axes[1].scatter(
            [item[0] for item in valid_phase],
            [item[1] for item in valid_phase],
            color="#59a14f",
            alpha=0.75,
        )
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(-0.05, 1.05)
    else:
        _no_data(axes[1])
    axes[1].set_xlabel("phase x")
    axes[1].set_ylabel("phase y")

    if state_rows:
        labels = [f"m{row.get('module_id')}" for row in state_rows]
        values = [_float(row.get("top6_explained_variance")) or 0.0 for row in state_rows]
        axes[2].bar(range(len(values)), values, color="#f28e2b")
        axes[2].set_xticks(range(len(values)), labels, rotation=35, ha="right", fontsize=7)
        axes[2].set_ylim(0.0, 1.0)
    else:
        _no_data(axes[2])
    axes[2].set_ylabel("PCA6 explained variance")
    return fig


def _plot_ablations(bundle: FigureDataBundle) -> plt.Figure:
    module_rows = bundle.table("module_summary")
    path_rows = bundle.table("path_invariance_summary")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    fig.suptitle("Ablations and seeds")
    variants = sorted({str(row.get("variant")) for row in module_rows + path_rows})
    if not variants:
        _no_data(axes[0])
        _no_data(axes[1])
        return fig
    module_counts = []
    module_scores = []
    path_scores = []
    for variant in variants:
        variant_modules = [row for row in module_rows if row.get("variant") == variant]
        module_counts.append(len(variant_modules))
        scores = [_float(row.get("mean_grid_score_60")) for row in variant_modules]
        scores = [score for score in scores if score is not None]
        module_scores.append(float(np.mean(scores)) if scores else 0.0)
        paths = [
            _float(row.get("path_invariance_score"))
            for row in path_rows
            if row.get("variant") == variant
        ]
        paths = [score for score in paths if score is not None]
        path_scores.append(float(np.mean(paths)) if paths else 0.0)
    x = np.arange(len(variants))
    axes[0].bar(x - 0.18, module_counts, width=0.36, color="#4c78a8", label="modules")
    axes[0].bar(x + 0.18, module_scores, width=0.36, color="#f28e2b", label="grid score")
    axes[0].set_xticks(x, variants, rotation=35, ha="right", fontsize=7)
    axes[0].legend(fontsize=7)
    axes[0].set_ylabel("count / score")

    axes[1].bar(x, path_scores, color="#59a14f")
    axes[1].set_xticks(x, variants, rotation=35, ha="right", fontsize=7)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("path invariance score")
    return fig


def _variant_order(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("variant")) for row in rows})


def _variant_colors(variants: list[str]) -> dict[str, str]:
    palette = ["#4c78a8", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2"]
    return {variant: palette[index % len(palette)] for index, variant in enumerate(variants)}


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _no_data(axis: plt.Axes) -> None:
    axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])
