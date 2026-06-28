from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def save_ratemap_pdf(ratemaps: np.ndarray, path: str | Path, title: str) -> None:
    _save_grid_pdf(ratemaps, path, title, cmap="viridis")


def save_sac_pdf(sacs: np.ndarray, path: str | Path, title: str) -> None:
    _save_grid_pdf(sacs, path, title, cmap="magma")


def save_summary_figure(
    ratemaps: np.ndarray,
    sacs: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_show = min(4, ratemaps.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 6.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, n_show)
    fig.suptitle(title)
    for index in range(n_show):
        axes[0, index].imshow(ratemaps[index], origin="lower", interpolation="nearest")
        axes[0, index].axis("off")
        axes[0, index].set_title(f"unit {index}")
        axes[1, index].imshow(sacs[index], origin="lower", interpolation="nearest")
        axes[1, index].axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_grid_pdf(arrays: np.ndarray, path: str | Path, title: str, cmap: str) -> None:
    arrays = np.asarray(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_page = 16
    with PdfPages(path) as pdf:
        for start in range(0, arrays.shape[0], per_page):
            chunk = arrays[start : start + per_page]
            rows = 4
            cols = 4
            fig, axes = plt.subplots(rows, cols, figsize=(10, 10), constrained_layout=True)
            fig.suptitle(title)
            for axis in axes.flat:
                axis.axis("off")
            for index, array in enumerate(chunk):
                axis = axes.flat[index]
                axis.imshow(array, origin="lower", interpolation="nearest", cmap=cmap)
                axis.axis("off")
                axis.set_title(str(start + index), fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
