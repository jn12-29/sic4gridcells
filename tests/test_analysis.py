from __future__ import annotations

import numpy as np

from sic4gridcells.analysis import GridScorer
import sic4gridcells.analysis as analysis_module


def test_ratemap_uses_nan_for_empty_bins() -> None:
    scorer = GridScorer(4, [[0.0, 1.0], [0.0, 1.0]])
    positions = np.array([[[0.1, 0.1], [0.2, 0.2]]])
    activations = np.array([[[1.0], [3.0]]])
    ratemaps = np.zeros((1, 4, 4), dtype=np.float64)
    counts = np.zeros((4, 4), dtype=np.int64)
    scorer.accumulate_ratemaps(positions, activations, ratemaps, counts)
    finalized = scorer.finalize_ratemaps(ratemaps, counts)
    assert np.isnan(finalized).any()
    assert np.isfinite(finalized[~np.isnan(finalized)]).all()


def test_grid_scores_and_scales_shapes_are_consistent() -> None:
    scorer = GridScorer(8, [[-1.0, 1.0], [-1.0, 1.0]])
    ratemap = np.zeros((8, 8), dtype=np.float64)
    ratemap[2, 2] = 1.0
    ratemap[5, 5] = 0.5
    result = scorer.get_scores_batch(np.stack([ratemap, ratemap], axis=0))
    assert result.sacs.shape == (2, 15, 15)
    assert result.score_60.shape == (2,)
    assert result.score_90.shape == (2,)
    scales, peak_counts = scorer.calculate_grid_scales(result.sacs)
    assert scales.shape == (2,)
    assert peak_counts.shape == (2,)
    metrics = scorer.calculate_grid_metrics(result.sacs)
    assert metrics.scale_pixels.shape == (2,)
    assert metrics.orientation_degrees.shape == (2,)
    assert metrics.peak_counts.shape == (2,)


def test_grid_scale_peaks_report_angles_and_distances() -> None:
    scorer = GridScorer(4, [[-1.0, 1.0], [-1.0, 1.0]])
    sac = np.zeros((7, 7), dtype=np.float64)
    sac[3, 5] = 1.0
    sac[5, 3] = 0.9

    peaks = scorer.grid_scale_peaks(sac, exclude_center_radius=1.0)

    assert peaks.distances.shape == (2,)
    assert peaks.angles_degrees.shape == (2,)
    assert np.all(peaks.distances > 1.0)


def test_sac_masks_nan_ratemap_bins() -> None:
    scorer = GridScorer(2, [[0.0, 1.0], [0.0, 1.0]])
    sparse = np.array([[1.0, np.nan], [np.nan, 3.0]], dtype=np.float64)
    zero_filled = np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float64)

    sparse_sac = scorer.calculate_sac(sparse)
    zero_filled_sac = scorer.calculate_sac(zero_filled)

    assert sparse_sac[1, 1] > 0.99
    assert sparse_sac[1, 0] == 0.0
    assert sparse_sac[0, 1] == 0.0
    assert not np.allclose(sparse_sac, zero_filled_sac)


def test_grid_scorer_source_attribution_is_retained() -> None:
    source = analysis_module.__loader__.get_source(analysis_module.__name__)
    assert source is not None
    assert "grid-cells-torch" in source
    assert "SPDX-License-Identifier: Apache-2.0" in source
    assert "Apache License, Version 2.0" in source
    assert "Copyright 2018 Google LLC" in source
