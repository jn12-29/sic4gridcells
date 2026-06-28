from __future__ import annotations

# Adapted from grid-cells-torch/grid_cells/analysis/scores.py:
# https://github.com/rylan-schaeffer/grid-cells-torch/blob/main/grid_cells/analysis/scores.py
#
# Original copyright and license notice:
# Copyright 2018 Google LLC
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes in this repository:
# - reduced the implementation to NumPy/SciPy utilities needed by SIC evaluation
# - kept ratemap NaN empty-bin semantics and batch SAC/grid-score behavior
# - removed plotting methods not used by the current evaluation CLI

from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.signal


@dataclass(frozen=True)
class GridScoreResult:
    score_60: np.ndarray
    score_90: np.ndarray
    mask_60: list[tuple[float, float]]
    mask_90: list[tuple[float, float]]
    sacs: np.ndarray


@dataclass(frozen=True)
class GridMetrics:
    scale_pixels: np.ndarray
    orientation_degrees: np.ndarray
    peak_counts: np.ndarray


class GridScorer:
    def __init__(
        self,
        nbins: int,
        coords_range: list[list[float]],
        mask_parameters: list[tuple[float, float]] | None = None,
        min_max: bool = False,
    ) -> None:
        self._nbins = nbins
        self._coords_range = coords_range
        self._min_max = min_max
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        self._angle_to_index = {angle: index for index, angle in enumerate(self._corr_angles)}
        self._mask_params = mask_parameters or [(0.2, 0.35), (0.2, 0.45), (0.3, 0.55)]
        self._masks = [
            (self._get_ring_mask(mask_min, mask_max), (mask_min, mask_max))
            for mask_min, mask_max in self._mask_params
        ]
        self._mask_stack = np.asarray([mask for mask, _ in self._masks], dtype=np.float64)
        self._mask_ring_areas = np.sum(self._mask_stack, axis=(1, 2))
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan,
        )

    def calculate_ratemap(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        activations: np.ndarray,
        statistic: str = "mean",
    ) -> np.ndarray:
        del statistic
        ratemap, _, _ = np.histogram2d(
            xs,
            ys,
            bins=self._nbins,
            range=self._coords_range,
            weights=activations,
        )
        counts, _, _ = np.histogram2d(
            xs,
            ys,
            bins=self._nbins,
            range=self._coords_range,
        )
        output = np.full_like(ratemap, np.nan, dtype=np.float64)
        valid = counts > 0
        np.divide(ratemap, counts, out=output, where=valid)
        return output

    def accumulate_ratemaps(
        self,
        positions: np.ndarray,
        activations: np.ndarray,
        sums: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        flat_pos = positions.reshape(-1, positions.shape[-1])
        flat_act = activations.reshape(-1, activations.shape[-1])
        x_idx, y_idx, valid = self._digitize_positions(flat_pos[:, 0], flat_pos[:, 1])
        if not np.any(valid):
            return
        np.add.at(counts, (x_idx, y_idx), 1)
        for unit in range(flat_act.shape[-1]):
            np.add.at(sums[unit], (x_idx, y_idx), flat_act[valid, unit])

    def finalize_ratemaps(self, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
        ratemaps = np.full(sums.shape, np.nan, dtype=sums.dtype)
        valid = counts > 0
        np.divide(sums, counts[None, :, :], out=ratemaps, where=valid[None, :, :])
        return ratemaps

    def calculate_sac(self, seq1: np.ndarray) -> np.ndarray:
        return self.calculate_sac_batch(np.asarray(seq1)[np.newaxis, ...])[0]

    def calculate_sac_batch(self, ratemaps: np.ndarray) -> np.ndarray:
        ratemaps = np.asarray(ratemaps)
        if ratemaps.ndim == 2:
            ratemaps = ratemaps[np.newaxis, ...]
        finite = np.isfinite(ratemaps)
        seq = np.where(finite, ratemaps, 0.0).astype(np.float64, copy=False)
        mask = finite.astype(np.float64, copy=False)
        seq_sq = np.square(seq)
        fft_shape = tuple(2 * size - 1 for size in seq.shape[-2:])
        fs = np.fft.rfft2(seq, s=fft_shape, axes=(-2, -1))
        fm = np.fft.rfft2(mask, s=fft_shape, axes=(-2, -1))
        fss = np.fft.rfft2(seq_sq, s=fft_shape, axes=(-2, -1))

        def corr(fx: np.ndarray, fb: np.ndarray) -> np.ndarray:
            return np.fft.fftshift(
                np.fft.irfft2(fx * np.conj(fb), s=fft_shape, axes=(-2, -1)),
                axes=(-2, -1),
            )

        seq1_x_seq2 = np.real(corr(fs, fs))
        sum_seq1 = np.real(corr(fm, fs))
        sum_seq2 = np.real(corr(fs, fm))
        sum_seq1_sq = np.real(corr(fm, fss))
        sum_seq2_sq = np.real(corr(fss, fm))
        n_bins = np.real(corr(fm, fm))
        n_bins[np.abs(n_bins) < 1e-8] = 0.0
        valid_overlap = n_bins > 0.0
        mean_seq1 = np.divide(
            sum_seq1,
            n_bins,
            out=np.zeros_like(sum_seq1),
            where=valid_overlap,
        )
        mean_seq2 = np.divide(
            sum_seq2,
            n_bins,
            out=np.zeros_like(sum_seq2),
            where=valid_overlap,
        )
        var_seq1 = np.subtract(
            np.divide(sum_seq1_sq, n_bins, out=np.zeros_like(sum_seq1_sq), where=valid_overlap),
            np.square(mean_seq1),
        )
        var_seq2 = np.subtract(
            np.divide(sum_seq2_sq, n_bins, out=np.zeros_like(sum_seq2_sq), where=valid_overlap),
            np.square(mean_seq2),
        )
        std_seq1 = np.sqrt(np.maximum(var_seq1, 0.0))
        std_seq2 = np.sqrt(np.maximum(var_seq2, 0.0))
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins, out=np.zeros_like(seq1_x_seq2), where=valid_overlap),
            mean_seq1 * mean_seq2,
        )
        denominator = std_seq1 * std_seq2
        x_coef = np.divide(
            covar,
            denominator,
            out=np.zeros_like(covar),
            where=(denominator > 1e-8) & valid_overlap,
        )
        x_coef = np.nan_to_num(x_coef)
        x_coef[np.abs(x_coef) < 2e-6] = 0.0
        return x_coef

    def grid_score_60(self, corr: dict[int, np.ndarray]) -> np.ndarray:
        if self._min_max:
            return np.minimum(corr[60], corr[120]) - np.maximum(
                corr[30], np.maximum(corr[90], corr[150])
            )
        return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    def grid_score_90(self, corr: dict[int, np.ndarray]) -> np.ndarray:
        return corr[90] - (corr[45] + corr[135]) / 2

    def get_scores_batch(self, ratemaps: np.ndarray) -> GridScoreResult:
        ratemaps = np.asarray(ratemaps)
        if ratemaps.ndim == 2:
            ratemaps = ratemaps[np.newaxis, ...]
        sacs = self.calculate_sac_batch(ratemaps)
        rotated_sacs = self.rotated_sacs_batch(sacs, self._corr_angles)
        mask_stack = self._mask_stack[None, :, None, :, :]
        ring_areas = self._mask_ring_areas[None, :, None]
        masked_sacs = sacs[:, None, :, :] * self._mask_stack[None, :, :, :]
        masked_sac_means = np.sum(masked_sacs, axis=(-1, -2)) / self._mask_ring_areas[None, :]
        masked_sac_centered = (
            masked_sacs - masked_sac_means[:, :, None, None]
        )[:, :, None, :, :] * mask_stack
        variance = np.sum(masked_sac_centered**2, axis=(-1, -2)) / ring_areas + 1e-5
        masked_rotated_sacs = (
            rotated_sacs[:, None, :, :, :] - masked_sac_means[:, :, None, None, None]
        ) * mask_stack
        cross_prod = np.sum(masked_sac_centered * masked_rotated_sacs, axis=(-1, -2)) / ring_areas
        corrs = cross_prod / variance

        corr = {angle: corrs[:, :, self._angle_to_index[angle]] for angle in self._corr_angles}
        score_60 = self.grid_score_60(corr)
        score_90 = self.grid_score_90(corr)
        max_60_ind = np.argmax(score_60, axis=1)
        max_90_ind = np.argmax(score_90, axis=1)
        unit_indices = np.arange(ratemaps.shape[0])
        return GridScoreResult(
            score_60=score_60[unit_indices, max_60_ind],
            score_90=score_90[unit_indices, max_90_ind],
            mask_60=[self._mask_params[index] for index in max_60_ind],
            mask_90=[self._mask_params[index] for index in max_90_ind],
            sacs=sacs,
        )

    def calculate_grid_scale(
        self,
        sac: np.ndarray,
        n_peaks: int = 6,
        exclude_center_radius: float = 1.0,
        min_peak_value: float = 0.0,
    ) -> tuple[float, int]:
        peaks = self.grid_scale_peaks(
            sac,
            exclude_center_radius=exclude_center_radius,
            min_peak_value=min_peak_value,
        )
        peak_distances = peaks.distances
        if peak_distances.size < n_peaks:
            return np.nan, int(peak_distances.size)
        return float(np.median(peak_distances[:n_peaks])), int(peak_distances.size)

    def calculate_grid_metrics(
        self,
        sacs: np.ndarray,
        n_peaks: int = 6,
        exclude_center_radius: float = 1.0,
        min_peak_value: float = 0.0,
    ) -> GridMetrics:
        sacs = np.asarray(sacs)
        if sacs.ndim == 2:
            sacs = sacs[np.newaxis, ...]
        scales = np.full(sacs.shape[0], np.nan, dtype=np.float64)
        orientations = np.full(sacs.shape[0], np.nan, dtype=np.float64)
        peak_counts = np.zeros(sacs.shape[0], dtype=np.int64)
        for index, sac in enumerate(sacs):
            peaks = self.grid_scale_peaks(
                sac,
                exclude_center_radius=exclude_center_radius,
                min_peak_value=min_peak_value,
            )
            peak_counts[index] = peaks.distances.size
            if peaks.distances.size < n_peaks:
                continue
            selected_distances = peaks.distances[:n_peaks]
            selected_angles = peaks.angles_degrees[:n_peaks]
            scales[index] = float(np.median(selected_distances))
            orientations[index] = _periodic_mean_degrees(selected_angles, period=60.0)
        return GridMetrics(
            scale_pixels=scales,
            orientation_degrees=orientations,
            peak_counts=peak_counts,
        )

    def calculate_grid_scales(
        self,
        sacs: np.ndarray,
        n_peaks: int = 6,
        exclude_center_radius: float = 1.0,
        min_peak_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        sacs = np.asarray(sacs)
        if sacs.ndim == 2:
            sacs = sacs[np.newaxis, ...]
        scales = np.full(sacs.shape[0], np.nan, dtype=np.float64)
        peak_counts = np.zeros(sacs.shape[0], dtype=np.int64)
        for index, sac in enumerate(sacs):
            scale, peak_count = self.calculate_grid_scale(
                sac,
                n_peaks=n_peaks,
                exclude_center_radius=exclude_center_radius,
                min_peak_value=min_peak_value,
            )
            scales[index] = scale
            peak_counts[index] = peak_count
        return scales, peak_counts

    def grid_scale_peak_distances(
        self,
        sac: np.ndarray,
        exclude_center_radius: float = 1.0,
        min_peak_value: float = 0.0,
    ) -> np.ndarray:
        return self.grid_scale_peaks(
            sac,
            exclude_center_radius=exclude_center_radius,
            min_peak_value=min_peak_value,
        ).distances

    def grid_scale_peaks(
        self,
        sac: np.ndarray,
        exclude_center_radius: float = 1.0,
        min_peak_value: float = 0.0,
    ) -> "SacPeaks":
        sac = np.asarray(sac, dtype=np.float64)
        filled = np.where(np.isfinite(sac), sac, -np.inf)
        local_max = filled == scipy.ndimage.maximum_filter(
            filled,
            size=3,
            mode="constant",
            cval=-np.inf,
        )
        local_max &= np.isfinite(sac)
        local_max &= sac > float(min_peak_value)
        center = np.asarray([(sac.shape[0] - 1) / 2.0, (sac.shape[1] - 1) / 2.0])
        coords = np.argwhere(local_max)
        if coords.size == 0:
            return SacPeaks.empty()
        distances = np.linalg.norm(coords - center[np.newaxis, :], axis=1)
        keep = distances > float(exclude_center_radius)
        coords = coords[keep]
        distances = distances[keep]
        if distances.size == 0:
            return SacPeaks.empty()
        values = sac[coords[:, 0], coords[:, 1]]
        order = np.lexsort((-values, distances))
        coords = coords[order]
        distances = distances[order]
        values = values[order]
        deltas = coords.astype(np.float64) - center[np.newaxis, :]
        angles = np.degrees(np.arctan2(deltas[:, 0], deltas[:, 1]))
        return SacPeaks(
            coords=coords,
            distances=distances,
            angles_degrees=np.mod(angles, 180.0),
            values=values,
        )

    def rotated_sacs(self, sac: np.ndarray, angles: list[int]) -> list[np.ndarray]:
        return [scipy.ndimage.rotate(sac, angle, reshape=False) for angle in angles]

    def rotated_sacs_batch(self, sacs: np.ndarray, angles: list[int]) -> np.ndarray:
        return np.stack(
            [scipy.ndimage.rotate(sacs, angle, axes=(1, 2), reshape=False) for angle in angles],
            axis=1,
        )

    def _get_ring_mask(self, mask_min: float, mask_max: float) -> np.ndarray:
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return circle_mask(n_points, mask_max * self._nbins) * (
            1 - circle_mask(n_points, mask_min * self._nbins)
        )

    def _digitize_positions(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_edges = np.linspace(self._coords_range[0][0], self._coords_range[0][1], self._nbins + 1)
        y_edges = np.linspace(self._coords_range[1][0], self._coords_range[1][1], self._nbins + 1)
        x_idx = np.searchsorted(x_edges, xs, side="right") - 1
        y_idx = np.searchsorted(y_edges, ys, side="right") - 1
        x_idx[xs == x_edges[-1]] = self._nbins - 1
        y_idx[ys == y_edges[-1]] = self._nbins - 1
        valid = (
            (x_idx >= 0)
            & (x_idx < self._nbins)
            & (y_idx >= 0)
            & (y_idx < self._nbins)
        )
        return x_idx[valid], y_idx[valid], valid


def circle_mask(size: list[int], radius: float, in_val: float = 1.0, out_val: float = 0.0) -> np.ndarray:
    sz = [size[0] // 2, size[1] // 2]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0).repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1).repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    return np.where(z, in_val, out_val)


@dataclass(frozen=True)
class SacPeaks:
    coords: np.ndarray
    distances: np.ndarray
    angles_degrees: np.ndarray
    values: np.ndarray

    @staticmethod
    def empty() -> "SacPeaks":
        return SacPeaks(
            coords=np.empty((0, 2), dtype=np.int64),
            distances=np.empty(0, dtype=np.float64),
            angles_degrees=np.empty(0, dtype=np.float64),
            values=np.empty(0, dtype=np.float64),
        )


def _periodic_mean_degrees(values: np.ndarray, period: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    angles = values[finite] / period * 2.0 * np.pi
    mean = np.angle(np.mean(np.exp(1j * angles)))
    return float(np.mod(mean / (2.0 * np.pi) * period, period))
