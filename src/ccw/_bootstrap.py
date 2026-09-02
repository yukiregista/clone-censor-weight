"""Shared subject-level bootstrap implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class BootstrapDistribution:
    """Numerical results from subject-level bootstrap refits."""

    values: np.ndarray

    @property
    def valid_per_statistic(self) -> np.ndarray:
        return np.sum(np.isfinite(self.values), axis=0).astype(int)

    @property
    def valid_all_statistics(self) -> int:
        return int(np.sum(np.all(np.isfinite(self.values), axis=1)))

    def standard_errors(self) -> np.ndarray:
        return np.asarray(
            [bootstrap_standard_error(column) for column in self.values.T]
        )

    def basic_interval(
        self,
        point_estimates: np.ndarray,
        *,
        confidence_level: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = 1.0 - confidence_level
        lower_quantiles = np.nanquantile(self.values, alpha / 2.0, axis=0)
        upper_quantiles = np.nanquantile(
            self.values,
            1.0 - alpha / 2.0,
            axis=0,
        )
        points = np.asarray(point_estimates, dtype=float)
        return 2.0 * points - upper_quantiles, 2.0 * points - lower_quantiles

    def studentized_interval(
        self,
        *,
        point_estimate: float,
        standard_error: float,
        estimate_column: int,
        standard_error_column: int,
        confidence_level: float,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> tuple[float, float, int]:
        replicate_estimates = self.values[:, estimate_column]
        if transform is not None:
            replicate_estimates = transform(replicate_estimates)
        replicate_standard_errors = self.values[:, standard_error_column]
        with np.errstate(divide="ignore", invalid="ignore"):
            pivots = (
                replicate_estimates - point_estimate
            ) / replicate_standard_errors
        pivots = pivots[np.isfinite(pivots)]
        if not np.isfinite(standard_error) or standard_error <= 0 or not pivots.size:
            return np.nan, np.nan, int(pivots.size)
        alpha = 1.0 - confidence_level
        lower_quantile, upper_quantile = np.quantile(
            pivots,
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        return (
            float(point_estimate - upper_quantile * standard_error),
            float(point_estimate - lower_quantile * standard_error),
            int(pivots.size),
        )


def subject_level_bootstrap(
    data: pd.DataFrame,
    *,
    id_col: str,
    n_resamples: int,
    seed: int,
    statistic: Callable[[pd.DataFrame], np.ndarray],
    confidence_level: float = 0.95,
) -> BootstrapDistribution:
    """Evaluate a statistic on ordinary subject-level bootstrap samples."""

    id_to_indices = {
        subject_id: group.index.to_numpy()
        for subject_id, group in data.groupby(id_col, sort=False)
    }
    subject_ids = np.asarray(sorted(id_to_indices))
    subject_positions = np.arange(len(subject_ids), dtype=int)

    def evaluate(sampled_positions: np.ndarray) -> np.ndarray:
        sampled_ids = subject_ids[np.asarray(sampled_positions, dtype=int)]
        row_indices = np.concatenate(
            [id_to_indices[subject_id] for subject_id in sampled_ids]
        )
        new_ids = np.concatenate(
            [
                np.full(len(id_to_indices[subject_id]), new_id, dtype=int)
                for new_id, subject_id in enumerate(sampled_ids)
            ]
        )
        sample = data.loc[row_indices].copy()
        sample[id_col] = new_ids
        return np.asarray(statistic(sample), dtype=float)

    result = stats.bootstrap(
        data=(subject_positions,),
        statistic=evaluate,
        vectorized=False,
        n_resamples=n_resamples,
        random_state=np.random.default_rng(seed),
        confidence_level=confidence_level,
        method="basic",
    )
    values = np.asarray(result.bootstrap_distribution, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    return BootstrapDistribution(values=values.T)


def bootstrap_standard_error(values: np.ndarray) -> float:
    """Calculate a standard error from finite bootstrap estimates."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=1)) if finite.size >= 2 else float("nan")
