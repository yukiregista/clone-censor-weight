"""Internal implementation of fitted-weight diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .run_diagnosis import diagnose_dataframe as _diagnose_dataframe


def _diagnose_weights(
    diag_df: pd.DataFrame,
    *,
    patterns: Iterable[str] = ("UNW", "NAT", "VAR", "HPREV2"),
    by_regime: bool = True,
    by_run: bool = True,
    max_time: int | None = None,
    eps: float = 1e-12,
    min_m: int = 200,
    min_k_abs: int = 20,
    min_k_frac: float = 0.01,
    max_k_frac: float = 0.10,
    k_points: int = 20,
    tail_index_selector: str = "plateau",
    reiss_thomas_beta: float = 0.3,
    weight_cap_quantile: float | None = None,
    alpha_borderline: float = 1.2,
    alpha_bad: float = 1.0,
    tail_tiny: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute time-specific tail diagnostics for censoring weights.

    Parameters
    ----------
    diag_df : pandas.DataFrame
        Row-level weight data. The table must identify subject, time,
        conditional censoring probability ``G_t``, and prior cumulative
        probability ``H_prev``. Subject may be named ``"id"`` or
        ``"subject_id"``; time may be named ``"t"``, ``"tstart"``, or
        ``"time"``. If ``S_t`` is absent, all rows are treated as at risk. For
        fitted results, prefer :meth:`ccw.CCWResult.weight_diagnostics`, which
        handles user-defined column names automatically.
    patterns : iterable of str, default=("UNW", "NAT", "VAR", "HPREV2")
        Tail-weighting patterns to evaluate. Supported values are ``"UNW"``,
        ``"NAT"``, ``"VAR"``, ``"DIRECT"``, and ``"HPREV2"``.
    by_regime : bool, default=True
        Group diagnostics by treatment-strategy arm when an arm column is
        present.
    by_run : bool, default=True
        Group diagnostics by ``run_id`` when that column is present.
    max_time : int, optional
        Largest analysis time to include.
    eps : float, default=1e-12
        Lower clipping bound for probability values.
    min_m : int, default=200
        Minimum number of valid observations required for tail estimation in
        a group.
    min_k_abs : int, default=20
        Absolute lower bound for candidate tail sizes.
    min_k_frac : float, default=0.01
        Sample-size fraction used as an additional lower bound for candidate
        tail sizes.
    max_k_frac : float, default=0.10
        Sample-size fraction used as the upper bound for candidate tail sizes.
    k_points : int, default=20
        Number of tail sizes reported across the candidate range.
    tail_index_selector : {"plateau", "reiss_thomas", "median"}, default="plateau"
        Rule used to select or aggregate the Hill tail-index path.
    reiss_thomas_beta : float, default=0.3
        Exponent used by the Reiss--Thomas selection criterion.
    weight_cap_quantile : float, optional
        Quantile at which diagnostic weights are capped before estimation.
    alpha_borderline : float, default=1.2
        Tail-index threshold below which a group is flagged as borderline.
    alpha_bad : float, default=1.0
        Tail-index threshold below which a group is flagged as problematic.
    tail_tiny : float, default=1e-8
        Tolerance used to identify groups with a negligible upper tail.

    Returns
    -------
    summary : pandas.DataFrame
        One row per time, group, and pattern with selected tail indices,
        effective sample sizes, status values, and flags.
    detail : pandas.DataFrame
        Hill estimates and effective sample sizes across candidate tail sizes.

    Raises
    ------
    KeyError
        If required diagnostic columns cannot be identified.
    ValueError
        If a requested pattern or tail-index selector is unsupported.
    """

    return _diagnose_dataframe(
        diag_df,
        patterns=patterns,
        by_regime=by_regime,
        by_run=by_run,
        max_time=max_time,
        eps=eps,
        min_m=min_m,
        min_k_abs=min_k_abs,
        min_k_frac=min_k_frac,
        max_k_frac=max_k_frac,
        k_points=k_points,
        tail_index_selector=tail_index_selector,
        reiss_thomas_beta=reiss_thomas_beta,
        weight_cap_quantile=weight_cap_quantile,
        alpha_borderline=alpha_borderline,
        alpha_bad=alpha_bad,
        tail_tiny=tail_tiny,
    )

__all__: list[str] = []
