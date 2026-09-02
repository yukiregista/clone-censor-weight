from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

TAIL_INDEX_SELECTORS = {"median", "plateau", "reiss_thomas"}


def build_k_grid(
    m: int,
    min_k_abs: int = 20,
    min_frac: float = 0.01,
    max_frac: float = 0.10,
    n_points: int = 20,
) -> list[int]:
    """
    Build a robust Hill tail-size grid.
    """
    if m < 2:
        return []

    k_min = max(int(min_k_abs), int(np.floor(min_frac * m)))
    k_max = min(int(np.floor(max_frac * m)), m - 1)
    if k_max < k_min:
        return []

    if n_points <= 1 or k_min == k_max:
        return [k_min]

    raw = np.linspace(k_min, k_max, num=n_points)
    grid = np.unique(np.rint(raw).astype(int))
    grid = grid[(grid >= k_min) & (grid <= k_max)]
    return grid.tolist()


def _empty_hill_result(m: int, selector: str) -> dict:
    empty_float = np.asarray([], dtype=float)
    empty_int = np.asarray([], dtype=int)
    return {
        "m": m,
        "k_values": empty_int,
        "gamma_by_k": empty_float,
        "alpha_by_k": empty_float,
        "ess_by_k": empty_float,
        "alpha_med": np.nan,
        "alpha_iqr": np.nan,
        "alpha_min": np.nan,
        "alpha_max": np.nan,
        "tail_index": np.nan,
        "selector": selector,
        "k_selected": np.nan,
        "gamma_selected": np.nan,
        "alpha_selected": np.nan,
        "ess_selected": np.nan,
        "selector_score": np.nan,
        "selector_status": "skip_insufficient_sample",
    }


def _weighted_hill_path_from_sorted(
    z_sorted: np.ndarray,
    w_sorted: np.ndarray,
    k_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k_arr = np.asarray(k_values, dtype=int)
    gamma = np.full(k_arr.shape, np.nan, dtype=float)
    ess = np.full(k_arr.shape, np.nan, dtype=float)
    if k_arr.size == 0:
        return gamma, ess

    valid_k = (k_arr > 0) & (k_arr < z_sorted.size)
    if not np.any(valid_k):
        return gamma, ess

    log_z = np.log(z_sorted)
    cum_w = np.cumsum(w_sorted)
    cum_w2 = np.cumsum(w_sorted * w_sorted)
    cum_w_log_z = np.cumsum(w_sorted * log_z)

    top_idx = k_arr[valid_k] - 1
    threshold_idx = k_arr[valid_k]
    W_k = cum_w[top_idx]
    W2_k = cum_w2[top_idx]
    threshold_log = log_z[threshold_idx]
    numerator = cum_w_log_z[top_idx] - W_k * threshold_log

    gamma_use = numerator / W_k
    ess_use = (W_k * W_k) / W2_k
    finite = (
        np.isfinite(gamma_use)
        & (gamma_use > 0)
        & np.isfinite(ess_use)
        & (ess_use > 0)
    )

    valid_positions = np.flatnonzero(valid_k)
    gamma[valid_positions[finite]] = gamma_use[finite]
    ess[valid_positions[finite]] = ess_use[finite]
    return gamma, ess


def _select_reiss_thomas(
    z_sorted: np.ndarray,
    w_sorted: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    beta: float,
) -> dict:
    if k_min <= 0 or k_max < k_min or k_max >= z_sorted.size:
        return {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "skip_invalid_k_range",
        }

    path_k = np.arange(1, k_max + 1, dtype=int)
    gamma_path, ess_path = _weighted_hill_path_from_sorted(z_sorted, w_sorted, path_k)
    valid = np.isfinite(gamma_path) & (gamma_path > 0) & np.isfinite(ess_path) & (ess_path > 0)
    if not np.any(valid[k_min - 1 : k_max]):
        return {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "skip_no_finite_candidates",
        }

    # Reiss-Thomas style criterion: choose the lowest admissible k minimizing
    # cumulative instability of the Hill path up to k. For unit weights,
    # ESS_i = i, so this reduces to the usual i^beta weighting.
    path_weight = np.where(valid, np.power(ess_path, beta), 0.0)
    gamma_clean = np.where(valid, gamma_path, 0.0)
    cum_w = np.cumsum(path_weight)
    cum_w_gamma = np.cumsum(path_weight * gamma_clean)
    cum_w_gamma2 = np.cumsum(path_weight * gamma_clean * gamma_clean)
    with np.errstate(invalid="ignore", divide="ignore"):
        instability = (
            cum_w_gamma2
            - 2.0 * gamma_path * cum_w_gamma
            + gamma_path * gamma_path * cum_w
        ) / path_k

    candidate_idx = np.arange(k_min - 1, k_max, dtype=int)
    candidate_mask = valid[candidate_idx] & np.isfinite(instability[candidate_idx])
    if not np.any(candidate_mask):
        return {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "skip_no_finite_score",
        }

    candidate_idx = candidate_idx[candidate_mask]
    scores = instability[candidate_idx]
    best_pos = int(np.nanargmin(scores))
    best_idx = int(candidate_idx[best_pos])
    gamma_selected = float(gamma_path[best_idx])
    return {
        "k_selected": int(path_k[best_idx]),
        "gamma_selected": gamma_selected,
        "alpha_selected": float(1.0 / gamma_selected),
        "ess_selected": float(ess_path[best_idx]),
        "selector_score": float(instability[best_idx]),
        "selector_status": "ok",
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    q_arr = np.asarray(quantiles, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(q_arr.shape, np.nan, dtype=float)

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0:
        return np.full(q_arr.shape, np.nan, dtype=float)

    cumulative = np.cumsum(weights) - 0.5 * weights
    probs = cumulative / total_weight
    return np.interp(np.clip(q_arr, 0.0, 1.0), probs, values, left=values[0], right=values[-1])


def _weighted_slope(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(valid) < 2:
        return np.nan

    x_use = x[valid]
    y_use = y[valid]
    w_use = weights[valid]
    weight_sum = float(np.sum(w_use))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return np.nan

    x_bar = float(np.sum(w_use * x_use) / weight_sum)
    y_bar = float(np.sum(w_use * y_use) / weight_sum)
    x_centered = x_use - x_bar
    denom = float(np.sum(w_use * x_centered * x_centered))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sum(w_use * x_centered * (y_use - y_bar)) / denom)


def _select_plateau_from_path(
    k_values: np.ndarray,
    gamma_path: np.ndarray,
    ess_path: np.ndarray,
    *,
    window_frac: float,
    min_window: int,
    score_tolerance: float,
) -> dict:
    valid = (
        np.isfinite(k_values)
        & (k_values > 0)
        & np.isfinite(gamma_path)
        & (gamma_path > 0)
        & np.isfinite(ess_path)
        & (ess_path > 0)
    )
    if not np.any(valid):
        return {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "skip_no_finite_candidates",
        }

    candidate_k = k_values[valid].astype(int)
    candidate_gamma = gamma_path[valid].astype(float)
    candidate_ess = ess_path[valid].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        candidate_alpha = 1.0 / candidate_gamma
    finite_alpha = np.isfinite(candidate_alpha) & (candidate_alpha > 0)
    candidate_k = candidate_k[finite_alpha]
    candidate_gamma = candidate_gamma[finite_alpha]
    candidate_ess = candidate_ess[finite_alpha]
    candidate_alpha = candidate_alpha[finite_alpha]

    n_candidates = int(candidate_k.size)
    if n_candidates == 0:
        return {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "skip_no_finite_candidates",
        }
    if n_candidates < 3:
        selected_pos = int(np.nanargmax(candidate_ess))
        gamma_selected = float(candidate_gamma[selected_pos])
        return {
            "k_selected": int(candidate_k[selected_pos]),
            "gamma_selected": gamma_selected,
            "alpha_selected": float(1.0 / gamma_selected),
            "ess_selected": float(candidate_ess[selected_pos]),
            "selector_score": np.nan,
            "selector_status": "fallback_insufficient_plateau_candidates",
        }

    window_size = max(int(np.ceil(window_frac * n_candidates)), int(min_window), 3)
    window_size = min(window_size, n_candidates)

    scores: list[float] = []
    starts: list[int] = []
    for start in range(n_candidates - window_size + 1):
        stop = start + window_size
        k_win = candidate_k[start:stop].astype(float)
        alpha_win = candidate_alpha[start:stop]
        ess_win = candidate_ess[start:stop]
        q25, alpha_center, q75 = _weighted_quantile(alpha_win, ess_win, [0.25, 0.5, 0.75])
        if not np.isfinite(alpha_center) or alpha_center <= 0:
            continue

        scale = max(abs(alpha_center), np.finfo(float).eps)
        rel_iqr = float((q75 - q25) / scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha_se = alpha_win / np.sqrt(ess_win)
        noise_floor = _weighted_quantile(alpha_se, ess_win, [0.5])[0]
        rel_noise_floor = (
            float(noise_floor / scale)
            if np.isfinite(noise_floor) and noise_floor > 0
            else 0.0
        )

        x = np.log(k_win)
        x_range = float(x[-1] - x[0])
        if x_range > 0:
            slope = _weighted_slope(x, alpha_win, ess_win)
            rel_slope = abs(slope) * x_range / scale if np.isfinite(slope) else 0.0
        else:
            rel_slope = 0.0

        score = max(0.0, rel_iqr - rel_noise_floor) + rel_slope
        if np.isfinite(score):
            starts.append(start)
            scores.append(float(score))

    if not scores:
        return {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "skip_no_finite_score",
        }

    score_arr = np.asarray(scores, dtype=float)
    start_arr = np.asarray(starts, dtype=int)
    best_score = float(np.nanmin(score_arr))
    max_near_best = best_score * (1.0 + float(score_tolerance))
    if best_score == 0.0:
        max_near_best = np.finfo(float).eps
    eligible = np.flatnonzero(score_arr <= max_near_best)
    best_window_pos = int(eligible[0])
    best_start = int(start_arr[best_window_pos])
    best_stop = best_start + window_size

    alpha_win = candidate_alpha[best_start:best_stop]
    ess_win = candidate_ess[best_start:best_stop]
    alpha_center = float(_weighted_quantile(alpha_win, ess_win, [0.5])[0])
    local_positions = np.arange(best_start, best_stop, dtype=float)
    window_mid = (best_start + best_stop - 1) / 2.0
    order = np.lexsort(
        (
            np.abs(local_positions - window_mid),
            np.abs(alpha_win - alpha_center),
        )
    )
    selected_pos = int(best_start + order[0])
    gamma_selected = float(candidate_gamma[selected_pos])
    return {
        "k_selected": int(candidate_k[selected_pos]),
        "gamma_selected": gamma_selected,
        "alpha_selected": float(1.0 / gamma_selected),
        "ess_selected": float(candidate_ess[selected_pos]),
        "selector_score": float(score_arr[best_window_pos]),
        "selector_status": "ok",
    }


def weighted_hill(
    z: np.ndarray | Sequence[float],
    w: np.ndarray | Sequence[float],
    k_grid: Sequence[int],
    *,
    tail_index_selector: str = "plateau",
    reiss_thomas_beta: float = 0.3,
    plateau_window_frac: float = 0.20,
    plateau_min_window: int = 5,
    plateau_score_tolerance: float = 0.10,
    plateau_max_candidates: int = 1000,
) -> dict:
    """
    Weighted Hill estimator over a grid of k values.

    Returns
    -------
    dict with keys:
      - m
      - k_values
      - gamma_by_k
      - alpha_by_k
      - ess_by_k
      - alpha_med / alpha_iqr / alpha_min / alpha_max
      - tail_index and selector diagnostics
    """
    selector = str(tail_index_selector).lower().replace("-", "_")
    if selector not in TAIL_INDEX_SELECTORS:
        allowed = ", ".join(sorted(TAIL_INDEX_SELECTORS))
        raise ValueError(f"Unknown tail_index_selector '{tail_index_selector}'. Allowed: {allowed}.")
    if not np.isfinite(reiss_thomas_beta) or reiss_thomas_beta < 0:
        raise ValueError("reiss_thomas_beta must be a finite non-negative number.")
    if not np.isfinite(plateau_window_frac) or not (0.0 < plateau_window_frac <= 1.0):
        raise ValueError("plateau_window_frac must be in (0, 1].")
    if int(plateau_min_window) < 3:
        raise ValueError("plateau_min_window must be at least 3.")
    if not np.isfinite(plateau_score_tolerance) or plateau_score_tolerance < 0:
        raise ValueError("plateau_score_tolerance must be a finite non-negative number.")
    if int(plateau_max_candidates) < int(plateau_min_window):
        raise ValueError("plateau_max_candidates must be at least plateau_min_window.")

    z_arr = np.asarray(z, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    if z_arr.shape != w_arr.shape:
        raise ValueError("z and w must have the same shape.")

    valid = np.isfinite(z_arr) & np.isfinite(w_arr) & (z_arr > 0) & (w_arr > 0)
    z_use = z_arr[valid]
    w_use = w_arr[valid]
    m = int(z_use.size)
    if m < 2:
        return _empty_hill_result(m, selector)

    order = np.argsort(-z_use)
    z_sorted = z_use[order]
    w_sorted = w_use[order]

    k_values = sorted({int(k) for k in k_grid if 1 <= int(k) < m})
    k_arr = np.asarray(k_values, dtype=int)
    gamma_arr, ess_arr = _weighted_hill_path_from_sorted(z_sorted, w_sorted, k_arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha_arr = 1.0 / gamma_arr
    alpha_arr[~np.isfinite(alpha_arr)] = np.nan
    finite_alpha = alpha_arr[np.isfinite(alpha_arr)]

    if finite_alpha.size == 0:
        alpha_med = np.nan
        alpha_iqr = np.nan
        alpha_min = np.nan
        alpha_max = np.nan
    else:
        alpha_med = float(np.nanmedian(finite_alpha))
        q75, q25 = np.nanquantile(finite_alpha, [0.75, 0.25])
        alpha_iqr = float(q75 - q25)
        alpha_min = float(np.nanmin(finite_alpha))
        alpha_max = float(np.nanmax(finite_alpha))

    if selector == "median":
        tail_index = alpha_med
        selection = {
            "k_selected": np.nan,
            "gamma_selected": np.nan,
            "alpha_selected": np.nan,
            "ess_selected": np.nan,
            "selector_score": np.nan,
            "selector_status": "median",
        }
    elif selector == "plateau":
        if k_arr.size == 0:
            selection = {
                "k_selected": np.nan,
                "gamma_selected": np.nan,
                "alpha_selected": np.nan,
                "ess_selected": np.nan,
                "selector_score": np.nan,
                "selector_status": "skip_empty_k_grid",
            }
        else:
            path_k = np.arange(int(np.min(k_arr)), int(np.max(k_arr)) + 1, dtype=int)
            if path_k.size > int(plateau_max_candidates):
                path_k = np.unique(
                    np.rint(
                        np.linspace(
                            int(path_k[0]),
                            int(path_k[-1]),
                            num=int(plateau_max_candidates),
                        )
                    ).astype(int)
                )
            path_gamma, path_ess = _weighted_hill_path_from_sorted(z_sorted, w_sorted, path_k)
            selection = _select_plateau_from_path(
                path_k,
                path_gamma,
                path_ess,
                window_frac=float(plateau_window_frac),
                min_window=int(plateau_min_window),
                score_tolerance=float(plateau_score_tolerance),
            )
        tail_index = (
            selection["alpha_selected"]
            if np.isfinite(selection["alpha_selected"])
            else alpha_med
        )
    else:
        if k_arr.size == 0:
            selection = {
                "k_selected": np.nan,
                "gamma_selected": np.nan,
                "alpha_selected": np.nan,
                "ess_selected": np.nan,
                "selector_score": np.nan,
                "selector_status": "skip_empty_k_grid",
            }
        else:
            selection = _select_reiss_thomas(
                z_sorted,
                w_sorted,
                k_min=int(np.min(k_arr)),
                k_max=int(np.max(k_arr)),
                beta=float(reiss_thomas_beta),
            )
        tail_index = (
            selection["alpha_selected"]
            if np.isfinite(selection["alpha_selected"])
            else alpha_med
        )

    return {
        "m": m,
        "k_values": k_arr,
        "gamma_by_k": gamma_arr,
        "alpha_by_k": alpha_arr,
        "ess_by_k": ess_arr,
        "alpha_med": alpha_med,
        "alpha_iqr": alpha_iqr,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "tail_index": float(tail_index) if np.isfinite(tail_index) else np.nan,
        "selector": selector,
        **selection,
    }


def hill_result_to_frame(result: dict) -> pd.DataFrame:
    k_values = result.get("k_values", np.asarray([], dtype=int))
    selected = np.zeros_like(k_values, dtype=bool)
    k_selected = result.get("k_selected", np.nan)
    if np.isfinite(k_selected):
        selected = k_values == int(k_selected)
    out = pd.DataFrame(
        {
            "k": k_values,
            "gamma_hat": result.get("gamma_by_k", np.asarray([], dtype=float)),
            "alpha_hat": result.get("alpha_by_k", np.asarray([], dtype=float)),
            "ess": result.get("ess_by_k", np.asarray([], dtype=float)),
            "selected": selected,
        }
    )
    if np.isfinite(k_selected) and not np.any(selected):
        selected_row = pd.DataFrame(
            {
                "k": [int(k_selected)],
                "gamma_hat": [result.get("gamma_selected", np.nan)],
                "alpha_hat": [result.get("alpha_selected", np.nan)],
                "ess": [result.get("ess_selected", np.nan)],
                "selected": [True],
            }
        )
        out = pd.concat([out, selected_row], ignore_index=True).sort_values("k", kind="stable")
    return out
