from __future__ import annotations

import argparse
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ccw import InitiateBy, NoInitiationThrough
from ccw._core.analysis import (
    ccw_preprocessing,
    create_counting_process_format,
    integrate_censoring_columns,
    remove_rows_after_fup,
)
from ccw._research.configs.config_override import set_config_override_dir
from ccw._research.configs.load_variables import load_experiment_settings
from ccw._research.data_generation.core import BayesianNetwork
from ccw._research.data_generation.scenarios.scenario import ScenarioVariables
from ccw.diagnostics.hill import build_k_grid, hill_result_to_frame, weighted_hill
from ccw.diagnostics.io_utils import write_table
from ccw.diagnostics.run_diagnosis import summarize_over_runs
from ccw._research.utils import create_datasets_in_df


DEFAULT_PATTERNS = ("VAR", "HPREV2")


@dataclass(frozen=True)
class OracleTailIndexOutputs:
    output_dir: Path
    summary_path: Path
    detail_path: Path
    over_time_path: Path
    min_path: Path
    metadata_path: Path
    summary_rows: int
    detail_rows: int
    distribution_dir: Path | None = None
    distribution_summary_path: Path | None = None
    distribution_plot_count: int = 0


class OracleTailAccumulator:
    def __init__(self, patterns: Iterable[str], *, eps: float = 1e-12):
        self.patterns = tuple(_normalize_patterns(patterns))
        self.eps = float(eps)
        self._z_parts: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
        self._w_parts: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)

    def add_diagnostics(self, df: pd.DataFrame, *, max_time: int) -> None:
        if df.empty:
            return

        work = df.copy()
        work["t"] = pd.to_numeric(work["t"], errors="coerce")
        work = work[np.isfinite(work["t"])].copy()
        if work.empty:
            return
        work["t"] = work["t"].astype(int)
        work = work[work["t"] <= int(max_time)].copy()
        if work.empty:
            return

        work["S_t"] = pd.to_numeric(work["S_t"], errors="coerce").fillna(0).astype(int)
        work["G_t"] = np.clip(pd.to_numeric(work["G_t"], errors="coerce"), self.eps, 1.0)
        work["H_prev"] = np.clip(pd.to_numeric(work["H_prev"], errors="coerce"), self.eps, 1.0)

        for (arm, t), group_df in work.groupby(["a", "t"], sort=True):
            valid = (
                (group_df["S_t"] == 1)
                & np.isfinite(group_df["G_t"])
                & np.isfinite(group_df["H_prev"])
                & (group_df["G_t"] > 0)
                & (group_df["H_prev"] > 0)
            )
            if not bool(np.any(valid)):
                continue

            g = group_df.loc[valid, "G_t"].to_numpy(dtype=float)
            h = group_df.loc[valid, "H_prev"].to_numpy(dtype=float)
            if "VAR" in self.patterns:
                key = (str(arm), int(t), "VAR")
                self._z_parts[key].append(1.0 / g)
                self._w_parts[key].append(1.0 / (h * h))

            if "HPREV2" in self.patterns:
                h2_mask = valid.copy()
                if "CENSOR_tstart" in group_df.columns:
                    c_t = pd.to_numeric(group_df["CENSOR_tstart"], errors="coerce")
                    h2_mask = h2_mask & (c_t == 0)
                h2 = group_df.loc[h2_mask, "H_prev"].to_numpy(dtype=float)
                if h2.size:
                    key = (str(arm), int(t), "HPREV2")
                    self._z_parts[key].append(1.0 / (h2 * h2))
                    self._w_parts[key].append(np.ones_like(h2, dtype=float))

    def merge_parts(
        self,
        z_parts: dict[tuple[str, int, str], list[np.ndarray]],
        w_parts: dict[tuple[str, int, str], list[np.ndarray]],
    ) -> None:
        for key, parts in z_parts.items():
            norm_key = (str(key[0]), int(key[1]), str(key[2]).upper())
            self._z_parts[norm_key].extend(parts)
        for key, parts in w_parts.items():
            norm_key = (str(key[0]), int(key[1]), str(key[2]).upper())
            self._w_parts[norm_key].extend(parts)

    def export_parts(
        self,
    ) -> tuple[dict[tuple[str, int, str], list[np.ndarray]], dict[tuple[str, int, str], list[np.ndarray]]]:
        return dict(self._z_parts), dict(self._w_parts)

    def to_summary_frames(
        self,
        *,
        max_time: int,
        min_m: int,
        min_k_abs: int,
        min_k_frac: float,
        max_k_frac: float,
        k_points: int,
        tail_index_selector: str,
        reiss_thomas_beta: float,
        alpha_borderline: float,
        alpha_bad: float,
        tail_tiny: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary_records: list[dict] = []
        detail_records: list[dict] = []

        all_arms = sorted({key[0] for key in self._z_parts})
        all_keys = [
            (arm, t, pattern)
            for arm in all_arms
            for t in range(0, int(max_time) + 1)
            for pattern in self.patterns
        ]

        for arm, t, pattern in all_keys:
            key = (arm, int(t), pattern)
            z_parts = self._z_parts.get(key, [])
            w_parts = self._w_parts.get(key, [])
            z = np.concatenate(z_parts) if z_parts else np.asarray([], dtype=float)
            w = np.concatenate(w_parts) if w_parts else np.asarray([], dtype=float)
            valid = np.isfinite(z) & np.isfinite(w) & (z > 0) & (w > 0)
            z = z[valid]
            w = w[valid]
            m = int(z.size)

            row = {
                "run_id": 0,
                "a": arm,
                "t": int(t),
                "pattern": pattern,
                "m": m,
                "k_min": np.nan,
                "k_max": np.nan,
                "alpha_med": np.nan,
                "alpha_iqr": np.nan,
                "alpha_min": np.nan,
                "alpha_max": np.nan,
                "tail_index": np.nan,
                "tail_index_selector": tail_index_selector,
                "k_selected": np.nan,
                "gamma_selected": np.nan,
                "alpha_selected": np.nan,
                "ess_selected": np.nan,
                "selector_score": np.nan,
                "selector_status": pd.NA,
                "tail_q99": np.nan,
                "flag": pd.NA,
                "status": "ok",
                "weight_cap_quantile": np.nan,
                "weight_cap_value": np.nan,
                "weights_capped": 0,
            }

            if m < int(min_m):
                row["status"] = "skip_small_sample"
                summary_records.append(row)
                continue

            tail_q99 = float(np.quantile(z, 0.99))
            row["tail_q99"] = tail_q99
            if not np.isfinite(tail_q99) or tail_q99 <= 1.0 + float(tail_tiny):
                row["status"] = "skip_no_tail_signal"
                summary_records.append(row)
                continue

            k_grid = build_k_grid(
                m,
                min_k_abs=min_k_abs,
                min_frac=min_k_frac,
                max_frac=max_k_frac,
                n_points=k_points,
            )
            if not k_grid:
                row["status"] = "skip_k_grid_empty"
                summary_records.append(row)
                continue

            hill_res = weighted_hill(
                z,
                w,
                k_grid,
                tail_index_selector=tail_index_selector,
                reiss_thomas_beta=reiss_thomas_beta,
            )
            alpha = hill_res["alpha_by_k"]
            if not np.any(np.isfinite(alpha)):
                row["status"] = "skip_all_alpha_nan"
                summary_records.append(row)
                continue

            row["k_min"] = int(np.min(hill_res["k_values"]))
            row["k_max"] = int(np.max(hill_res["k_values"]))
            row["alpha_med"] = float(hill_res["alpha_med"])
            row["alpha_iqr"] = float(hill_res["alpha_iqr"])
            row["alpha_min"] = float(hill_res["alpha_min"])
            row["alpha_max"] = float(hill_res["alpha_max"])
            row["tail_index"] = float(hill_res["tail_index"])
            row["tail_index_selector"] = str(hill_res["selector"])
            row["k_selected"] = (
                int(hill_res["k_selected"]) if np.isfinite(hill_res["k_selected"]) else np.nan
            )
            row["gamma_selected"] = (
                float(hill_res["gamma_selected"])
                if np.isfinite(hill_res["gamma_selected"])
                else np.nan
            )
            row["alpha_selected"] = (
                float(hill_res["alpha_selected"])
                if np.isfinite(hill_res["alpha_selected"])
                else np.nan
            )
            row["ess_selected"] = (
                float(hill_res["ess_selected"])
                if np.isfinite(hill_res["ess_selected"])
                else np.nan
            )
            row["selector_score"] = (
                float(hill_res["selector_score"])
                if np.isfinite(hill_res["selector_score"])
                else np.nan
            )
            row["selector_status"] = hill_res["selector_status"]
            row["flag"] = bool(
                (row["tail_index"] <= alpha_borderline) or (row["alpha_min"] <= alpha_bad)
            )
            summary_records.append(row)

            by_k = hill_result_to_frame(hill_res)
            for _, r in by_k.iterrows():
                detail_records.append(
                    {
                        "run_id": 0,
                        "a": arm,
                        "t": int(t),
                        "pattern": pattern,
                        "m": m,
                        "k": int(r["k"]),
                        "gamma_hat": (
                            float(r["gamma_hat"]) if np.isfinite(r["gamma_hat"]) else np.nan
                        ),
                        "alpha_hat": (
                            float(r["alpha_hat"]) if np.isfinite(r["alpha_hat"]) else np.nan
                        ),
                        "ess": float(r["ess"]) if np.isfinite(r["ess"]) else np.nan,
                        "selected": bool(r["selected"]),
                    }
                )

        return pd.DataFrame.from_records(summary_records), pd.DataFrame.from_records(detail_records)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: Iterable[float]) -> np.ndarray:
    probs_arr = np.asarray(list(probs), dtype=float)
    if values.size == 0:
        return np.full(probs_arr.shape, np.nan, dtype=float)

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(probs_arr.shape, np.nan, dtype=float)

    cum_w = np.cumsum(w)
    targets = np.clip(probs_arr, 0.0, 1.0) * total
    return np.interp(targets, cum_w, v, left=v[0], right=v[-1])


def _lookup_summary_row(summary_df: pd.DataFrame, *, arm: str, t: int, pattern: str) -> pd.Series | None:
    if summary_df.empty:
        return None
    mask = (
        (summary_df["a"].astype(str) == str(arm))
        & (pd.to_numeric(summary_df["t"], errors="coerce") == int(t))
        & (summary_df["pattern"].astype(str).str.upper() == str(pattern).upper())
    )
    rows = summary_df.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _safe_plot_name(*parts: object) -> str:
    import re

    text = "_".join(str(part) for part in parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _survival_plot_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    full_n = max(2, max_points // 2)
    tail_n = max(2, max_points - full_n)
    full_idx = np.linspace(0, n - 1, full_n, dtype=int)
    tail_offsets = np.geomspace(1, n, tail_n).astype(int)
    tail_idx = np.clip(n - tail_offsets, 0, n - 1)
    return np.unique(np.concatenate([full_idx, tail_idx]))


def write_weighted_distribution_outputs(
    accumulator: OracleTailAccumulator,
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
    max_time: int,
    max_points: int = 2000,
) -> tuple[Path, Path, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if int(max_points) < 10:
        raise ValueError("distribution max_points must be at least 10.")

    distribution_dir = output_dir / "weighted_distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    quantile_specs = [
        ("q0", 0.0),
        ("q50", 0.5),
        ("q90", 0.9),
        ("q95", 0.95),
        ("q99", 0.99),
        ("q995", 0.995),
        ("q999", 0.999),
        ("q100", 1.0),
    ]
    quantile_probs = [prob for _, prob in quantile_specs]
    records: list[dict] = []
    plot_count = 0

    all_arms = sorted({key[0] for key in accumulator._z_parts})
    all_keys = [
        (arm, t, pattern)
        for arm in all_arms
        for t in range(0, int(max_time) + 1)
        for pattern in accumulator.patterns
    ]

    for arm, t, pattern in all_keys:
        key = (arm, int(t), pattern)
        z_parts = accumulator._z_parts.get(key, [])
        w_parts = accumulator._w_parts.get(key, [])
        z = np.concatenate(z_parts) if z_parts else np.asarray([], dtype=float)
        w = np.concatenate(w_parts) if w_parts else np.asarray([], dtype=float)
        valid = np.isfinite(z) & np.isfinite(w) & (z > 0) & (w > 0)
        z = z[valid]
        w = w[valid]
        m = int(z.size)

        summary_row = _lookup_summary_row(summary_df, arm=arm, t=int(t), pattern=pattern)
        status = "missing"
        tail_index = np.nan
        k_selected = np.nan
        threshold_selected = np.nan
        if summary_row is not None:
            status = str(summary_row.get("status", ""))
            tail_index = pd.to_numeric(pd.Series([summary_row.get("tail_index", np.nan)]), errors="coerce").iloc[0]
            k_selected = pd.to_numeric(pd.Series([summary_row.get("k_selected", np.nan)]), errors="coerce").iloc[0]

        q_values = _weighted_quantile(z, w, quantile_probs)
        weight_sum = float(np.sum(w)) if m else np.nan
        weight_ess = (
            float(weight_sum * weight_sum / np.sum(w * w))
            if m and np.isfinite(weight_sum) and np.sum(w * w) > 0.0
            else np.nan
        )
        if m and pd.notna(k_selected) and np.isfinite(k_selected) and int(k_selected) > 0:
            k_int = min(int(k_selected), m)
            threshold_selected = float(np.sort(z)[-k_int])

        record = {
            "a": arm,
            "t": int(t),
            "pattern": pattern,
            "m": m,
            "weight_sum": weight_sum,
            "weight_ess": weight_ess,
            "status": status,
            "tail_index": float(tail_index) if pd.notna(tail_index) else np.nan,
            "k_selected": float(k_selected) if pd.notna(k_selected) else np.nan,
            "threshold_selected": threshold_selected,
        }
        for (label, _), value in zip(quantile_specs, q_values, strict=True):
            record[f"z_weighted_{label}"] = float(value) if np.isfinite(value) else np.nan
        records.append(record)

        if m == 0:
            continue

        order = np.argsort(z)
        z_sorted = z[order]
        w_sorted = w[order]
        survival = np.cumsum(w_sorted[::-1])[::-1] / float(np.sum(w_sorted))
        idx = _survival_plot_indices(len(z_sorted), int(max_points))

        fig, axis = plt.subplots(figsize=(7.5, 5.5))
        axis.plot(z_sorted[idx], survival[idx], linewidth=1.4)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("z used for tail diagnostic")
        axis.set_ylabel("weighted P(Z >= z)")
        title = f"Oracle weighted survival: {arm}, {pattern}, t={int(t)}"
        if pd.notna(tail_index):
            title += f"; tail index={float(tail_index):.3g}"
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")

        q99 = record.get("z_weighted_q99", np.nan)
        if pd.notna(q99) and np.isfinite(q99):
            axis.axvline(float(q99), color="tab:orange", linestyle="--", linewidth=1.0, label="weighted q99")
        if pd.notna(threshold_selected) and np.isfinite(threshold_selected):
            axis.axvline(
                float(threshold_selected),
                color="tab:red",
                linestyle=":",
                linewidth=1.2,
                label="selected-k threshold",
            )
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best")

        fig.tight_layout()
        plot_path = distribution_dir / f"weighted_survival_{_safe_plot_name(arm, pattern, 't' + str(int(t)))}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_count += 1

    summary_path = distribution_dir / "weighted_distribution_summary.csv"
    pd.DataFrame.from_records(records).to_csv(summary_path, index=False)
    return distribution_dir, summary_path, plot_count


def estimate_oracle_tail_index(
    *,
    experiment: str,
    cutoff_time_of_intervention: int,
    sample_size: int,
    seed: int,
    n_time: int,
    cutoff_time_of_observation: int,
    config_dir: str | Path | None,
    output_dir: str | Path,
    setting: str,
    chunk_size: int = 100_000,
    n_workers: int = 1,
    patterns: Iterable[str] = DEFAULT_PATTERNS,
    min_m: int = 200,
    min_k_abs: int = 20,
    min_k_frac: float = 0.01,
    max_k_frac: float = 0.10,
    k_points: int = 20,
    tail_index_selector: str = "plateau",
    reiss_thomas_beta: float = 0.3,
    alpha_borderline: float = 1.2,
    alpha_bad: float = 1.0,
    tail_tiny: float = 1e-8,
    eps: float = 1e-12,
    write_distribution_plots: bool = False,
    distribution_max_points: int = 2000,
    verbose: bool = True,
) -> OracleTailIndexOutputs:
    if int(sample_size) <= 0:
        raise ValueError("sample_size must be positive.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    if int(n_workers) <= 0:
        raise ValueError("n_workers must be positive.")
    if int(cutoff_time_of_intervention) > int(cutoff_time_of_observation):
        raise ValueError("cutoff_time_of_intervention must be <= cutoff_time_of_observation.")

    patterns_use = _normalize_patterns(patterns)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir_str = None if config_dir is None else str(Path(config_dir).expanduser().resolve())

    set_config_override_dir(config_dir_str)
    _, _, configs = load_experiment_settings(experiment)
    args = argparse.Namespace(cutoff_time_of_intervention=int(cutoff_time_of_intervention))
    strategies = {key: maker(args) for key, maker in configs["strategy_creator"].items()}
    _validate_supported_strategies(strategies)

    accumulator = OracleTailAccumulator(patterns_use, eps=eps)
    rng = np.random.default_rng(int(seed))
    chunk_specs: list[dict[str, int]] = []
    generated = 0
    while generated < int(sample_size):
        n_chunk = min(int(chunk_size), int(sample_size) - generated)
        chunk_seed = int(rng.integers(0, 2**32 - 1))
        chunk_specs.append(
            {
                "chunk_idx": len(chunk_specs),
                "n_chunk": int(n_chunk),
                "chunk_seed": int(chunk_seed),
                "id_offset": int(generated),
            }
        )
        generated += n_chunk

    common_worker_kwargs = {
        "experiment": experiment,
        "cutoff_time_of_intervention": int(cutoff_time_of_intervention),
        "n_time": int(n_time),
        "cutoff_time_of_observation": int(cutoff_time_of_observation),
        "config_dir": config_dir_str,
        "patterns": tuple(patterns_use),
        "eps": float(eps),
    }
    if int(n_workers) == 1 or len(chunk_specs) == 1:
        for spec in chunk_specs:
            if verbose:
                print(
                    f"[oracle-tail] chunk {spec['chunk_idx'] + 1}/{len(chunk_specs)}: "
                    f"simulating {spec['n_chunk']} subjects "
                    f"({spec['id_offset'] + spec['n_chunk']}/{sample_size})",
                    flush=True,
                )
            _, _, z_parts, w_parts = _process_oracle_tail_chunk(**common_worker_kwargs, **spec)
            accumulator.merge_parts(z_parts, w_parts)
    else:
        max_workers = min(int(n_workers), len(chunk_specs))
        if verbose:
            print(
                f"[oracle-tail] processing {len(chunk_specs)} chunks with n_workers={max_workers}",
                flush=True,
            )
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_oracle_tail_chunk, **common_worker_kwargs, **spec): spec
                for spec in chunk_specs
            }
            completed_subjects = 0
            for future in as_completed(futures):
                spec = futures[future]
                chunk_idx, n_chunk, z_parts, w_parts = future.result()
                accumulator.merge_parts(z_parts, w_parts)
                completed_subjects += int(n_chunk)
                if verbose:
                    print(
                        f"[oracle-tail] chunk {chunk_idx + 1}/{len(chunk_specs)} done "
                        f"({completed_subjects}/{sample_size} subjects merged)",
                        flush=True,
                    )

    summary_df, detail_df = accumulator.to_summary_frames(
        max_time=int(cutoff_time_of_intervention),
        min_m=min_m,
        min_k_abs=min_k_abs,
        min_k_frac=min_k_frac,
        max_k_frac=max_k_frac,
        k_points=k_points,
        tail_index_selector=tail_index_selector,
        reiss_thomas_beta=reiss_thomas_beta,
        alpha_borderline=alpha_borderline,
        alpha_bad=alpha_bad,
        tail_tiny=tail_tiny,
    )
    over_time_df = summarize_over_runs(summary_df)
    min_df = _oracle_min_over_grace_period(
        over_time_df,
        setting=setting,
        experiment=experiment,
        cutoff_time_of_intervention=int(cutoff_time_of_intervention),
        sample_size=int(sample_size),
        seed=int(seed),
        config_dir=config_dir,
    )

    distribution_dir = None
    distribution_summary_path = None
    distribution_plot_count = 0
    if write_distribution_plots:
        distribution_dir, distribution_summary_path, distribution_plot_count = write_weighted_distribution_outputs(
            accumulator,
            output_dir=output_dir,
            summary_df=summary_df,
            max_time=int(cutoff_time_of_intervention),
            max_points=int(distribution_max_points),
        )

    summary_path = output_dir / "hill_summary.csv"
    detail_path = output_dir / "hill_by_k.csv"
    over_time_path = output_dir / "tail_index_over_runs.csv"
    min_path = output_dir / "oracle_tail_index_min.csv"
    metadata_path = output_dir / "oracle_tail_index_meta.json"

    write_table(summary_df, summary_path, mode="overwrite")
    write_table(detail_df, detail_path, mode="overwrite")
    over_time_df.to_csv(over_time_path, index=False)
    min_df.to_csv(min_path, index=False)
    _write_metadata(
        metadata_path,
        {
            "experiment": experiment,
            "setting": setting,
            "cutoff_time_of_intervention": int(cutoff_time_of_intervention),
            "sample_size": int(sample_size),
            "seed": int(seed),
            "n_time": int(n_time),
            "cutoff_time_of_observation": int(cutoff_time_of_observation),
            "chunk_size": int(chunk_size),
            "n_workers": int(n_workers),
            "patterns": list(patterns_use),
            "tail_index_selector": tail_index_selector,
            "config_dir": config_dir_str,
            "write_distribution_plots": bool(write_distribution_plots),
            "distribution_max_points": int(distribution_max_points),
            "distribution_dir": None if distribution_dir is None else str(distribution_dir),
            "distribution_summary_path": (
                None if distribution_summary_path is None else str(distribution_summary_path)
            ),
            "distribution_plot_count": int(distribution_plot_count),
        },
    )

    return OracleTailIndexOutputs(
        output_dir=output_dir,
        summary_path=summary_path,
        detail_path=detail_path,
        over_time_path=over_time_path,
        min_path=min_path,
        metadata_path=metadata_path,
        summary_rows=len(summary_df),
        detail_rows=len(detail_df),
        distribution_dir=distribution_dir,
        distribution_summary_path=distribution_summary_path,
        distribution_plot_count=distribution_plot_count,
    )


def infer_setting_label(*, coef_a: str | None, coef_d: str | None, config_dir: str | Path | None) -> str:
    if coef_a is not None and coef_d is not None:
        return f"a{coef_a}d{coef_d}"
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve().name
    return "default"


def infer_config_dir(
    *,
    repo_root: str | Path,
    coef_a: str | None,
    coef_d: str | None,
    config_dir: str | Path | None,
) -> Path | None:
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    if coef_a is None and coef_d is None:
        return None
    if coef_a is None or coef_d is None:
        raise ValueError("--coefA and --coefD must be supplied together.")
    setting = infer_setting_label(coef_a=coef_a, coef_d=coef_d, config_dir=None)
    config_setting = "setting1" if setting == "a1d1" else setting
    return (
        Path(repo_root).expanduser().resolve()
        / "scripts"
        / "paper"
        / "config_overrides"
        / config_setting
    )


def default_output_dir(
    *,
    repo_root: str | Path,
    setting: str,
    experiment: str,
    cutoff_time_of_intervention: int,
) -> Path:
    return (
        Path(repo_root).expanduser().resolve()
        / "output_diagnostics"
        / "oracle_tail_index"
        / setting
        / experiment
        / f"cut{int(cutoff_time_of_intervention)}"
    )


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Estimate oracle Monte Carlo tail indexes for experimentA-style clone-censor "
            "weights using true treatment propensities."
        )
    )
    parser.add_argument("--experiment", default="experimentA")
    parser.add_argument("--cutoff_time_of_intervention", type=int, required=True)
    parser.add_argument("--coefA", default=None, choices=["0.5", "1", "2"])
    parser.add_argument("--coefD", default=None, choices=["0.5", "1", "2"])
    parser.add_argument(
        "--config_dir",
        type=Path,
        default=None,
        help="Optional full config override directory. Overrides --coefA/--coefD if supplied.",
    )
    parser.add_argument("--sample_size", type=int, default=500_000)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=917_503)
    parser.add_argument("--n_time", type=int, default=32)
    parser.add_argument("--cutoff_time_of_observation", type=int, default=31)
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--min-m", type=int, default=200)
    parser.add_argument("--min-k-abs", type=int, default=20)
    parser.add_argument("--min-k-frac", type=float, default=0.01)
    parser.add_argument("--max-k-frac", type=float, default=0.10)
    parser.add_argument("--k-points", type=int, default=20)
    parser.add_argument(
        "--tail-index-selector",
        choices=["median", "plateau", "reiss_thomas"],
        default="plateau",
    )
    parser.add_argument("--reiss-thomas-beta", type=float, default=0.3)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[4]
    config_dir = infer_config_dir(
        repo_root=repo_root,
        coef_a=args.coefA,
        coef_d=args.coefD,
        config_dir=args.config_dir,
    )
    if config_dir is not None and not config_dir.exists():
        raise FileNotFoundError(f"Config override directory not found: {config_dir}")

    setting = infer_setting_label(coef_a=args.coefA, coef_d=args.coefD, config_dir=config_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(
            repo_root=repo_root,
            setting=setting,
            experiment=args.experiment,
            cutoff_time_of_intervention=args.cutoff_time_of_intervention,
        )
    )
    outputs = estimate_oracle_tail_index(
        experiment=args.experiment,
        cutoff_time_of_intervention=args.cutoff_time_of_intervention,
        sample_size=args.sample_size,
        seed=args.seed,
        n_time=args.n_time,
        cutoff_time_of_observation=args.cutoff_time_of_observation,
        config_dir=config_dir,
        output_dir=output_dir,
        setting=setting,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
        patterns=args.patterns,
        min_m=args.min_m,
        min_k_abs=args.min_k_abs,
        min_k_frac=args.min_k_frac,
        max_k_frac=args.max_k_frac,
        k_points=args.k_points,
        tail_index_selector=args.tail_index_selector,
        reiss_thomas_beta=args.reiss_thomas_beta,
        verbose=not args.quiet,
    )

    print(f"Output directory: {outputs.output_dir}")
    print(f"Summary: {outputs.summary_path} (rows={outputs.summary_rows})")
    print(f"By-k: {outputs.detail_path} (rows={outputs.detail_rows})")
    print(f"Over-time: {outputs.over_time_path}")
    print(f"Oracle min-over-grace-period: {outputs.min_path}")
    print(f"Metadata: {outputs.metadata_path}")
    return 0


def _process_oracle_tail_chunk(
    *,
    experiment: str,
    cutoff_time_of_intervention: int,
    n_time: int,
    cutoff_time_of_observation: int,
    config_dir: str | None,
    patterns: tuple[str, ...],
    eps: float,
    chunk_idx: int,
    n_chunk: int,
    chunk_seed: int,
    id_offset: int,
) -> tuple[int, int, dict[tuple[str, int, str], list[np.ndarray]], dict[tuple[str, int, str], list[np.ndarray]]]:
    set_config_override_dir(config_dir)
    _, scenario_vars, configs = load_experiment_settings(experiment)
    args = argparse.Namespace(cutoff_time_of_intervention=int(cutoff_time_of_intervention))
    strategies = {key: maker(args) for key, maker in configs["strategy_creator"].items()}
    _validate_supported_strategies(strategies)

    treatment_var = configs["treatment_var"]
    outcome_var = configs["outcome_var"]
    treatment_col = treatment_var.name
    outcome_col = outcome_var.name
    censor_vars = configs.get("censor_vars", None)
    censor_cols = [] if censor_vars is None else [var.name for var in censor_vars]
    time_varying_vars = configs.get("time_varying_vars", None)
    time_varying_cols = None if time_varying_vars is None else [var.name for var in time_varying_vars]

    bn = BayesianNetwork(scenario_vars, int(n_time) + 1)
    treatment_cpd = bn.variable_ids_to_variables[treatment_var].CPD
    sample = bn.sample(sample_size=int(n_chunk), seed=int(chunk_seed))
    _, _, _, df_joined_raw = create_datasets_in_df(
        bn,
        sample,
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        cut_data_after_outcome=configs["cut_data_after_outcome"],
        cutoff_time_of_observation=int(cutoff_time_of_observation),
        censor_vars=censor_vars,
    )
    df_joined_raw = df_joined_raw.copy()
    df_joined_raw["id"] = df_joined_raw["id"].astype(int) + int(id_offset)
    df_joined_raw["oracle_treatment_prob"] = _oracle_treatment_probability(
        df_joined_raw,
        treatment_cpd=treatment_cpd,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )

    df_joined, _ = configs["preprocess_pipeline"](df_joined_raw)
    df_pre = ccw_preprocessing(
        df_joined,
        int(cutoff_time_of_observation),
        outcome_col,
        censor_cols,
    )

    accumulator = OracleTailAccumulator(patterns, eps=eps)
    for arm, strategy in strategies.items():
        cloned_df = df_pre.copy()
        cloned_df["arm"] = arm
        cloned_df = strategy.artificial_censor(
            cloned_df,
            id_col="id",
            time_col="time",
            treatment_col=treatment_col,
        )
        cloned_df = create_counting_process_format(
            cloned_df,
            outcome_col,
            censor_cols,
            time_varying_cols,
        )
        cloned_df = remove_rows_after_fup(cloned_df)
        cloned_df = integrate_censoring_columns(cloned_df, censor_cols)
        diag_df = _oracle_diagnostic_table_for_arm(
            cloned_df,
            strategy=strategy,
            treatment_col=treatment_col,
            cutoff_time_of_intervention=int(cutoff_time_of_intervention),
            eps=eps,
        )
        accumulator.add_diagnostics(diag_df, max_time=int(cutoff_time_of_intervention))

    z_parts, w_parts = accumulator.export_parts()
    return int(chunk_idx), int(n_chunk), z_parts, w_parts


def _normalize_patterns(patterns: Iterable[str]) -> list[str]:
    allowed = set(DEFAULT_PATTERNS)
    out: list[str] = []
    for pattern in patterns:
        value = str(pattern).upper()
        if value not in allowed:
            raise ValueError(f"Unsupported oracle pattern '{pattern}'. Allowed: {sorted(allowed)}.")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("At least one pattern is required.")
    return out


def _validate_supported_strategies(strategies: dict[str, object]) -> None:
    required = {"control", "intervention"}
    missing = required - set(strategies)
    if missing:
        raise ValueError(f"Experiment must define strategies {sorted(required)}; missing {sorted(missing)}.")
    for arm, strategy in strategies.items():
        if not isinstance(strategy, (InitiateBy, NoInitiationThrough)):
            raise ValueError(
                "Oracle tail index currently supports only InitiateBy and "
                f"NoInitiationThrough strategies. Got {arm}={strategy!r}."
            )


def _oracle_treatment_probability(
    df: pd.DataFrame,
    *,
    treatment_cpd,
    treatment_col: str,
    outcome_col: str,
) -> np.ndarray:
    treatment_prob_vectorized = getattr(treatment_cpd, "_treatment_prob_vectorized", None)
    if not callable(treatment_prob_vectorized):
        raise TypeError("Treatment CPD does not expose _treatment_prob_vectorized.")
    uses_spo2 = _vectorized_treatment_probability_uses_spo2(treatment_prob_vectorized)

    required = {
        "id",
        "time",
        ScenarioVariables.SEX.name,
        ScenarioVariables.AGE.name,
        ScenarioVariables.CCI.name,
        treatment_col,
        outcome_col,
    }
    if uses_spo2:
        required.add(ScenarioVariables.SPO2.name)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns needed for oracle treatment probabilities: {sorted(missing)}")

    probs = np.zeros(len(df), dtype=float)
    work = df.copy()
    work["_oracle_row_pos"] = np.arange(len(work), dtype=int)
    work = work.sort_values(["id", "time"]).copy()
    treated_before = (
        work.groupby("id", sort=False)[treatment_col]
        .cumsum()
        .groupby(work["id"], sort=False)
        .shift(1, fill_value=0)
    )
    can_treat = (treated_before.to_numpy(dtype=float) <= 0) & (
        pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(dtype=float) != 1
    )

    for time_value, idx in work.groupby("time", sort=True).groups.items():
        pos = work.index.get_indexer(idx)
        valid_pos = pos[can_treat[pos]]
        if valid_pos.size == 0:
            continue
        rows = work.iloc[valid_pos]
        vectorized_args = [
            rows[ScenarioVariables.SEX.name].to_numpy(),
            rows[ScenarioVariables.AGE.name].to_numpy(dtype=float),
            rows[ScenarioVariables.CCI.name].to_numpy(dtype=float),
        ]
        if uses_spo2:
            vectorized_args.append(rows[ScenarioVariables.SPO2.name].to_numpy(dtype=float))
        vectorized_args.append(int(time_value))

        probs_for_time = treatment_prob_vectorized(*vectorized_args)
        probs[rows["_oracle_row_pos"].to_numpy(dtype=int)] = probs_for_time

    probs = np.clip(probs, 0.0, 1.0)
    return probs


def _vectorized_treatment_probability_uses_spo2(treatment_prob_vectorized) -> bool:
    params = [
        param
        for param in inspect.signature(treatment_prob_vectorized).parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    n_positional = len(params)
    if n_positional == 5:
        return True
    if n_positional == 4:
        return False
    raise TypeError(
        "Unsupported _treatment_prob_vectorized signature. Expected "
        "(sexes, ages, ccis, time) or (sexes, ages, ccis, spo2s, time)."
    )


def _oracle_diagnostic_table_for_arm(
    df: pd.DataFrame,
    *,
    strategy,
    treatment_col: str,
    cutoff_time_of_intervention: int,
    eps: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["a", "t", "id", "S_t", "G_t", "H_prev", "CENSOR_tstart"])
    if "oracle_treatment_prob" not in df.columns:
        raise KeyError("oracle_treatment_prob column is required.")

    work = df.sort_values(["id", "time"]).copy()
    p = np.clip(
        pd.to_numeric(work["oracle_treatment_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        eps,
        1.0 - eps,
    )
    g_t = np.ones(len(work), dtype=float)

    if isinstance(strategy, InitiateBy):
        treated_before_x = (
            work[(work["time"] < cutoff_time_of_intervention) & (work[treatment_col] == 1)]
            .groupby("id")["id"]
            .first()
            .index
        )
        at_risk = (~work["id"].isin(treated_before_x)).to_numpy() & (
            work["time"].to_numpy(dtype=int) == int(cutoff_time_of_intervention)
        )
        g_t[at_risk] = p[at_risk]
    elif isinstance(strategy, NoInitiationThrough):
        at_risk = work["time"].to_numpy(dtype=int) <= int(cutoff_time_of_intervention)
        g_t[at_risk] = 1.0 - p[at_risk]
    else:
        raise TypeError(f"Unsupported strategy for oracle diagnostic table: {strategy!r}")

    g_t = np.clip(g_t, eps, 1.0)
    g_series = pd.Series(g_t, index=work.index)
    h_full = g_series.groupby(work["id"], sort=False).cumprod()
    h_prev = h_full.groupby(work["id"], sort=False).shift(1, fill_value=1.0)

    out = pd.DataFrame(
        {
            "a": work["arm"].to_numpy(),
            "t": work["tstart"].to_numpy(dtype=int),
            "id": work["id"].to_numpy(),
            "S_t": 1,
            "G_t": g_series.to_numpy(dtype=float),
            "H_prev": h_prev.to_numpy(dtype=float),
            "CENSOR_tstart": work.get("CENSOR_tstart", pd.Series(0, index=work.index)).to_numpy(),
        }
    )
    return out


def _oracle_min_over_grace_period(
    over_time_df: pd.DataFrame,
    *,
    setting: str,
    experiment: str,
    cutoff_time_of_intervention: int,
    sample_size: int,
    seed: int,
    config_dir: str | Path | None,
) -> pd.DataFrame:
    if over_time_df.empty:
        return pd.DataFrame()

    df = over_time_df.copy()
    if "tail_index_mean" in df.columns:
        tail_col = "tail_index_mean"
    elif "alpha_selected_mean" in df.columns:
        tail_col = "alpha_selected_mean"
    else:
        tail_col = "alpha_med_mean"
    df[tail_col] = pd.to_numeric(df[tail_col], errors="coerce")
    df["tail_q99_mean"] = pd.to_numeric(df.get("tail_q99_mean"), errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64")

    records: list[dict] = []
    for (arm, pattern), group_df in df.groupby(["a", "pattern"], sort=True):
        values = group_df[tail_col].to_numpy(dtype=float)
        q99_values = group_df["tail_q99_mean"].to_numpy(dtype=float)
        valid_values = values[np.isfinite(values)]
        valid_q99 = q99_values[np.isfinite(q99_values)]
        row = {
            "setting": setting,
            "experiment": experiment,
            "cutoff": int(cutoff_time_of_intervention),
            "cutoff_time_of_intervention": int(cutoff_time_of_intervention),
            "sample_size": int(sample_size),
            "seed": int(seed),
            "a": arm,
            "group": "treated" if arm == "intervention" else arm,
            "tail_pattern": pattern,
            "pattern": pattern,
            "oracle_tail_index_min": float(np.min(valid_values)) if valid_values.size else np.nan,
            "oracle_tail_index_median_over_t": (
                float(np.median(valid_values)) if valid_values.size else np.nan
            ),
            "oracle_tail_q99_min_over_t": float(np.min(valid_q99)) if valid_q99.size else np.nan,
            "tail_summary_times": f"0..{int(cutoff_time_of_intervention)}",
            "n_time_ok": int(np.sum(np.isfinite(values))),
            "config_dir": "" if config_dir is None else str(Path(config_dir).expanduser().resolve()),
        }
        for _, r in group_df.iterrows():
            t = r["t"]
            if pd.isna(t):
                continue
            row[f"oracle_tail_index_t{int(t)}"] = float(r[tail_col]) if np.isfinite(r[tail_col]) else np.nan
            row[f"oracle_tail_q99_t{int(t)}"] = (
                float(r["tail_q99_mean"]) if np.isfinite(r["tail_q99_mean"]) else np.nan
            )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _write_metadata(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
