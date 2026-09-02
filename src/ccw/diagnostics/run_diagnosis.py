from __future__ import annotations

import argparse
import re
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .hill import build_k_grid, hill_result_to_frame, weighted_hill
from .io_utils import read_table, write_table
from .log_weights import standardize_diag_table


def _infer_run_id_from_name(path: Path) -> int | None:
    m = re.search(r"_run_(\d+)", path.name)
    if m:
        return int(m.group(1))
    return None


def _infer_regime_from_name(path: Path) -> str | None:
    name = path.name.lower()
    if "treated" in name or "intervention" in name:
        return "intervention"
    if "control" in name:
        return "control"
    return None


def _supported_file(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".parquet", ".feather"}


def _find_diagnostics_input_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        return [path]

    direct_files = sorted(p for p in path.iterdir() if p.is_file() and _supported_file(p))
    direct_ipw_named = [p for p in direct_files if p.name.startswith("ipw_weights_")]
    if direct_ipw_named:
        return direct_ipw_named

    files = sorted(p for p in path.rglob("*") if p.is_file() and _supported_file(p))
    if not files:
        raise FileNotFoundError(f"No supported files found under directory: {path}")

    ipw_named = [p for p in files if p.name.startswith("ipw_weights_")]
    if ipw_named:
        files = ipw_named
    return files


def _load_diagnostics_input_file(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    try:
        df = read_table(path)
    except pd.errors.ParserError as exc:
        if path.suffix.lower() != ".csv":
            raise RuntimeError(
                "Malformed diagnostics input detected: "
                f"{path}. Only CSV inputs can be retried by skipping malformed rows."
            ) from exc
        warnings.warn(
            "Malformed diagnostics input CSV detected; retrying with malformed rows skipped: "
            f"{path}",
            RuntimeWarning,
            stacklevel=2,
        )
        df = pd.read_csv(
            path,
            engine="python",
            on_bad_lines="warn",
        )
    if "run_id" not in df.columns:
        run_id = _infer_run_id_from_name(path)
        if run_id is not None:
            df["run_id"] = run_id
    if "arm" not in df.columns:
        inferred_arm = _infer_regime_from_name(path)
        if inferred_arm is not None:
            df["arm"] = inferred_arm
    return df


def _summary_output_columns(*, by_regime: bool, by_run: bool) -> list[str]:
    cols: list[str] = []
    if by_run:
        cols.append("run_id")
    if by_regime:
        cols.append("a")
    cols.extend(
        [
            "t",
            "pattern",
            "m",
            "k_min",
            "k_max",
            "alpha_med",
            "alpha_iqr",
            "alpha_min",
            "alpha_max",
            "tail_index",
            "tail_index_selector",
            "k_selected",
            "gamma_selected",
            "alpha_selected",
            "ess_selected",
            "selector_score",
            "selector_status",
            "tail_q99",
            "flag",
            "status",
            "weight_cap_quantile",
            "weight_cap_value",
            "weights_capped",
        ]
    )
    return cols


def _detail_output_columns(*, by_regime: bool, by_run: bool) -> list[str]:
    cols: list[str] = []
    if by_run:
        cols.append("run_id")
    if by_regime:
        cols.append("a")
    cols.extend(["t", "pattern", "m", "k", "gamma_hat", "alpha_hat", "ess", "selected"])
    return cols


def _empty_summary_output(*, by_regime: bool, by_run: bool) -> pd.DataFrame:
    return pd.DataFrame(columns=_summary_output_columns(by_regime=by_regime, by_run=by_run))


def _empty_detail_output(*, by_regime: bool, by_run: bool) -> pd.DataFrame:
    return pd.DataFrame(columns=_detail_output_columns(by_regime=by_regime, by_run=by_run))


def _normalize_patterns(patterns: Iterable[str]) -> list[str]:
    out = []
    for p in patterns:
        q = str(p).upper()
        if q not in {"UNW", "NAT", "VAR", "DIRECT", "HPREV2"}:
            raise ValueError(f"Unknown pattern '{p}'. Allowed: UNW, NAT, VAR, DIRECT, HPREV2.")
        out.append(q)
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def diagnose_dataframe(
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
    """
    Compute IPCW tail diagnostics by time for NAT/VAR/(optional DIRECT).
    """
    if diag_df is None or diag_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    patterns_use = _normalize_patterns(patterns)

    df = standardize_diag_table(diag_df)
    if "run_id" in diag_df.columns and "run_id" not in df.columns:
        df["run_id"] = diag_df["run_id"].to_numpy()

    df = df.copy()
    df["S_t"] = pd.to_numeric(df["S_t"], errors="coerce").fillna(0).astype(int)
    df["G_t"] = pd.to_numeric(df["G_t"], errors="coerce")
    df["H_prev"] = pd.to_numeric(df["H_prev"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df[np.isfinite(df["t"])].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["t"] = df["t"].astype(int)
    df["G_t"] = np.clip(df["G_t"].astype(float), eps, 1.0 - eps)
    df["H_prev"] = np.clip(df["H_prev"].astype(float), eps, 1.0)
    if max_time is not None:
        df = df[df["t"] <= int(max_time)].copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

    group_cols = ["t"]
    if by_regime and "a" in df.columns:
        group_cols = ["a"] + group_cols
    if by_run and "run_id" in df.columns:
        group_cols = ["run_id"] + group_cols

    summary_records: list[dict] = []
    detail_records: list[dict] = []

    grouped = df.groupby(group_cols, sort=True)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols, group_key, strict=True))

        mask = (
            (group_df["S_t"] == 1)
            & np.isfinite(group_df["G_t"])
            & np.isfinite(group_df["H_prev"])
            & (group_df["G_t"] > 0)
            & (group_df["H_prev"] > 0)
        )
        if not np.any(mask):
            for pattern in patterns_use:
                summary_records.append(
                    {
                        **group_meta,
                        "pattern": pattern,
                        "m": 0,
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
                        "status": "skip_no_valid_rows",
                        "weight_cap_quantile": weight_cap_quantile,
                        "weight_cap_value": np.nan,
                        "weights_capped": 0,
                    }
                )
            continue

        g = group_df.loc[mask, "G_t"].to_numpy(dtype=float)
        h = group_df.loc[mask, "H_prev"].to_numpy(dtype=float)
        z_base = 1.0 / g
        h2_mask = mask.copy()
        if "CENSOR_tstart" in group_df.columns:
            c_t = pd.to_numeric(group_df["CENSOR_tstart"], errors="coerce")
            h2_mask = h2_mask & (c_t == 0)
        h2 = group_df.loc[h2_mask, "H_prev"].to_numpy(dtype=float)

        for pattern in patterns_use:
            if pattern == "UNW":
                z = z_base
                w = np.ones_like(z)
            elif pattern == "NAT":
                z = z_base
                w = 1.0 / h
            elif pattern == "VAR":
                z = z_base
                w = 1.0 / (h ** 2)
            elif pattern == "HPREV2":
                # tail diagnostic for past accumulation among uncensored rows at time t
                z = 1.0 / (h2 ** 2)
                w = np.ones_like(z)
            else:
                z = 1.0 / (g * (h ** 2))
                w = np.ones_like(z)

            valid = np.isfinite(z) & np.isfinite(w) & (z > 0) & (w > 0)
            z_use = z[valid]
            w_use = w[valid]
            m = int(z_use.size)

            row = {
                **group_meta,
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
                "weight_cap_quantile": weight_cap_quantile,
                "weight_cap_value": np.nan,
                "weights_capped": 0,
            }

            if m < min_m:
                row["status"] = "skip_small_sample"
                summary_records.append(row)
                continue

            tail_q99 = float(np.quantile(z_use, 0.99))
            row["tail_q99"] = tail_q99
            if not np.isfinite(tail_q99) or tail_q99 <= 1.0 + tail_tiny:
                row["status"] = "skip_no_tail_signal"
                summary_records.append(row)
                continue

            if pattern in {"NAT", "VAR"} and weight_cap_quantile is not None:
                cap_q = float(weight_cap_quantile)
                if not (0.0 < cap_q < 1.0):
                    raise ValueError("weight_cap_quantile must be in (0, 1).")
                cap_val = float(np.quantile(w_use, cap_q))
                if np.isfinite(cap_val) and cap_val > 0:
                    capped = int(np.sum(w_use > cap_val))
                    if capped > 0:
                        w_use = np.minimum(w_use, cap_val)
                        row["weights_capped"] = capped
                    row["weight_cap_value"] = cap_val

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
                z_use,
                w_use,
                k_grid,
                tail_index_selector=tail_index_selector,
                reiss_thomas_beta=reiss_thomas_beta,
            )
            alpha = hill_res["alpha_by_k"]
            finite_alpha = np.isfinite(alpha)
            if not np.any(finite_alpha):
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
                int(hill_res["k_selected"])
                if np.isfinite(hill_res["k_selected"])
                else np.nan
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
                (row["tail_index"] <= alpha_borderline)
                or (row["alpha_min"] <= alpha_bad)
            )
            summary_records.append(row)

            by_k = hill_result_to_frame(hill_res)
            for _, r in by_k.iterrows():
                detail_records.append(
                    {
                        **group_meta,
                        "pattern": pattern,
                        "m": m,
                        "k": int(r["k"]),
                        "gamma_hat": float(r["gamma_hat"]) if np.isfinite(r["gamma_hat"]) else np.nan,
                        "alpha_hat": float(r["alpha_hat"]) if np.isfinite(r["alpha_hat"]) else np.nan,
                        "ess": float(r["ess"]) if np.isfinite(r["ess"]) else np.nan,
                        "selected": bool(r["selected"]),
                    }
                )

    summary_df = pd.DataFrame.from_records(summary_records)
    detail_df = pd.DataFrame.from_records(detail_records)
    return summary_df, detail_df


def _diagnose_input_file(
    task: tuple[Path, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path, kwargs = task
    input_df = _load_diagnostics_input_file(input_path)
    return diagnose_dataframe(input_df, **kwargs)


def diagnose_path_to_outputs(
    in_path: str | Path,
    out_summary_path: str | Path,
    *,
    out_detail_path: str | Path | None = None,
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
    n_workers: int = 1,
) -> dict[str, int]:
    summary_rows = 0
    detail_rows = 0
    wrote_summary = False
    wrote_detail = False
    input_paths = _find_diagnostics_input_files(in_path)
    input_files = len(input_paths)
    if int(n_workers) <= 0:
        raise ValueError("n_workers must be positive.")
    workers_used = min(int(n_workers), input_files)

    diagnose_kwargs = {
        "patterns": patterns,
        "by_regime": by_regime,
        "by_run": by_run,
        "max_time": max_time,
        "eps": eps,
        "min_m": min_m,
        "min_k_abs": min_k_abs,
        "min_k_frac": min_k_frac,
        "max_k_frac": max_k_frac,
        "k_points": k_points,
        "tail_index_selector": tail_index_selector,
        "reiss_thomas_beta": reiss_thomas_beta,
        "weight_cap_quantile": weight_cap_quantile,
        "alpha_borderline": alpha_borderline,
        "alpha_bad": alpha_bad,
        "tail_tiny": tail_tiny,
    }

    tasks = [(path, diagnose_kwargs) for path in input_paths]

    def write_result(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
        nonlocal summary_rows, detail_rows, wrote_summary, wrote_detail
        if not summary_df.empty:
            write_table(summary_df, out_summary_path, mode="append" if wrote_summary else "overwrite")
            summary_rows += len(summary_df)
            wrote_summary = True
        if out_detail_path is not None and not detail_df.empty:
            write_table(detail_df, out_detail_path, mode="append" if wrote_detail else "overwrite")
            detail_rows += len(detail_df)
            wrote_detail = True

    if workers_used == 1:
        for summary_df, detail_df in map(_diagnose_input_file, tasks):
            write_result(summary_df, detail_df)
    else:
        with ProcessPoolExecutor(max_workers=workers_used) as pool:
            for summary_df, detail_df in pool.map(_diagnose_input_file, tasks):
                write_result(summary_df, detail_df)

    if not wrote_summary:
        write_table(_empty_summary_output(by_regime=by_regime, by_run=by_run), out_summary_path, mode="overwrite")
    if out_detail_path is not None and not wrote_detail:
        write_table(_empty_detail_output(by_regime=by_regime, by_run=by_run), out_detail_path, mode="overwrite")

    return {
        "input_files": input_files,
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "workers_used": workers_used,
    }


def summarize_over_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-time tail behavior over runs for each arm/pattern.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy()
    keys = [c for c in ("a", "pattern", "t") if c in df.columns]
    if not keys:
        return pd.DataFrame()

    run_col = "run_id" if "run_id" in df.columns else None
    if "tail_index" not in df.columns:
        if "alpha_selected" in df.columns:
            df["tail_index"] = pd.to_numeric(df["alpha_selected"], errors="coerce")
            df["tail_index"] = df["tail_index"].fillna(pd.to_numeric(df["alpha_med"], errors="coerce"))
        else:
            df["tail_index"] = pd.to_numeric(df["alpha_med"], errors="coerce")

    total_rows = df.groupby(keys).size().rename("n_rows_total")
    if run_col is not None:
        total_runs = df.groupby(keys)[run_col].nunique().rename("n_runs_total")
    else:
        total_runs = total_rows.rename("n_runs_total")

    ok = df[(df["status"] == "ok") & np.isfinite(pd.to_numeric(df["tail_index"], errors="coerce"))].copy()
    if ok.empty:
        out = pd.concat([total_rows, total_runs], axis=1).reset_index()
        out["n_rows_ok"] = 0
        out["n_runs_ok"] = 0
        out["tail_index_mean"] = np.nan
        out["tail_index_median"] = np.nan
        out["tail_index_q25"] = np.nan
        out["tail_index_q75"] = np.nan
        out["tail_index_min"] = np.nan
        out["tail_index_max"] = np.nan
        out["alpha_med_mean"] = np.nan
        out["alpha_med_median"] = np.nan
        out["alpha_med_q25"] = np.nan
        out["alpha_med_q75"] = np.nan
        out["alpha_med_min"] = np.nan
        out["alpha_med_max"] = np.nan
        out["alpha_selected_mean"] = np.nan
        out["alpha_selected_median"] = np.nan
        out["k_selected_median"] = np.nan
        out["ess_selected_median"] = np.nan
        out["selector_score_median"] = np.nan
        out["tail_q99_mean"] = np.nan
        return out

    ok["tail_index"] = pd.to_numeric(ok["tail_index"], errors="coerce")
    ok["alpha_med"] = pd.to_numeric(ok["alpha_med"], errors="coerce")
    if "alpha_selected" not in ok.columns:
        ok["alpha_selected"] = np.nan
    if "k_selected" not in ok.columns:
        ok["k_selected"] = np.nan
    if "ess_selected" not in ok.columns:
        ok["ess_selected"] = np.nan
    if "selector_score" not in ok.columns:
        ok["selector_score"] = np.nan
    ok["alpha_selected"] = pd.to_numeric(ok["alpha_selected"], errors="coerce")
    ok["k_selected"] = pd.to_numeric(ok["k_selected"], errors="coerce")
    ok["ess_selected"] = pd.to_numeric(ok["ess_selected"], errors="coerce")
    ok["selector_score"] = pd.to_numeric(ok["selector_score"], errors="coerce")
    ok["tail_q99"] = pd.to_numeric(ok["tail_q99"], errors="coerce")

    ok_rows = ok.groupby(keys).size().rename("n_rows_ok")
    if run_col is not None:
        ok_runs = ok.groupby(keys)[run_col].nunique().rename("n_runs_ok")
    else:
        ok_runs = ok_rows.rename("n_runs_ok")

    stats = ok.groupby(keys).agg(
        tail_index_mean=("tail_index", "mean"),
        tail_index_median=("tail_index", "median"),
        tail_index_q25=("tail_index", lambda x: float(np.quantile(x, 0.25))),
        tail_index_q75=("tail_index", lambda x: float(np.quantile(x, 0.75))),
        tail_index_min=("tail_index", "min"),
        tail_index_max=("tail_index", "max"),
        alpha_med_mean=("alpha_med", "mean"),
        alpha_med_median=("alpha_med", "median"),
        alpha_med_q25=("alpha_med", lambda x: float(np.quantile(x, 0.25))),
        alpha_med_q75=("alpha_med", lambda x: float(np.quantile(x, 0.75))),
        alpha_med_min=("alpha_med", "min"),
        alpha_med_max=("alpha_med", "max"),
        alpha_selected_mean=("alpha_selected", "mean"),
        alpha_selected_median=("alpha_selected", "median"),
        k_selected_median=("k_selected", "median"),
        ess_selected_median=("ess_selected", "median"),
        selector_score_median=("selector_score", "median"),
        tail_q99_mean=("tail_q99", "mean"),
    )

    out = pd.concat([total_rows, total_runs, ok_rows, ok_runs, stats], axis=1).reset_index()
    out["n_rows_ok"] = out["n_rows_ok"].fillna(0).astype(int)
    out["n_runs_ok"] = out["n_runs_ok"].fillna(0).astype(int)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IPCW Hill tail diagnostics on saved IPCW component files."
    )
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument(
        "--output-summary",
        required=True,
        help="Output summary file path (.csv/.parquet/.feather).",
    )
    parser.add_argument(
        "--output-detail",
        default=None,
        help="Optional per-k output path (.csv/.parquet/.feather).",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["UNW", "NAT", "VAR", "HPREV2"],
        help="Patterns to run from: UNW NAT VAR DIRECT HPREV2",
    )
    parser.add_argument("--by-regime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--by-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-time", type=int, default=None, help="Keep only rows with t <= max_time.")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--min-m", type=int, default=200)
    parser.add_argument("--min-k-abs", type=int, default=20)
    parser.add_argument("--min-k-frac", type=float, default=0.01)
    parser.add_argument("--max-k-frac", type=float, default=0.10)
    parser.add_argument("--k-points", type=int, default=20)
    parser.add_argument(
        "--tail-index-selector",
        choices=["median", "plateau", "reiss_thomas"],
        default="plateau",
        help="Rule used for the primary tail_index column.",
    )
    parser.add_argument(
        "--reiss-thomas-beta",
        type=float,
        default=0.3,
        help="Power parameter for the Reiss-Thomas-style stability criterion.",
    )
    parser.add_argument("--weight-cap-quantile", type=float, default=None)
    parser.add_argument("--alpha-borderline", type=float, default=1.2)
    parser.add_argument("--alpha-bad", type=float, default=1.0)
    parser.add_argument("--tail-tiny", type=float, default=1e-8)
    parser.add_argument(
        "--n_workers",
        "--n-workers",
        dest="n_workers",
        type=int,
        default=1,
        help="Number of input files/runs to process in parallel. Default: 1.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = diagnose_path_to_outputs(
        args.input,
        args.output_summary,
        out_detail_path=args.output_detail,
        patterns=args.patterns,
        by_regime=args.by_regime,
        by_run=args.by_run,
        max_time=args.max_time,
        eps=args.eps,
        min_m=args.min_m,
        min_k_abs=args.min_k_abs,
        min_k_frac=args.min_k_frac,
        max_k_frac=args.max_k_frac,
        k_points=args.k_points,
        tail_index_selector=args.tail_index_selector,
        reiss_thomas_beta=args.reiss_thomas_beta,
        weight_cap_quantile=args.weight_cap_quantile,
        alpha_borderline=args.alpha_borderline,
        alpha_bad=args.alpha_bad,
        tail_tiny=args.tail_tiny,
        n_workers=args.n_workers,
    )

    print(f"Input files processed: {result['input_files']}")
    print(f"Workers used: {result['workers_used']}")
    print(f"Summary rows: {result['summary_rows']}")
    print(f"Summary output: {args.output_summary}")
    if args.output_detail is not None:
        print(f"Detail rows: {result['detail_rows']}")
        print(f"Detail output: {args.output_detail}")


if __name__ == "__main__":
    main()
