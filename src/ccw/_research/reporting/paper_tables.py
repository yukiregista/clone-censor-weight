#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ccw._research.reporting import common


ESTIMANDS = {
    "mortality_control": {
        "strategy": "control",
        "estimate_col": "ccw_outcome_rate_control",
        "coverage_col": "ci_coverage_ccw_outcome_rate_control",
    },
    "mortality_intervention": {
        "strategy": "intervention",
        "estimate_col": "ccw_outcome_rate_intervention",
        "coverage_col": "ci_coverage_ccw_outcome_rate_intervention",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper summary tables from experiment outputs and a paper specification."
    )
    parser.add_argument("--paper-spec", type=Path, default=common.DEFAULT_PAPER_SPEC)
    parser.add_argument("--output-root", type=Path, default=Path("output/experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_diagnostics/summary/tables"))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=list(common.DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--cutoffs", nargs="+", type=int, default=list(common.DEFAULT_CUTOFFS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-print-summary", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_all_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if {"metric_name", "metric_value"}.issubset(df.columns):
        out: dict[str, float] = {}
        for _, row in df.iterrows():
            out[str(row["metric_name"])] = numeric_or_nan(row["metric_value"])
        return out
    if len(df.index):
        row = df.iloc[0]
        return {str(col): numeric_or_nan(row[col]) for col in df.columns}
    return {}


def numeric_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def resolve_path(path_text: str | Path | None) -> Path | None:
    if path_text is None or str(path_text).strip() == "":
        return None
    return common.repo_path(path_text)


def find_ground_truth_path(run_dir: Path, cutoff: int) -> tuple[Path | None, str]:
    info = read_json(run_dir / "ground_truth_cache_info.json")
    cache_path = resolve_path(info.get("cache_path"))
    if cache_path is not None and cache_path.exists():
        return cache_path, "cache_info"

    local_csv = run_dir / "ground_truth_simulation" / "ground_truth_result.csv"
    if local_csv.exists():
        return local_csv, "run_ground_truth"

    setting_cache = run_dir.parent.parent / f"ground_truth_cache_cut{cutoff}" / "artifacts" / "ground_truth.pkl"
    if setting_cache.exists():
        return setting_cache, "setting_cache"

    candidates = sorted(run_dir.rglob("ground_truth_result.csv"))
    if candidates:
        return candidates[0], "nested_ground_truth"
    return None, ""


def load_ground_truth_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as file:
            payload = pickle.load(file)
        df = payload.get("ground_truth_result") if isinstance(payload, dict) else payload
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{path} did not contain a ground_truth_result DataFrame")
        return df.copy()
    return pd.read_csv(path)


def load_truths(path: Path, target_time: int) -> dict[str, float]:
    gt = load_ground_truth_frame(path)
    required = {"time", "strategy", "incident_rate"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    gt = gt.copy()
    gt["time"] = pd.to_numeric(gt["time"], errors="coerce")
    gt["incident_rate"] = pd.to_numeric(gt["incident_rate"], errors="coerce")
    rows = gt[gt["time"] == int(target_time)]
    if rows.empty:
        raise ValueError(f"{path} has no ground-truth rows for time={target_time}")
    return {
        str(row["strategy"]): float(row["incident_rate"])
        for _, row in rows.iterrows()
        if pd.notna(row["strategy"]) and pd.notna(row["incident_rate"])
    }


def finite_series(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return out[np.isfinite(out)]


def summarize_estimates(values: pd.Series, truth: float) -> dict[str, float | int]:
    finite = finite_series(values)
    n = int(finite.shape[0])
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "bias": np.nan, "rmse": np.nan}
    errors = finite - truth if math.isfinite(truth) else pd.Series(np.nan, index=finite.index)
    return {
        "n": n,
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if n > 1 else np.nan,
        "bias": float(errors.mean()) if errors.notna().any() else np.nan,
        "rmse": float(math.sqrt(np.mean(np.square(errors)))) if errors.notna().any() else np.nan,
    }


def base_row(job: common.PaperJob, run: common.ResolvedRun | None = None) -> dict[str, Any]:
    row = {
        "scenario": job.scenario,
        "setting": job.setting,
        "experiment": job.experiment,
        "experiment_display": job.experiment,
        "sample_size": job.sample_size,
        "cutoff": job.cutoff,
        "timestamp": run.timestamp if run is not None else "",
        "run_dir": str(run.run_dir) if run is not None else "",
    }
    if job.commit:
        row["commit"] = job.commit
    return row


def build_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = common.load_paper_spec(args.paper_spec)
    output_root = common.repo_path(args.output_root)
    jobs = common.iter_paper_jobs(
        specs,
        sample_sizes=args.sample_sizes,
        cutoffs=args.cutoffs,
        scenarios=args.scenarios,
        experiments=args.experiments,
        settings=args.settings,
    )

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    truth_cache: dict[tuple[str, int], dict[str, float]] = {}

    for job in jobs:
        run = common.resolve_run(output_root=output_root, job=job)
        row_base = base_row(job, run)
        if run is None:
            missing_rows.append({**row_base, "reason": "missing timestamped run"})
            run_rows.append({**row_base, "status": "missing_run"})
            continue

        params = read_json(run.run_dir / "batch_params.json")
        target_time = int(params.get("cutoff_time_of_observation_display", 30))
        row_base["cutoff_time_of_observation_display"] = target_time
        row_base["analysis_results_csv"] = str(run.analysis_csv)
        row_base["all_metrics_csv"] = str(run.all_metrics_csv)

        if not run.analysis_csv.exists():
            missing_rows.append({**row_base, "reason": "missing analysis_results_detailed.csv"})
            run_rows.append({**row_base, "status": "missing_analysis"})
            continue

        gt_path, gt_source = find_ground_truth_path(run.run_dir, job.cutoff)
        row_base["ground_truth_path"] = str(gt_path) if gt_path is not None else ""
        row_base["ground_truth_source"] = gt_source
        if gt_path is None:
            missing_rows.append({**row_base, "reason": "missing ground truth"})
            run_rows.append({**row_base, "status": "missing_ground_truth"})
            continue

        try:
            truth_key = (str(gt_path), target_time)
            if truth_key not in truth_cache:
                truth_cache[truth_key] = load_truths(gt_path, target_time)
            truths = truth_cache[truth_key]
            analysis = pd.read_csv(
                run.analysis_csv,
                usecols=lambda col: col in {spec["estimate_col"] for spec in ESTIMANDS.values()},
            )
        except Exception as exc:
            missing_rows.append({**row_base, "reason": f"failed to load data: {exc}"})
            run_rows.append({**row_base, "status": "failed"})
            continue

        metrics = load_all_metrics(run.all_metrics_csv)
        wide = {**row_base, "status": "ok"}
        for estimand, spec in ESTIMANDS.items():
            strategy = str(spec["strategy"])
            estimate_col = str(spec["estimate_col"])
            coverage_col = str(spec["coverage_col"])
            truth = numeric_or_nan(truths.get(strategy, np.nan))
            summary = summarize_estimates(
                analysis[estimate_col] if estimate_col in analysis.columns else pd.Series(dtype=float),
                truth,
            )
            coverage = metrics.get(coverage_col, np.nan)
            coverage_count = metrics.get(f"{coverage_col}_count", np.nan)
            coverage_total = metrics.get(f"{coverage_col}_total", np.nan)
            prefix = estimand
            wide[f"{prefix}_truth"] = truth
            wide[f"{prefix}_mean"] = summary["mean"]
            wide[f"{prefix}_estimate_sd"] = summary["sd"]
            wide[f"{prefix}_bias"] = summary["bias"]
            wide[f"{prefix}_rmse"] = summary["rmse"]
            wide[f"{prefix}_coverage"] = coverage
            wide[f"{prefix}_coverage_count"] = coverage_count
            wide[f"{prefix}_coverage_total"] = coverage_total
            wide[f"{prefix}_n_estimates"] = summary["n"]
            long_rows.append(
                {
                    **row_base,
                    "status": "ok",
                    "estimand": estimand,
                    "strategy": strategy,
                    "estimate_col": estimate_col,
                    "coverage_col": coverage_col,
                    "truth": truth,
                    "estimate_mean": summary["mean"],
                    "estimate_sd": summary["sd"],
                    "bias": summary["bias"],
                    "rmse": summary["rmse"],
                    "coverage": coverage,
                    "coverage_count": coverage_count,
                    "coverage_total": coverage_total,
                    "n_estimates": summary["n"],
                }
            )
        wide_rows.append(wide)
        run_rows.append({**row_base, "status": "ok"})

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    run_df = pd.DataFrame(run_rows)
    missing_df = pd.DataFrame(missing_rows)
    sort_cols = ["scenario", "setting", "experiment", "cutoff", "sample_size"]
    for df in (wide_df, long_df, run_df, missing_df):
        if not df.empty:
            df.sort_values([col for col in sort_cols if col in df.columns], inplace=True, ignore_index=True)
    return wide_df, long_df, run_df, missing_df


def main() -> int:
    args = parse_args()
    wide_df, long_df, run_df, missing_df = build_tables(args)
    output_dir = common.repo_path(args.output_dir)

    if not args.no_print_summary:
        print(f"Paper metric rows: {len(wide_df)}")
        print(f"Long metric rows: {len(long_df)}")
        print(f"Run index rows: {len(run_df)}")
        print(f"Missing rows: {len(missing_df)}")

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_dir / "paper_metrics_one_row_per_run.csv", index=False)
    long_df.to_csv(output_dir / "paper_metrics_long.csv", index=False)
    run_df.to_csv(output_dir / "paper_run_index.csv", index=False)
    missing_df.to_csv(output_dir / "paper_missing_runs.csv", index=False)
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
