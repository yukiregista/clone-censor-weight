#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
DIAGNOSTICS_ROOT = Path(
    os.environ.get("CCW_DIAGNOSTICS_ROOT", Path.cwd().resolve() / "output_diagnostics")
).expanduser().resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from ccw.diagnostics.io_utils import read_table  # noqa: E402
from ccw.diagnostics.run_diagnosis import diagnose_path_to_outputs, summarize_over_runs  # noqa: E402


def _looks_like_ipw_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file() and p.name.startswith("ipw_weights_") and p.suffix.lower() in {".csv", ".parquet", ".feather"}:
            return True
    return False


def _infer_batch_root_from_ipw_dir(ipw_dir: Path) -> Path:
    # .../<batch>/artifacts/ipw_weights
    if ipw_dir.name == "ipw_weights" and ipw_dir.parent.name == "artifacts":
        maybe_parent = ipw_dir.parent.parent
        # .../<batch>/multi_method_batch_analysis/artifacts/ipw_weights
        if maybe_parent.name == "multi_method_batch_analysis":
            return maybe_parent.parent
        return maybe_parent
    return ipw_dir.parent


def _find_ipw_dir_under(base_path: Path) -> Path | None:
    preferred = [
        base_path / "artifacts" / "ipw_weights",
        base_path / "multi_method_batch_analysis" / "artifacts" / "ipw_weights",
    ]
    for cand in preferred:
        if cand.exists() and cand.is_dir() and _looks_like_ipw_dir(cand):
            return cand

    for cand in sorted(base_path.rglob("ipw_weights")):
        if cand.is_dir() and _looks_like_ipw_dir(cand):
            return cand
    return None


def _diagnostics_relative_dir(batch_root: Path) -> Path:
    parts = batch_root.parts
    if "output" in parts:
        idx = parts.index("output")
        return Path(*parts[idx + 1 :])

    if batch_root.is_absolute():
        return batch_root.relative_to(batch_root.anchor)

    return batch_root


def _mirror_diagnostics_dir(batch_root: Path) -> Path:
    rel = _diagnostics_relative_dir(batch_root)
    return DIAGNOSTICS_ROOT / rel / "ipcw_diagnostics"


def _resolve_paths(base_path: Path) -> tuple[Path, Path]:
    """
    Return (input_path, output_dir) from a single user-provided path.
    """
    if base_path.is_dir():
        found_ipw = _find_ipw_dir_under(base_path)
        if found_ipw is not None:
            batch_root = _infer_batch_root_from_ipw_dir(found_ipw)
            return found_ipw, _mirror_diagnostics_dir(batch_root)

        if base_path.name == "ipw_weights":
            batch_root = _infer_batch_root_from_ipw_dir(base_path)
            return base_path, _mirror_diagnostics_dir(batch_root)

        if _looks_like_ipw_dir(base_path):
            batch_root = _infer_batch_root_from_ipw_dir(base_path)
            return base_path, _mirror_diagnostics_dir(batch_root)

        return base_path, _mirror_diagnostics_dir(base_path)

    if base_path.parent.name == "ipw_weights":
        batch_root = _infer_batch_root_from_ipw_dir(base_path.parent)
        return base_path, _mirror_diagnostics_dir(batch_root)

    return base_path, _mirror_diagnostics_dir(base_path.parent)


def _single_tail_index_from_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy()
    keys = [c for c in ("run_id", "a", "pattern") if c in df.columns]
    if not keys:
        keys = ["pattern"] if "pattern" in df.columns else []
    if not keys:
        return pd.DataFrame()

    tail_col = "tail_index"
    if tail_col not in df.columns:
        if "alpha_selected" in df.columns:
            df[tail_col] = pd.to_numeric(df["alpha_selected"], errors="coerce")
            df[tail_col] = df[tail_col].fillna(pd.to_numeric(df.get("alpha_med"), errors="coerce"))
        else:
            df[tail_col] = pd.to_numeric(df.get("alpha_med"), errors="coerce")

    valid = (
        (df.get("status", pd.Series(index=df.index, dtype=object)) == "ok")
        & pd.to_numeric(df[tail_col], errors="coerce").notna()
    )
    df_valid = df.loc[valid].copy()
    if df_valid.empty:
        out = df[keys].drop_duplicates().copy()
        out["alpha_single"] = np.nan
        out["alpha_time_min"] = np.nan
        out["alpha_time_max"] = np.nan
        out["n_time_ok"] = 0
        out["flag_any"] = pd.NA
        return out

    grouped = df_valid.groupby(keys, dropna=False, sort=True)
    out = grouped.agg(
        alpha_single=(tail_col, "median"),
        alpha_time_min=(tail_col, "min"),
        alpha_time_max=(tail_col, "max"),
        n_time_ok=("t", "nunique"),
        flag_any=("flag", "max"),
    ).reset_index()
    return out


def _infer_grace_period(batch_root: Path) -> int | None:
    batch_params = batch_root / "batch_params.json"
    if batch_params.exists():
        try:
            payload = json.loads(batch_params.read_text())
            if "cutoff_time_of_intervention" in payload:
                return int(payload["cutoff_time_of_intervention"])
        except Exception:
            pass

    run_meta = batch_root / "_run_meta.txt"
    if run_meta.exists():
        try:
            for line in run_meta.read_text().splitlines():
                if line.startswith("cutoff_time_of_intervention="):
                    return int(line.split("=", 1)[1].strip())
        except Exception:
            pass

    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run IPCW Hill diagnostics with a single path argument.\n"
            "You can pass either a batch output folder or an ipw_weights folder."
        )
    )
    parser.add_argument(
        "path",
        help="Batch output directory (recommended) or direct ipw_weights path.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["UNW", "NAT", "VAR", "HPREV2"],
        help="Patterns to run from: UNW NAT VAR DIRECT HPREV2 (default: includes HPREV2).",
    )
    parser.add_argument("--by-regime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--by-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-m", type=int, default=200)
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
    parser.add_argument(
        "--grace-period",
        type=int,
        default=None,
        help="Maximum time t to include. If omitted, inferred from batch metadata.",
    )
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

    raw_path = Path(args.path).expanduser().resolve()
    input_path, output_dir = _resolve_paths(raw_path)
    summary_path = output_dir / "hill_summary.csv"
    detail_path = output_dir / "hill_by_k.csv"
    avg_path = output_dir / "tail_index_over_runs.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    batch_root = output_dir.parent
    grace_period = args.grace_period
    if grace_period is None:
        grace_period = _infer_grace_period(batch_root)

    result = diagnose_path_to_outputs(
        in_path=input_path,
        out_summary_path=summary_path,
        out_detail_path=detail_path,
        patterns=args.patterns,
        by_regime=args.by_regime,
        by_run=args.by_run,
        max_time=grace_period,
        min_m=args.min_m,
        tail_index_selector=args.tail_index_selector,
        reiss_thomas_beta=args.reiss_thomas_beta,
        weight_cap_quantile=args.weight_cap_quantile,
        n_workers=args.n_workers,
    )
    summary_df = read_table(summary_path)
    avg_df = summarize_over_runs(summary_df)
    avg_path.parent.mkdir(parents=True, exist_ok=True)
    avg_df.to_csv(avg_path, index=False)

    single_df = _single_tail_index_from_summary(summary_df)
    single_path = output_dir / "single_tail_index.csv"
    single_path.parent.mkdir(parents=True, exist_ok=True)
    single_df.to_csv(single_path, index=False)

    print(f"Input path: {input_path}")
    print(f"Input files processed: {result['input_files']}")
    print(f"Workers used: {result['workers_used']}")
    print(f"Grace period (max t): {grace_period if grace_period is not None else 'not found; all t included'}")
    print(f"Summary: {summary_path} (rows={result['summary_rows']})")
    print(f"By-k: {detail_path} (rows={result['detail_rows']})")
    print(f"Avg over runs: {avg_path} (rows={len(avg_df)})")
    print(f"Single: {single_path} (rows={len(single_df)})")


if __name__ == "__main__":
    main()
