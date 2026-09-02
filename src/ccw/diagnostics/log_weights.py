from __future__ import annotations

import pandas as pd


def standardize_diag_table(
    df: pd.DataFrame,
    *,
    regime_col: str = "arm",
    time_col: str = "tstart",
    id_col: str = "id",
    g_col: str = "G_t",
    h_prev_col: str = "H_prev",
    s_col: str = "S_t",
) -> pd.DataFrame:
    """
    Convert an exported weight table into the minimal diagnostics schema.
    """
    out = df.copy()

    if "a" not in out.columns:
        if regime_col in out.columns:
            out["a"] = out[regime_col]
        elif "regime" in out.columns:
            out["a"] = out["regime"]
        elif "protocol" in out.columns:
            out["a"] = out["protocol"]
        else:
            out["a"] = "all"

    if "t" not in out.columns:
        if time_col in out.columns:
            out["t"] = out[time_col]
        elif "time" in out.columns:
            out["t"] = out["time"]

    if "id" not in out.columns:
        if id_col in out.columns:
            out["id"] = out[id_col]
        elif "subject_id" in out.columns:
            out["id"] = out["subject_id"]

    if "G_t" not in out.columns:
        if g_col in out.columns:
            out["G_t"] = out[g_col]
        elif "g_t" in out.columns:
            out["G_t"] = out["g_t"]

    if "H_prev" not in out.columns:
        if h_prev_col in out.columns:
            out["H_prev"] = out[h_prev_col]
        elif "h_prev" in out.columns:
            out["H_prev"] = out["h_prev"]
    if "S_t" not in out.columns:
        if s_col in out.columns:
            out["S_t"] = out[s_col]
        else:
            out["S_t"] = 1

    required = {"a", "t", "id", "S_t", "G_t", "H_prev"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing required columns for diagnostics: {sorted(missing)}")

    keep_cols = ["a", "t", "id", "S_t", "G_t", "H_prev"]
    if "run_id" in out.columns:
        keep_cols.append("run_id")
    if "CENSOR_tstart" in out.columns:
        keep_cols.append("CENSOR_tstart")
    return out[keep_cols].copy()
