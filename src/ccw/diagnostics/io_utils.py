from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_table(path: str | Path) -> pd.DataFrame:
    file_path = _normalize_path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)
    raise ValueError(f"Unsupported file extension for read: '{suffix}' ({file_path})")


def write_table(df: pd.DataFrame, path: str | Path, mode: str = "overwrite") -> None:
    file_path = _normalize_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be either 'overwrite' or 'append'.")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        if mode == "append" and file_path.exists():
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)
        return

    out_df = df
    if mode == "append" and file_path.exists():
        existing = read_table(file_path)
        out_df = pd.concat([existing, df], ignore_index=True)

    if suffix == ".parquet":
        out_df.to_parquet(file_path, index=False)
        return
    if suffix == ".feather":
        out_df.reset_index(drop=True).to_feather(file_path)
        return
    raise ValueError(f"Unsupported file extension for write: '{suffix}' ({file_path})")
