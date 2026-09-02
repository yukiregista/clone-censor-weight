#!/usr/bin/env python3
"""CLI wrapper for the reusable batch simulation workflow."""

from __future__ import annotations

import os


def _configure_numeric_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def main() -> object:
    _configure_numeric_threads()
    from ccw._research.workflows.batch import main as batch_main

    return batch_main()


if __name__ == "__main__":
    main()
