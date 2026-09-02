#!/usr/bin/env python3
"""Run IPCW diagnostics for one run or for the full paper experiment grid."""

from __future__ import annotations

import sys

from ccw._research.reporting import diagnostic_jobs, ipcw_diagnostics


_BATCH_ONLY_FLAGS = {
    "--paper-spec",
    "--output-root",
    "--scenarios",
    "--experiments",
    "--settings",
    "--sample-sizes",
    "--cutoffs",
    "--dry-run",
    "--write-manifest",
}


def _is_batch_invocation(argv: list[str]) -> bool:
    if not argv:
        return True
    if any(arg in _BATCH_ONLY_FLAGS for arg in argv):
        return True
    return argv[0].startswith("-")


def main() -> int:
    if _is_batch_invocation(sys.argv[1:]):
        return diagnostic_jobs.main()
    ipcw_diagnostics.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
