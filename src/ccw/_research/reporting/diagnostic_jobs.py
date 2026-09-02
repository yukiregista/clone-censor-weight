#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ccw._research.reporting import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IPCW diagnostics for runs defined in the paper experiment specification."
    )
    parser.add_argument("--paper-spec", type=Path, default=common.DEFAULT_PAPER_SPEC)
    parser.add_argument("--output-root", type=Path, default=Path("output/experiments"))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=list(common.DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--cutoffs", nargs="+", type=int, default=list(common.DEFAULT_CUTOFFS))
    parser.add_argument("--patterns", nargs="+", default=list(common.DEFAULT_PATTERNS))
    parser.add_argument("--min-m", type=int, default=200)
    parser.add_argument("--tail-index-selector", choices=["median", "plateau", "reiss_thomas"], default="plateau")
    parser.add_argument("--reiss-thomas-beta", type=float, default=0.3)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Workers forwarded to run_ipcw_diagnostics.py for each run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", type=Path, default=None)
    return parser.parse_args()


def build_command(args: argparse.Namespace, run_dir: Path, *, cutoff: int) -> list[str]:
    cmd = [
        sys.executable,
        str(common.PAPER_DIR / "2_run_ipcw_diagnostics.py"),
        str(run_dir),
        "--patterns",
        *[str(pattern) for pattern in args.patterns],
        "--grace-period",
        str(cutoff),
        "--min-m",
        str(args.min_m),
        "--tail-index-selector",
        str(args.tail_index_selector),
        "--reiss-thomas-beta",
        str(args.reiss_thomas_beta),
        "--n_workers",
        str(args.n_workers),
    ]
    return cmd


def manifest_row(
    *,
    job: common.PaperJob,
    status: str,
    run_dir: Path | None = None,
    command: list[str] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    row = {
        "status": status,
        "scenario": job.scenario,
        "experiment": job.experiment,
        "setting": job.setting,
        "sample_size": job.sample_size,
        "cutoff": job.cutoff,
        "run_dir": str(run_dir or ""),
        "command": common.command_text(command or []),
        "reason": reason,
    }
    if job.commit:
        row["commit"] = job.commit
    return row


def main() -> int:
    args = parse_args()
    if args.n_workers < 1:
        raise SystemExit("--n-workers must be at least 1.")

    output_root = common.repo_path(args.output_root)
    specs = common.load_paper_spec(args.paper_spec)
    paper_jobs = common.iter_paper_jobs(
        specs,
        sample_sizes=args.sample_sizes,
        cutoffs=args.cutoffs,
        scenarios=args.scenarios,
        experiments=args.experiments,
        settings=args.settings,
    )

    manifest: list[dict[str, Any]] = []
    runnable: list[tuple[common.PaperJob, Path, list[str]]] = []
    for job in paper_jobs:
        resolved = common.resolve_run(output_root=output_root, job=job)
        if resolved is None:
            manifest.append(manifest_row(job=job, status="missing", reason="missing timestamped run"))
            continue
        if not common.has_ipw_weights(resolved.run_dir):
            status = "missing_ipw_weights"
            manifest.append(manifest_row(job=job, status=status, run_dir=resolved.run_dir, reason=status))
            continue
        cmd = build_command(args, resolved.run_dir, cutoff=job.cutoff)
        manifest.append(manifest_row(job=job, status="ready", run_dir=resolved.run_dir, command=cmd))
        runnable.append((job, resolved.run_dir, cmd))

    if args.write_manifest is not None:
        path = common.write_manifest(args.write_manifest, manifest)
        print(f"Manifest written: {path}")

    print(f"Paper jobs: {len(paper_jobs)}")
    print(f"Runnable diagnostics jobs: {len(runnable)}")
    print(f"Missing/skipped jobs: {sum(1 for row in manifest if row['status'] != 'ready')}")

    if args.dry_run:
        for _, _, cmd in runnable:
            print(common.command_text(cmd))
        return 0

    failures: list[tuple[common.PaperJob, Path, int]] = []
    for index, (job, run_dir, cmd) in enumerate(runnable, start=1):
        completed = common.run_command(cmd)
        common.print_completed(f"[{index}/{len(runnable)}] {run_dir}", completed)
        if completed.returncode != 0:
            failures.append((job, run_dir, completed.returncode))

    if failures:
        first = failures[0]
        raise SystemExit(f"{len(failures)} diagnostics job(s) failed; first failure: {first[1]} rc={first[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
