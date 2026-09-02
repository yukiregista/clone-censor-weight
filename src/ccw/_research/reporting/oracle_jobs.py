#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ccw._research.reporting import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute shared oracle IPCW tail indexes for paper scenarios."
    )
    parser.add_argument("--paper-spec", type=Path, default=common.DEFAULT_PAPER_SPEC)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_diagnostics/summary/oracle_tail_index"),
        help="Root for shared oracle outputs.",
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=list(common.DEFAULT_CUTOFFS))
    parser.add_argument("--patterns", nargs="+", default=list(common.DEFAULT_PATTERNS))
    parser.add_argument("--sample-size", type=int, default=10_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123_456)
    parser.add_argument("--min-m", type=int, default=200)
    parser.add_argument("--tail-index-selector", choices=["median", "plateau", "reiss_thomas"], default="plateau")
    parser.add_argument("--reiss-thomas-beta", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", type=Path, default=None)
    return parser.parse_args()


def output_dir_for_job(root: Path, job: common.PaperJob) -> Path:
    output_dir = root / job.scenario / job.setting / f"cut{int(job.cutoff)}"
    return output_dir / f"commit_{job.commit}" if job.commit else output_dir


def build_command(args: argparse.Namespace, job: common.PaperJob, output_dir: Path, config_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(common.PAPER_DIR / "estimate_oracle_tail_index.py"),
        "--experiment",
        job.experiment,
        "--cutoff_time_of_intervention",
        str(job.cutoff),
        "--sample_size",
        str(args.sample_size),
        "--chunk_size",
        str(args.chunk_size),
        "--n_workers",
        str(args.n_workers),
        "--seed",
        str(args.seed),
        "--patterns",
        *[str(pattern) for pattern in args.patterns],
        "--config_dir",
        str(config_dir),
        "--min-m",
        str(args.min_m),
        "--tail-index-selector",
        str(args.tail_index_selector),
        "--reiss-thomas-beta",
        str(args.reiss_thomas_beta),
        "--output_dir",
        str(output_dir),
    ]
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def metadata_for_job(
    *,
    args: argparse.Namespace,
    job: common.PaperJob,
    reused_experiments: tuple[str, ...],
    output_dir: Path,
    config_dir: Path,
    command: list[str],
) -> dict[str, Any]:
    metadata = {
        "scenario": job.scenario,
        "setting": job.setting,
        "cutoff": job.cutoff,
        "representative_experiment": job.experiment,
        "reused_experiments": list(reused_experiments),
        "config_dir": str(config_dir),
        "output_dir": str(output_dir),
        "sample_size": args.sample_size,
        "chunk_size": args.chunk_size,
        "n_workers": args.n_workers,
        "seed": args.seed,
        "patterns": list(args.patterns),
        "command": common.command_text(command),
    }
    if job.commit:
        metadata["commit"] = job.commit
    return metadata


def manifest_row(
    *,
    job: common.PaperJob,
    reused_experiments: tuple[str, ...],
    output_dir: Path,
    config_dir: Path,
    command: list[str] | None,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    row = {
        "status": status,
        "scenario": job.scenario,
        "setting": job.setting,
        "cutoff": job.cutoff,
        "representative_experiment": job.experiment,
        "reused_experiments": ",".join(reused_experiments),
        "config_dir": str(config_dir),
        "output_dir": str(output_dir),
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

    specs = common.load_paper_spec(args.paper_spec)
    base_jobs = common.iter_paper_jobs(
        specs,
        sample_sizes=(0,),
        cutoffs=args.cutoffs,
        scenarios=args.scenarios,
        experiments=args.experiments,
        settings=args.settings,
    )
    oracle_jobs = common.unique_oracle_jobs(base_jobs)
    output_root = common.repo_path(args.output_root)

    manifest: list[dict[str, Any]] = []
    runnable: list[tuple[common.PaperJob, tuple[str, ...], Path, Path, list[str]]] = []
    for job in oracle_jobs:
        reused_experiments = common.experiments_for_oracle_job(specs, job)
        config_dir = common.default_config_dir(job.setting)
        output_dir = output_dir_for_job(output_root, job)
        existing = output_dir / "oracle_tail_index_min.csv"
        if not config_dir.exists():
            manifest.append(
                manifest_row(
                    job=job,
                    reused_experiments=reused_experiments,
                    output_dir=output_dir,
                    config_dir=config_dir,
                    command=None,
                    status="missing_config",
                    reason=f"config directory not found: {config_dir}",
                )
            )
            continue
        command = build_command(args, job, output_dir, config_dir)
        if existing.exists() and not args.overwrite:
            manifest.append(
                manifest_row(
                    job=job,
                    reused_experiments=reused_experiments,
                    output_dir=output_dir,
                    config_dir=config_dir,
                    command=command,
                    status="existing",
                    reason="oracle_tail_index_min.csv exists; use --overwrite to recompute",
                )
            )
            continue
        manifest.append(
            manifest_row(
                job=job,
                reused_experiments=reused_experiments,
                output_dir=output_dir,
                config_dir=config_dir,
                command=command,
                status="ready",
            )
        )
        runnable.append((job, reused_experiments, output_dir, config_dir, command))

    if args.write_manifest is not None:
        path = common.write_manifest(args.write_manifest, manifest)
        print(f"Manifest written: {path}")

    print(f"Shared oracle jobs: {len(oracle_jobs)}")
    print(f"Runnable oracle jobs: {len(runnable)}")
    print(f"Skipped/missing/existing jobs: {sum(1 for row in manifest if row['status'] != 'ready')}")

    if args.dry_run:
        for *_prefix, command in runnable:
            print(common.command_text(command))
        return 0

    failures: list[tuple[common.PaperJob, int]] = []

    def run_one(item: tuple[common.PaperJob, tuple[str, ...], Path, Path, list[str]]):
        job, reused_experiments, output_dir, config_dir, command = item
        completed = common.run_command(command)
        if completed.returncode == 0:
            common.write_json(
                output_dir / "paper_oracle_metadata.json",
                metadata_for_job(
                    args=args,
                    job=job,
                    reused_experiments=reused_experiments,
                    output_dir=output_dir,
                    config_dir=config_dir,
                    command=command,
                ),
            )
        return item, completed

    for index, item in enumerate(runnable, start=1):
        job, _experiments, output_dir, _config_dir, _command = item
        _item, completed = run_one(item)
        common.print_completed(f"[{index}/{len(runnable)}] {output_dir}", completed)
        if completed.returncode != 0:
            failures.append((job, completed.returncode))

    if failures:
        job, returncode = failures[0]
        raise SystemExit(
            f"{len(failures)} oracle job(s) failed; first failure: "
            f"{job.scenario}/{job.setting}/cut{job.cutoff} rc={returncode}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
