#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = REPO_ROOT / "scripts" / "paper"
DEFAULT_PAPER_SPEC = PAPER_DIR / "experiments.yaml"
DEFAULT_SAMPLE_SIZES = (100, 1000, 10000)
DEFAULT_CUTOFFS = (0, 2, 4)
DEFAULT_PATTERNS = ("VAR", "HPREV2")
@dataclass(frozen=True)
class ScenarioSpec:
    scenario: str
    experiments: tuple[str, ...]
    commits: dict[str, str]
    experiment_settings: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PaperJob:
    scenario: str
    experiment: str
    setting: str
    commit: str | None
    sample_size: int
    cutoff: int


@dataclass(frozen=True)
class ResolvedRun:
    job: PaperJob
    run_dir: Path
    timestamp: str
    analysis_csv: Path
    all_metrics_csv: Path


def repo_path(path: Path | str) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    return out.resolve()


def load_paper_spec(path: Path | str = DEFAULT_PAPER_SPEC) -> list[ScenarioSpec]:
    paper_path = repo_path(path)
    with paper_path.open() as file:
        payload = yaml.safe_load(file) or {}
    specs: list[ScenarioSpec] = []
    for scenario, raw in payload.items():
        experiments = tuple(str(value) for value in (raw or {}).get("experiments", []) if str(value).strip())
        commits = {
            str(setting): str(commit)
            for setting, commit in ((raw or {}).get("commits", {}) or {}).items()
        }
        raw_experiment_settings = (raw or {}).get("experiment_settings", {}) or {}
        experiment_settings = {
            str(experiment): tuple(str(setting) for setting in settings if str(setting).strip())
            for experiment, settings in raw_experiment_settings.items()
        }
        specs.append(
            ScenarioSpec(
                scenario=str(scenario),
                experiments=experiments,
                commits=commits,
                experiment_settings=experiment_settings,
            )
        )
    return specs


def filter_specs(
    specs: Sequence[ScenarioSpec],
    *,
    scenarios: Iterable[str] | None = None,
) -> list[ScenarioSpec]:
    scenario_set = {str(value) for value in scenarios} if scenarios is not None else None
    return [spec for spec in specs if scenario_set is None or spec.scenario in scenario_set]


def iter_paper_jobs(
    specs: Sequence[ScenarioSpec],
    *,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    scenarios: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
    settings: Iterable[str] | None = None,
) -> list[PaperJob]:
    experiment_set = {str(value) for value in experiments} if experiments is not None else None
    setting_set = {str(value) for value in settings} if settings is not None else None
    jobs: list[PaperJob] = []
    for spec in filter_specs(specs, scenarios=scenarios):
        for experiment in spec.experiments:
            if experiment_set is not None and experiment not in experiment_set:
                continue
            experiment_settings = spec.experiment_settings.get(experiment, tuple(spec.commits))
            if not experiment_settings:
                raise ValueError(
                    f"{spec.scenario}/{experiment} has no settings in the paper specification."
                )
            for setting in experiment_settings:
                if setting_set is not None and setting not in setting_set:
                    continue
                if spec.commits and setting not in spec.commits:
                    raise ValueError(
                        f"{spec.scenario}/{experiment} lists setting {setting!r}, "
                        "but no commit is defined for it."
                    )
                commit = spec.commits.get(setting)
                for sample_size in sample_sizes:
                    for cutoff in cutoffs:
                        jobs.append(
                            PaperJob(
                                scenario=spec.scenario,
                                experiment=experiment,
                                setting=setting,
                                commit=commit,
                                sample_size=int(sample_size),
                                cutoff=int(cutoff),
                            )
                        )
    return jobs


def unique_oracle_jobs(jobs: Sequence[PaperJob]) -> list[PaperJob]:
    """Collapse experiment-level jobs to scenario/setting/source/cutoff jobs."""

    seen: set[tuple[str, str, str | None, int]] = set()
    out: list[PaperJob] = []
    for job in jobs:
        key = (job.scenario, job.setting, job.commit, int(job.cutoff))
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


def experiments_for_oracle_job(specs: Sequence[ScenarioSpec], job: PaperJob) -> tuple[str, ...]:
    for spec in specs:
        if spec.scenario == job.scenario:
            if spec.experiment_settings:
                return tuple(
                    experiment
                    for experiment in spec.experiments
                    if job.setting in spec.experiment_settings.get(experiment, ())
                )
            return spec.experiments
    return (job.experiment,)


def timestamp_sort_key(path: Path) -> tuple[int, str]:
    text = path.name
    digits = text.replace("_", "")
    if digits.isdigit():
        return int(digits), text
    return -1, text


def iter_timestamp_dirs(sample_dir: Path, *, include_archive: bool = False) -> list[Path]:
    if not sample_dir.exists():
        return []
    out: list[Path] = []
    for child in sorted((path for path in sample_dir.iterdir() if path.is_dir()), key=timestamp_sort_key):
        if child.name == "archive":
            if include_archive:
                out.extend(sorted((path for path in child.iterdir() if path.is_dir()), key=timestamp_sort_key))
            continue
        out.append(child)
    return out


def latest_run_dir(
    *,
    output_root: Path,
    job: PaperJob,
    include_archive: bool = False,
) -> Path | None:
    sample_dir = output_root
    if job.commit:
        sample_dir /= job.commit
    sample_dir = (
        sample_dir
        / job.setting
        / job.experiment
        / f"N{int(job.sample_size)}_cut{int(job.cutoff)}"
    )
    runs = iter_timestamp_dirs(sample_dir, include_archive=include_archive)
    return runs[-1] if runs else None


def has_ipw_weights(run_dir: Path) -> bool:
    return any(run_dir.rglob("ipw_weights_*.csv"))


def resolve_run(
    *,
    output_root: Path,
    job: PaperJob,
    include_archive: bool = False,
) -> ResolvedRun | None:
    run_dir = latest_run_dir(output_root=output_root, job=job, include_archive=include_archive)
    if run_dir is None:
        return None
    analysis_csv = run_dir / "multi_method_batch_analysis" / "artifacts" / "analysis_results_detailed.csv"
    return ResolvedRun(
        job=job,
        run_dir=run_dir,
        timestamp=run_dir.name,
        analysis_csv=analysis_csv,
        all_metrics_csv=run_dir / "all_metrics.csv",
    )


def command_text(cmd: Sequence[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def write_manifest(path: Path | str, rows: Sequence[dict[str, Any]]) -> Path:
    manifest_path = repo_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest_path


def run_command(cmd: Sequence[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def print_completed(label: str, completed: subprocess.CompletedProcess[str]) -> None:
    status = "ok" if completed.returncode == 0 else f"failed rc={completed.returncode}"
    print(f"{label}: {status}")
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)


def default_config_dir(setting: str) -> Path:
    candidate = PAPER_DIR / "config_overrides" / setting
    if candidate.exists():
        return candidate
    if setting == "a1d1":
        return PAPER_DIR / "config_overrides" / "setting1"
    return candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
