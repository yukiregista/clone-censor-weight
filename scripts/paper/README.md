# Paper reproduction workflow

This directory contains the release-facing scripts and configuration needed to
reproduce the paper experiments. Paths identify results by setting, experiment,
sample size, cutoff, and timestamp; they do not depend on a Git commit ID.

## Ordered reproduction steps

Run commands from the repository root:

```bash
scripts/paper/1_experiment.sh \
  --experiment experimentA \
  --cutoff_time_of_intervention 0 2 4 \
  --n_workers 5 \
  --coefA 1 \
  --coefD 1
```

Outputs are stored under:

```text
output/experiments/<setting>/<experiment>/N<size>_cut<cutoff>/<timestamp>/
```

The scenario overrides used by these runs are versioned in
`scripts/paper/config_overrides/`. Only the `--coefA`/`--coefD` combinations
with a matching override directory can be run: `a1d1` (stored as `setting1`),
`a0.5d0.5`, `a0.5d1`, `a0.5d2`, `a1d0.5`, `a1d2`, `a1d4`, `a1d8`, `a2d0.5`,
`a2d1`, `a2d2`, `a4d1`, and `a8d1`, where `a<coefA>d<coefD>`.

The experiment grid is defined in `scripts/paper/experiments.yaml`.

```bash
uv run python scripts/paper/2_run_ipcw_diagnostics.py --n-workers 8
uv run python scripts/paper/3_run_oracle_tail_index.py --n-workers 8
uv run python scripts/paper/4_build_paper_tables.py
uv run marimo edit scripts/paper/5_figures.marimo.py
```

`2_run_ipcw_diagnostics.py` processes one run directory when given a path. With
no path, or with paper-grid options such as `--paper-spec`, it finds every run
requested by `experiments.yaml` and processes them.

`3_run_oracle_tail_index.py` creates shared oracle tail-index outputs under
`output_diagnostics/summary/oracle_tail_index/`. Step 4 builds the persistent
bias, RMSE, and coverage tables under `output_diagnostics/summary/tables/`.

`5_figures.marimo.py` is the release-safe interactive figure notebook. It reads
the step-4 metrics table and writes performance figures under
`output_diagnostics/summary/figures/performance/`. Its defaults can be changed
in `scripts/paper/figure_config.yaml`. Marimo is included in the optional
research dependencies installed by the experiment workflow.
