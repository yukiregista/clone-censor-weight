import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import ccw

    return ccw, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # A small clone-censor-weight analysis

    This example creates synthetic longitudinal data and compares remaining
    untreated through day 2 with initiating treatment by day 2. It uses 25
    subject-level bootstrap refits so the example remains quick to run.
    """)
    return


@app.cell
def _(np, pd):
    rng = np.random.default_rng(42)
    n_subjects = 200
    n_times = 31

    ages = rng.integers(40, 80, size=n_subjects)
    baseline_severity = rng.normal(size=n_subjects)
    initiation_days = rng.choice([0, 1, 2, -1], size=n_subjects)
    has_outcome = rng.random(n_subjects) < 0.35
    outcome_days = np.where(
        has_outcome,
        rng.integers(4, n_times, size=n_subjects),
        -1,
    )

    records = []
    for patient_id in range(n_subjects):
        for day in range(n_times):
            records.append(
                {
                    "patient_id": patient_id,
                    "day": day,
                    "treatment_started": int(day == initiation_days[patient_id]),
                    "outcome": int(day == outcome_days[patient_id]),
                    "age": ages[patient_id],
                    "severity": baseline_severity[patient_id]
                    + 0.05 * day
                    + rng.normal(scale=0.25),
                }
            )

    longitudinal_data = pd.DataFrame(records)
    longitudinal_data.head(10)
    return (longitudinal_data,)


@app.cell
def _(ccw):
    spec = ccw.DataSpec(
        id="patient_id",
        time="day",
        treatment="treatment_started",
        outcome="outcome",
        baseline=("age",),
        time_varying=("severity",),
    )

    analysis = ccw.CCW(
        spec=spec,
        strategies={
            "control": ccw.NoInitiationThrough(2),
            "intervention": ccw.InitiateBy(2),
        },
        weight_models="C(time) + age + severity",
        followup_end=30,
        estimate_at=30,
        n_bootstrap=25,
        bootstrap_seed=2025,
    )
    return (analysis,)


@app.cell
def _(analysis, longitudinal_data):
    result = analysis.fit(longitudinal_data)
    result.summary()
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    contrast = result.contrast("intervention", "control")
    mo.md(f"""
    The estimated risk difference is
    **{contrast.risk_difference:.3f}**, with bootstrap standard error
    **{contrast.risk_difference_std_error:.3f}** based on
    **{result.n_bootstrap}** refits.
    """)
    return


if __name__ == "__main__":
    app.run()
