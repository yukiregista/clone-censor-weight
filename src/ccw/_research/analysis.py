"""Analysis helpers used only by the bundled research workflows."""

from __future__ import annotations

from logging import getLogger

import numpy as np
import pandas as pd

from ccw._logging import TagAdapter, temp_stderr_logging, to_level


_LOGGER = getLogger(__name__)


def simulate_true_effect(
    base_variables,
    n_time,
    strategies,
    treatment_var,
    cutoff_time_of_observation_display=10,
    sample_size=1000,
    seed=123,
    plot_figures=False,
    return_raw_data=False,
    logger_=None,
    verbose=None,
):
    """Simulate counterfactual risks for the paper's treatment strategies."""

    from ccw._research.counterfactual_simulation.simulate_counterfactual import (
        CounterfactualSimulator,
    )
    from ccw._research.data_generation.scenarios.scenario import ScenarioVariables
    from ccw._research.metrics.fulldata_metrics import IncidentRate
    from ccw._research.visualize import visualize

    logger = TagAdapter(logger_ or _LOGGER, tag="simulate_true_effect")
    with temp_stderr_logging(
        logger.logger,
        level=to_level(verbose),
        user_supplied=logger_ is not None,
    ):
        strategy_models = {
            name: {treatment_var: strategy}
            for name, strategy in strategies.items()
        }
        simulator = CounterfactualSimulator(base_variables=base_variables, n_time=n_time)
        simulations = simulator.simulate(sample_size, seed, strategy_models)

        figures = {}
        if plot_figures:
            for name in strategy_models:
                figures[name] = visualize.distribution_plot(
                    sample=simulations[name], bn=simulator.bn[name]
                )

        rows = []
        for time in range(n_time):
            calculator = IncidentRate(ScenarioVariables.D, time)
            incident_arrays = {
                name: calculator.convert_to_incident(values)
                for name, values in simulations.items()
            }
            if time == cutoff_time_of_observation_display:
                logger.info(
                    "95%% confidence interval of incident probability at time %s:",
                    time,
                )
            for name, values in incident_arrays.items():
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                margin = 1.96 * (std / np.sqrt(len(values)))
                if time == cutoff_time_of_observation_display:
                    logger.info(
                        "%s: %.4f (95%% CI: %.4f - %.4f)",
                        name,
                        mean,
                        mean - margin,
                        mean + margin,
                    )
                rows.append([time, name, mean, mean - margin, mean + margin])

        result = pd.DataFrame(
            rows,
            columns=["time", "strategy", "incident_rate", "ci_lower", "ci_upper"],
        )
        at_cutoff = result[result["time"] == cutoff_time_of_observation_display]
        intervention_risk = at_cutoff.loc[
            at_cutoff["strategy"] == "intervention", "incident_rate"
        ].iloc[0]
        control_risk = at_cutoff.loc[
            at_cutoff["strategy"] == "control", "incident_rate"
        ].iloc[0]
        simulated_or = (intervention_risk / (1 - intervention_risk)) / (
            control_risk / (1 - control_risk)
        )

    if return_raw_data:
        return result, simulated_or, figures, simulations
    return result, simulated_or, figures


def _calculate_time_to_events(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    weight_col: str | None = None,
) -> pd.DataFrame:
    subject_data = data.groupby("id", as_index=False)["time"].max()
    treatment_times = (
        data.loc[data[treatment_col] == 1, ["id", "time"]]
        .groupby("id", as_index=False)["time"]
        .min()
        .rename(columns={"time": "time_to_intervention"})
    )
    outcome_times = (
        data.loc[data[outcome_col] == 1, ["id", "time"]]
        .groupby("id", as_index=False)["time"]
        .min()
        .rename(columns={"time": "time_to_outcome"})
    )
    subject_data = subject_data.merge(treatment_times, on="id", how="left").merge(
        outcome_times, on="id", how="left"
    )
    if weight_col is not None:
        weights = data.groupby("id", as_index=False)[weight_col].first()
        subject_data = subject_data.merge(weights, on="id", how="left")
    return subject_data


def _two_group_km_analysis(
    data: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    grace_period: int,
    followup_end: int,
    estimate_at: int,
    exclude_early_outcomes: bool,
    analysis_name: str,
    logger,
    weight_col: str | None,
):
    from lifelines import KaplanMeierFitter

    subject_data = _calculate_time_to_events(
        data, treatment_col, outcome_col, weight_col=weight_col
    )
    if exclude_early_outcomes:
        subject_data = subject_data.loc[
            subject_data["time_to_outcome"].isna()
            | (subject_data["time_to_outcome"] > grace_period)
        ].copy()

    subject_data["group"] = np.where(
        subject_data["time_to_intervention"].notna()
        & (subject_data["time_to_intervention"] <= grace_period),
        "intervention",
        "control",
    )
    subject_data["event"] = subject_data["time_to_outcome"].notna()
    subject_data["duration"] = subject_data["time_to_outcome"].fillna(followup_end)

    outcomes = {}
    for group in ("intervention", "control"):
        group_data = subject_data.loc[subject_data["group"] == group]
        if group_data.empty:
            logger.warning("Empty %s group detected in %s.", group, analysis_name)
            outcomes[group] = np.nan
            continue
        durations = group_data["duration"]
        if estimate_at > durations.max():
            logger.warning(
                "%s group has insufficient follow-up for %s-d estimate in %s.",
                group.capitalize(),
                estimate_at,
                analysis_name,
            )
            outcomes[group] = np.nan
            continue
        fitter = KaplanMeierFitter()
        weights = group_data[weight_col] if weight_col is not None else None
        fitter.fit(
            durations,
            group_data["event"],
            label=f"{group} group",
            weights=weights,
        )
        outcomes[group] = 1 - fitter.predict(estimate_at)

    intervention_risk = outcomes["intervention"]
    control_risk = outcomes["control"]
    if np.isfinite(intervention_risk) and np.isfinite(control_risk):
        difference = intervention_risk - control_risk
        ratio = intervention_risk / control_risk if control_risk != 0 else np.inf
        odds_ratio = (
            (intervention_risk / (1 - intervention_risk))
            / (control_risk / (1 - control_risk))
            if control_risk != 1 and intervention_risk != 1
            else np.inf
        )
        logger.info("%s-d outcome (%s):", estimate_at, analysis_name)
        logger.info(
            "intervention group: %.4f (%.2f%%)",
            intervention_risk,
            intervention_risk * 100,
        )
        logger.info("control group: %.4f (%.2f%%)", control_risk, control_risk * 100)
        logger.info("Rate Difference: %.4f", difference)
        logger.info("Rate Ratio: %.4f", ratio)
        logger.info("Odds Ratio: %.4f", odds_ratio)
    else:
        difference = ratio = odds_ratio = np.nan

    results = {
        "outcome_rate_intervention": intervention_risk,
        "outcome_rate_control": control_risk,
        "outcome_rate_difference": difference,
        "outcome_rate_ratio": ratio,
    }
    return results, odds_ratio


def analysis_with_immortal_time_bias(
    data,
    configs,
    cutoff_time_of_intervention=2,
    cutoff_time_of_observation=30,
    cutoff_time_of_observation_display=10,
    logger_=None,
    verbose=None,
    weight_col: str | None = None,
):
    """Run the paper's intentionally immortal-time-biased comparator."""

    logger = TagAdapter(logger_ or _LOGGER, tag="analysis_with_immortal_time_bias")
    with temp_stderr_logging(
        logger.logger,
        level=to_level(verbose),
        user_supplied=logger_ is not None,
    ):
        return _two_group_km_analysis(
            data,
            treatment_col=configs["treatment_var"].name,
            outcome_col=configs["outcome_var"].name,
            grace_period=cutoff_time_of_intervention,
            followup_end=cutoff_time_of_observation,
            estimate_at=cutoff_time_of_observation_display,
            exclude_early_outcomes=False,
            analysis_name="immortal time bias analysis",
            logger=logger,
            weight_col=weight_col,
        )


def landmark_analysis(
    data,
    configs,
    cutoff_time_of_intervention=2,
    cutoff_time_of_observation=30,
    cutoff_time_of_observation_display=10,
    logger_=None,
    verbose=None,
    weight_col: str | None = None,
):
    """Run the paper's landmark comparator after excluding early outcomes."""

    logger = TagAdapter(logger_ or _LOGGER, tag="landmark_analysis")
    with temp_stderr_logging(
        logger.logger,
        level=to_level(verbose),
        user_supplied=logger_ is not None,
    ):
        return _two_group_km_analysis(
            data,
            treatment_col=configs["treatment_var"].name,
            outcome_col=configs["outcome_var"].name,
            grace_period=cutoff_time_of_intervention,
            followup_end=cutoff_time_of_observation,
            estimate_at=cutoff_time_of_observation_display,
            exclude_early_outcomes=True,
            analysis_name="landmark analysis",
            logger=logger,
            weight_col=weight_col,
        )
