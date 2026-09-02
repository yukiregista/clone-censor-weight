"""Private implementation of the public real-data estimator."""

from __future__ import annotations

import ast
import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
from patsy import ModelDesc

from ._bootstrap import subject_level_bootstrap
from ._censoring import CensoringModel
from ._result import CCWResult
from ._schema import DataSpec
from .strategies import TreatmentStrategy


def _normalize_weight_models(
    weight_models: str | Mapping[str, str | Mapping[str, str]],
    *,
    strategy_names: tuple[str, ...],
    censoring_model: CensoringModel,
    censoring_columns: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    if isinstance(weight_models, str):
        formulas: dict[str, str | Mapping[str, str]] = {
            name: weight_models for name in strategy_names
        }
    elif isinstance(weight_models, Mapping):
        formulas = dict(weight_models)
    else:
        raise TypeError("weight_models must be a formula string or an arm-to-formula mapping.")
    if set(formulas) != set(strategy_names):
        raise ValueError("weight_models must define exactly the configured strategy names.")

    if censoring_model is CensoringModel.JOINT:
        required_components = ("all",)
    elif censoring_model is CensoringModel.SEPARATE:
        required_components = ("artificial_censor", *censoring_columns)
    elif censoring_model is CensoringModel.PROTOCOL_ONLY:
        required_components = ("artificial_censor",)
    else:
        required_components = ("treatment", *censoring_columns)

    normalized: dict[str, dict[str, str]] = {}
    for strategy_name, specification in formulas.items():
        if isinstance(specification, str):
            components = {name: specification for name in required_components}
        elif isinstance(specification, Mapping):
            components = dict(specification)
        else:
            raise TypeError(
                f"The {strategy_name!r} weight model must be a formula string "
                "or a component-to-formula mapping."
            )
        if set(components) != set(required_components):
            expected = ", ".join(repr(name) for name in required_components)
            raise ValueError(
                f"The {strategy_name!r} weight model must define exactly: {expected}."
            )
        for component, formula in components.items():
            if not isinstance(formula, str) or not formula.strip():
                raise ValueError(
                    f"The {strategy_name!r}/{component!r} weight model must be "
                    "a non-empty formula."
                )
            if "~" in formula:
                raise ValueError("Weight models must contain only the formula right-hand side.")
        normalized[strategy_name] = {
            component: formula.strip() for component, formula in components.items()
        }
    return normalized


_FORMULA_SYMBOLS = {"grace_end", "np"}


def _formula_column_names(formula: str) -> set[str]:
    """Return bare column references from a Patsy right-hand-side formula."""

    description = ModelDesc.from_formula(f"~ {formula}")
    names: set[str] = set()
    for term in description.rhs_termlist:
        for factor in term.factors:
            tree = ast.parse(factor.code, mode="eval")
            called_names = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            names.update(
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and node.id not in _FORMULA_SYMBOLS | called_names
            )
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Q"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    names.add(node.args[0].value)
    return names


def _validate_weight_model_columns(
    models: Mapping[str, Mapping[str, str]], spec: DataSpec
) -> None:
    generated = {
        "tstart",
        "tend",
        "time",
        *(f"{name}_baseline" for name in spec.time_varying),
    }
    available = (set(spec.required_columns) - {spec.id, spec.time}) | {
        "id",
        "time",
        *generated,
    }
    for strategy, components in models.items():
        for component, formula in components.items():
            unknown = sorted(_formula_column_names(formula) - available)
            if unknown:
                raise ValueError(
                    f"The {strategy!r}/{component!r} weight model references "
                    f"undeclared columns: {unknown}. Declare covariates in "
                    "DataSpec.baseline or DataSpec.time_varying."
                )


def _normalize_censoring_at_baseline(
    value: bool | Mapping[str, bool], censoring_columns: tuple[str, ...]
) -> dict[str, bool]:
    if isinstance(value, bool):
        return {column: value for column in censoring_columns}
    if not isinstance(value, Mapping):
        raise TypeError("censoring_at_baseline must be a boolean or a column mapping.")
    normalized = dict(value)
    if set(normalized) != set(censoring_columns):
        raise ValueError(
            "censoring_at_baseline must define exactly the observed censoring columns."
        )
    if any(not isinstance(item, bool) for item in normalized.values()):
        raise TypeError("censoring_at_baseline values must be boolean.")
    return normalized


@dataclass(frozen=True, slots=True)
class CCW:
    """Configure a clone-censor-weight analysis.

    Parameters
    ----------
    spec : DataSpec
        Column roles and validation rules for the longitudinal input data.
    strategies : mapping of str to TreatmentStrategy
        Named treatment strategies to compare. At least two strategies are
        required. Names are arbitrary; ``"control"`` and ``"intervention"``
        additionally enable the corresponding convenience properties on
        :class:`ccw.CCWResult`.
    weight_models : str or mapping
        Right-hand side of the censoring-weight model formula. A string is
        used for every strategy and required censoring component. A mapping
        must define every strategy. For ``censoring_model="separate"``, each
        strategy may map ``"artificial_censor"`` and every observed censoring
        column to separate formulas. For
        ``censoring_model="treatment_probability"``, replace
        ``"artificial_censor"`` with ``"treatment"``.
    followup_end : int
        Last follow-up time included in the analysis.
    estimate_at : int, optional
        Time at which risks and contrasts are estimated. Defaults to
        ``followup_end``.
    random_state : int, default=4
        Seed used by the numerical analysis engine.
    censoring_model : {"joint", "separate", "protocol_only", "treatment_probability"}, default="joint"
        How protocol-deviation and observed-censoring probabilities are
        modeled. See :class:`ccw.CensoringModel`.
    censoring_at_baseline : bool or mapping of str to bool, default=False
        Whether each observed censoring process can occur at time zero. A
        boolean applies to every observed censoring column; a mapping permits
        column-specific values. This affects separate censoring models.
    n_bootstrap : int, default=0
        Number of subject-level bootstrap refits used to estimate standard
        errors. Zero disables bootstrapping and returns point estimates only.
        Use at least two refits when bootstrapping.
    bootstrap_seed : int, default=2025
        Seed used only to generate bootstrap samples.
    verbose : int, optional
        Verbosity forwarded to the numerical analysis engine.

    Notes
    -----
    A configured estimator is immutable. Calling :meth:`fit` does not modify
    the input data or the process-wide NumPy and Python random states.
    """

    spec: DataSpec
    strategies: Mapping[str, TreatmentStrategy]
    weight_models: str | Mapping[str, str | Mapping[str, str]]
    followup_end: int
    estimate_at: int | None = None
    random_state: int = 4
    censoring_model: CensoringModel | str = CensoringModel.JOINT
    censoring_at_baseline: bool | Mapping[str, bool] = False
    n_bootstrap: int = 0
    bootstrap_seed: int = 2025
    verbose: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DataSpec):
            raise TypeError("spec must be a ccw.DataSpec instance.")
        strategies = dict(self.strategies)
        if len(strategies) < 2:
            raise ValueError("strategies must define at least two named strategies.")
        if any(not isinstance(name, str) or not name.strip() for name in strategies):
            raise ValueError("Strategy names must be non-empty strings.")
        try:
            censoring_model = CensoringModel(self.censoring_model)
        except ValueError as error:
            choices = ", ".join(repr(item.value) for item in CensoringModel)
            raise ValueError(f"censoring_model must be one of: {choices}.") from error
        if (
            isinstance(self.followup_end, bool)
            or not isinstance(self.followup_end, int)
            or self.followup_end < 1
        ):
            raise ValueError("followup_end must be a positive integer.")
        estimate_at = self.followup_end if self.estimate_at is None else self.estimate_at
        if (
            isinstance(estimate_at, bool)
            or not isinstance(estimate_at, int)
            or not 1 <= estimate_at <= self.followup_end
        ):
            raise ValueError("estimate_at must satisfy 1 <= estimate_at <= followup_end.")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or not 0 <= self.random_state < 2**32
        ):
            raise ValueError("random_state must be an integer from 0 through 2**32 - 1.")
        if (
            isinstance(self.n_bootstrap, bool)
            or not isinstance(self.n_bootstrap, int)
            or self.n_bootstrap < 0
            or self.n_bootstrap == 1
        ):
            raise ValueError("n_bootstrap must be zero or an integer of at least two.")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or not 0 <= self.bootstrap_seed < 2**32
        ):
            raise ValueError("bootstrap_seed must be an integer from 0 through 2**32 - 1.")
        for arm, strategy in strategies.items():
            if not callable(getattr(strategy, "artificial_censor", None)) or not callable(
                getattr(strategy, "censoring_prob_mask", None)
            ):
                raise TypeError(
                    f"The {arm!r} strategy must implement artificial_censor and "
                    "censoring_prob_mask."
                )
            grace_period = getattr(strategy, "grace_period", None)
            if (
                isinstance(grace_period, bool)
                or not isinstance(grace_period, int)
                or grace_period < 0
            ):
                raise ValueError(
                    f"The {arm!r} strategy grace_period must be a non-negative integer."
                )
            if grace_period >= estimate_at:
                raise ValueError(
                    f"The {arm!r} strategy grace period must be earlier than estimate_at."
                )
            if censoring_model is CensoringModel.TREATMENT_PROBABILITY and (
                not callable(getattr(strategy, "treatment_prob_mask", None))
                or not callable(
                    getattr(strategy, "convert_treatment_prob_to_ipcw", None)
                )
            ):
                raise TypeError(
                    f"The {arm!r} strategy must implement treatment_prob_mask and "
                    "convert_treatment_prob_to_ipcw when censoring_model is "
                    "'treatment_probability'."
                )
        strategy_names = tuple(strategies)
        normalized_models = _normalize_weight_models(
            self.weight_models,
            strategy_names=strategy_names,
            censoring_model=censoring_model,
            censoring_columns=self.spec.censoring,
        )
        _validate_weight_model_columns(normalized_models, self.spec)
        public_models = {
            name: MappingProxyType(components)
            for name, components in normalized_models.items()
        }
        censoring_at_baseline = _normalize_censoring_at_baseline(
            self.censoring_at_baseline, self.spec.censoring
        )
        object.__setattr__(self, "strategies", MappingProxyType(strategies))
        object.__setattr__(self, "weight_models", MappingProxyType(public_models))
        object.__setattr__(self, "estimate_at", estimate_at)
        object.__setattr__(self, "censoring_model", censoring_model)
        object.__setattr__(
            self,
            "censoring_at_baseline",
            MappingProxyType(censoring_at_baseline),
        )

    @property
    def grace_periods(self) -> dict[str, int]:
        """Return the protocol grace period for every strategy.

        Returns
        -------
        dict of str to int
            Strategy names mapped to their non-negative protocol deadlines.
        """

        return {
            name: int(strategy.grace_period)
            for name, strategy in self.strategies.items()
        }

    def fit(self, data: pd.DataFrame) -> CCWResult:
        """Fit the configured analysis.

        Parameters
        ----------
        data : pandas.DataFrame
            Discrete-time longitudinal observations. The columns and their
            interpretation are defined by ``spec``.

        Returns
        -------
        CCWResult
            Estimated risks, contrasts, row-level weights, and
            weight-diagnostic helpers. Bootstrap standard errors and
            replicate estimates are included when ``n_bootstrap`` is
            positive.

        Raises
        ------
        TypeError
            If ``data`` is not a pandas data frame.
        ValueError
            If the data violate the configured schema or do not extend to
            ``estimate_at``.

        Notes
        -----
        The input data frame is copied before validation and analysis.
        """

        # These imports are deliberately lazy so ``import ccw`` does not load
        # the simulation and plotting stack used by the research workflow.
        from ._core.censor_weight import (
            simple_all_at_once,
            simple_ignore_censor_cols,
            simple_separate,
            use_treatment_prob,
        )
        from ._pipeline import cloning_censoring_weighting_analysis

        weighting_functions = {
            CensoringModel.JOINT: simple_all_at_once,
            CensoringModel.SEPARATE: simple_separate,
            CensoringModel.PROTOCOL_ONLY: simple_ignore_censor_cols,
            CensoringModel.TREATMENT_PROBABILITY: use_treatment_prob,
        }

        frame = self.spec.prepare(data)
        if int(frame["time"].max()) < int(self.estimate_at):
            raise ValueError(
                f"estimate_at={self.estimate_at} exceeds the maximum observed time "
                f"({int(frame['time'].max())})."
            )
        configs = {
            "treatment_var": SimpleNamespace(name=self.spec.treatment),
            "outcome_var": SimpleNamespace(name=self.spec.outcome),
            "censor_vars": [SimpleNamespace(name=name) for name in self.spec.censoring] or None,
            "censor_day0": [
                self.censoring_at_baseline[name] for name in self.spec.censoring
            ]
            or None,
            "time_varying_vars": [
                SimpleNamespace(name=name) for name in self.spec.time_varying
            ]
            or None,
            "ipcw_func": weighting_functions[self.censoring_model],
        }
        def fit_prepared(
            prepared: pd.DataFrame,
            *,
            weight_col: str | None,
            return_weights: bool,
        ):
            return cloning_censoring_weighting_analysis(
                df_joined=prepared,
                strategies=dict(self.strategies),
                ipw_explanatory_formula={
                    name: dict(components)
                    for name, components in self.weight_models.items()
                },
                configs=configs,
                cutoff_time_of_intervention=self.grace_periods,
                cutoff_time_of_observation=self.followup_end,
                cutoff_time_of_observation_display=int(self.estimate_at),
                seed=self.random_state,
                verbose=self.verbose,
                weight_col=weight_col,
                return_weight_df=return_weights,
            )

        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            output = fit_prepared(
                frame,
                weight_col=self.spec.sample_weight,
                return_weights=True,
            )
            bootstrap_results = self._bootstrap(
                frame,
                fit_prepared=fit_prepared,
            )
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        if output.weights is None:
            raise RuntimeError("The analysis pipeline did not return row-level weights.")
        return CCWResult(
            _estimates=dict(output.estimates),
            _weight_data=output.weights.copy(deep=True),
            _data_spec=self.spec,
            _strategy_names=tuple(self.strategies),
            _bootstrap_results=bootstrap_results,
        )

    def _bootstrap(self, frame: pd.DataFrame, *, fit_prepared) -> pd.DataFrame | None:
        if self.n_bootstrap == 0:
            return None

        failures = 0

        def statistic(bootstrap_frame: pd.DataFrame) -> np.ndarray:
            nonlocal failures
            try:
                replicate = fit_prepared(
                    bootstrap_frame.reset_index(drop=True),
                    weight_col=self.spec.sample_weight,
                    return_weights=False,
                )
                return np.asarray(
                    [
                        replicate.estimates[f"outcome_rate_{name}"]
                        for name in self.strategies
                    ],
                    dtype=float,
                )
            except Exception:  # noqa: BLE001 - numerical bootstrap refits may fail
                failures += 1
                return np.full(len(self.strategies), np.nan)

        distribution = subject_level_bootstrap(
            frame,
            id_col="id",
            n_resamples=self.n_bootstrap,
            seed=self.bootstrap_seed,
            statistic=statistic,
        )

        if failures:
            warnings.warn(
                f"{failures} of {self.n_bootstrap} bootstrap refits failed; "
                "standard errors use the successful refits.",
                RuntimeWarning,
                stacklevel=2,
            )
        results = pd.DataFrame(distribution.values, columns=self.strategies)
        results.index.name = "iteration"
        return results
