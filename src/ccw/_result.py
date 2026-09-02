"""Private result implementation for the public CCW API."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._bootstrap import bootstrap_standard_error
from ._schema import DataSpec


def _risk_odds(risk: float) -> float:
    if risk == 1.0:
        return float("inf")
    return risk / (1.0 - risk)


@dataclass(frozen=True, slots=True, repr=False)
class CCWContrast:
    """Estimated contrast between two treatment strategies.

    Parameters
    ----------
    comparison : str
        Name of the strategy in the numerator or minuend.
    reference : str
        Name of the reference strategy.
    comparison_risk : float
        Estimated outcome risk under ``comparison``.
    reference_risk : float
        Estimated outcome risk under ``reference``.
    risk_difference : float
        Estimated ``comparison_risk - reference_risk``.
    risk_ratio : float
        Estimated ``comparison_risk / reference_risk``.
    log_risk_ratio : float
        Natural logarithm of the estimated risk ratio.
    odds_ratio : float
        Estimated comparison-versus-reference risk odds ratio.
    risk_difference_std_error : float, optional
        Estimated bootstrap standard error of the risk difference, or
        ``None`` when bootstrapping was not requested.
    risk_ratio_std_error : float, optional
        Estimated bootstrap standard error of the risk ratio, or ``None``
        when bootstrapping was not requested.
    log_risk_ratio_std_error : float, optional
        Estimated bootstrap standard error of the log risk ratio, or ``None``
        when bootstrapping was not requested.
    odds_ratio_std_error : float, optional
        Estimated bootstrap standard error of the odds ratio, or ``None``
        when bootstrapping was not requested.

    """

    comparison: str
    reference: str
    comparison_risk: float
    reference_risk: float
    risk_difference: float
    risk_ratio: float
    log_risk_ratio: float
    odds_ratio: float
    risk_difference_std_error: float | None
    risk_ratio_std_error: float | None
    log_risk_ratio_std_error: float | None
    odds_ratio_std_error: float | None

    def __repr__(self) -> str:
        fields = [
            f"comparison={self.comparison!r}",
            f"reference={self.reference!r}",
            f"comparison_risk={self.comparison_risk!r}",
            f"reference_risk={self.reference_risk!r}",
            f"risk_difference={self.risk_difference!r}",
            f"risk_ratio={self.risk_ratio!r}",
            f"log_risk_ratio={self.log_risk_ratio!r}",
            f"odds_ratio={self.odds_ratio!r}",
        ]
        for name in (
            "risk_difference_std_error",
            "risk_ratio_std_error",
            "log_risk_ratio_std_error",
            "odds_ratio_std_error",
        ):
            value = getattr(self, name)
            if value is not None:
                fields.append(f"{name}={value!r}")
        return f"CCWContrast({', '.join(fields)})"


@dataclass(frozen=True, slots=True)
class CCWResult:
    """Fitted clone-censor-weight analysis results.
    """

    _estimates: Mapping[str, Any]
    _weight_data: pd.DataFrame
    _data_spec: DataSpec
    _strategy_names: tuple[str, ...] = ("control", "intervention")
    _bootstrap_results: pd.DataFrame | None = None

    @property
    def has_bootstrap(self) -> bool:
        """Return whether the analysis includes bootstrap refits.

        Returns
        -------
        bool
            ``True`` when bootstrap replicate estimates are available.
        """

        return self._bootstrap_results is not None

    @property
    def n_bootstrap(self) -> int:
        """Return the number of requested bootstrap refits.

        Returns
        -------
        int
            Number of bootstrap replicate rows, or zero without bootstrapping.
        """

        return 0 if self._bootstrap_results is None else len(self._bootstrap_results)

    @property
    def bootstrap_results(self) -> pd.DataFrame | None:
        """Return strategy-risk estimates from each bootstrap refit.

        Returns
        -------
        pandas.DataFrame or None
            One row per bootstrap replicate and one column per strategy, or
            ``None`` when bootstrapping was not requested. Failed refits are
            represented by missing values.
        """

        if self._bootstrap_results is None:
            return None
        return self._bootstrap_results.copy(deep=True)

    @property
    def strategy_names(self) -> tuple[str, ...]:
        """Return strategy names in their configured order.

        Returns
        -------
        tuple of str
            Names of all fitted treatment strategies.
        """

        return self._strategy_names

    @property
    def risks(self) -> dict[str, float]:
        """Return estimated outcome risks by treatment strategy.

        Returns
        -------
        dict of str to float
            Estimated risk at the configured estimation time for every
            strategy.
        """

        return {name: self.risk(name) for name in self.strategy_names}

    @property
    def risk_std_errors(self) -> dict[str, float] | None:
        """Return estimated bootstrap risk standard errors by strategy.

        Returns
        -------
        dict of str to float or None
            Estimated bootstrap standard error for every strategy, or
            ``None`` when bootstrapping was not requested.
        """

        if not self.has_bootstrap:
            return None
        return {name: self.risk_std_error(name) for name in self.strategy_names}

    def risk(self, strategy: str) -> float:
        """Return the estimated outcome risk for one strategy.

        Parameters
        ----------
        strategy : str
            Configured treatment-strategy name.

        Returns
        -------
        float
            Estimated risk at the configured estimation time.

        Raises
        ------
        KeyError
            If ``strategy`` was not fitted.
        """

        self._validate_strategy(strategy)
        return float(self._estimates[f"outcome_rate_{strategy}"])

    def risk_std_error(self, strategy: str) -> float | None:
        """Return the estimated bootstrap standard error of a strategy's risk.

        Parameters
        ----------
        strategy : str
            Configured treatment-strategy name.

        Returns
        -------
        float or None
            Estimated bootstrap standard error, or ``None`` when
            bootstrapping was not requested.

        Raises
        ------
        KeyError
            If ``strategy`` was not fitted.
        """

        self._validate_strategy(strategy)
        if self._bootstrap_results is None:
            return None
        return self._bootstrap_std(self._bootstrap_results[strategy].to_numpy())

    @staticmethod
    def _bootstrap_std(values) -> float:
        return bootstrap_standard_error(values)

    def _validate_strategy(self, strategy: str) -> None:
        if strategy not in self.strategy_names:
            available = ", ".join(repr(name) for name in self.strategy_names)
            raise KeyError(f"Unknown strategy {strategy!r}; available strategies: {available}.")

    def contrast(self, comparison: str, reference: str) -> CCWContrast:
        """Return estimated contrasts between two strategies.

        Parameters
        ----------
        comparison : str
            Strategy used as the numerator or minuend.
        reference : str
            Strategy used as the denominator or subtrahend.

        Returns
        -------
        CCWContrast
            Estimated risks, risk difference, risk ratio, odds ratio, and
            bootstrap standard errors when available.
        """

        self._validate_strategy(comparison)
        self._validate_strategy(reference)
        if comparison == reference:
            raise ValueError("comparison and reference must name different strategies.")
        comparison_risk = self.risk(comparison)
        reference_risk = self.risk(reference)
        difference = comparison_risk - reference_risk
        if reference_risk > 0:
            ratio = comparison_risk / reference_risk
        else:
            ratio = float("inf") if comparison_risk > 0 else 1.0
        if ratio > 0:
            log_ratio = math.log(ratio)
        else:
            log_ratio = float("nan")
        comparison_odds = _risk_odds(comparison_risk)
        reference_odds = _risk_odds(reference_risk)
        if reference_odds > 0:
            odds_ratio = comparison_odds / reference_odds
        else:
            odds_ratio = float("inf") if comparison_odds > 0 else 1.0

        rd_std = None
        rr_std = None
        log_rr_std = None
        odds_ratio_std = None
        if self._bootstrap_results is not None:
            comparison_values = self._bootstrap_results[comparison].to_numpy(float)
            reference_values = self._bootstrap_results[reference].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                differences = comparison_values - reference_values
                ratios = comparison_values / reference_values
                log_ratios = pd.Series(ratios).map(
                    lambda value: math.log(value) if value > 0 else float("nan")
                )
                comparison_odds_values = comparison_values / (1.0 - comparison_values)
                reference_odds_values = reference_values / (1.0 - reference_values)
                odds_ratios = comparison_odds_values / reference_odds_values
            rd_std = self._bootstrap_std(differences)
            rr_std = self._bootstrap_std(ratios)
            log_rr_std = self._bootstrap_std(log_ratios)
            odds_ratio_std = self._bootstrap_std(odds_ratios)
        return CCWContrast(
            comparison=comparison,
            reference=reference,
            comparison_risk=comparison_risk,
            reference_risk=reference_risk,
            risk_difference=float(difference),
            risk_ratio=float(ratio),
            log_risk_ratio=log_ratio,
            odds_ratio=float(odds_ratio),
            risk_difference_std_error=rd_std,
            risk_ratio_std_error=rr_std,
            log_risk_ratio_std_error=log_rr_std,
            odds_ratio_std_error=odds_ratio_std,
        )

    def contrasts(self, reference: str | None = None) -> pd.DataFrame:
        """Compare every other strategy with one reference strategy.

        Parameters
        ----------
        reference : str, optional
            Reference strategy. Defaults to ``"control"`` when available,
            otherwise to the first configured strategy.

        Returns
        -------
        pandas.DataFrame
            One row per comparison with estimated risks and contrasts.
        """

        reference = (
            "control"
            if reference is None and "control" in self.strategy_names
            else self.strategy_names[0]
            if reference is None
            else reference
        )
        self._validate_strategy(reference)
        result = pd.DataFrame.from_records(
            [
                asdict(self.contrast(name, reference))
                for name in self.strategy_names
                if name != reference
            ]
        )
        if not self.has_bootstrap:
            result.drop(
                columns=[name for name in result if name.endswith("_std_error")],
                inplace=True,
            )
        return result

    @property
    def control_risk(self) -> float:
        """Return the estimated outcome risk under the control strategy.

        Returns
        -------
        float
            Estimated control-strategy risk at the configured estimation time.
        """

        return self.risk("control")

    @property
    def intervention_risk(self) -> float:
        """Return the estimated outcome risk under the intervention strategy.

        Returns
        -------
        float
            Estimated intervention-strategy risk at the configured estimation
            time.
        """

        return self.risk("intervention")

    @property
    def risk_difference(self) -> float:
        """Return the estimated intervention-versus-control risk difference.

        Returns
        -------
        float
            Estimated intervention-strategy risk minus the estimated
            control-strategy risk.
        """

        return self.contrast("intervention", "control").risk_difference

    @property
    def risk_ratio(self) -> float:
        """Return the estimated intervention-versus-control risk ratio.

        Returns
        -------
        float
            Estimated intervention-strategy risk divided by the estimated
            control-strategy risk.
        """

        return self.contrast("intervention", "control").risk_ratio

    @property
    def odds_ratio(self) -> float:
        """Return the estimated intervention-versus-control risk odds ratio.

        Returns
        -------
        float
            Estimated risk odds ratio at the configured estimation time.
        """

        return self.contrast("intervention", "control").odds_ratio

    @property
    def control_risk_std_error(self) -> float | None:
        """Return the estimated bootstrap standard error of the control risk.

        Returns
        -------
        float or None
            Estimated bootstrap standard error, or ``None`` without bootstrap
            results.
        """

        return self.risk_std_error("control")

    @property
    def intervention_risk_std_error(self) -> float | None:
        """Return the estimated bootstrap standard error of intervention risk.

        Returns
        -------
        float or None
            Estimated bootstrap standard error, or ``None`` without bootstrap
            results.
        """

        return self.risk_std_error("intervention")

    @property
    def risk_difference_std_error(self) -> float | None:
        """Return the estimated bootstrap risk-difference standard error.

        Returns
        -------
        float or None
            Estimated bootstrap standard error, or ``None`` without bootstrap
            results.
        """

        return self.contrast("intervention", "control").risk_difference_std_error

    @property
    def log_risk_ratio_std_error(self) -> float | None:
        """Return the estimated bootstrap log-risk-ratio standard error.

        Returns
        -------
        float or None
            Estimated bootstrap standard error, or ``None`` without bootstrap
            results.
        """

        return self.contrast("intervention", "control").log_risk_ratio_std_error

    @property
    def estimates(self) -> dict[str, Any]:
        """Return all numerical-engine estimates from the fitted analysis.

        Returns
        -------
        dict of str to Any
            Shallow copy of the numerical engine's estimated quantities. This
            low-level compatibility output can include analytical variance
            calculations; documented standard-error accessors use bootstrap
            refits instead.
        """

        return dict(self._estimates)

    @property
    def weights(self) -> pd.DataFrame:
        """Return estimated row-level inverse-probability-of-censoring weights.

        Returns
        -------
        pandas.DataFrame
            Deep copy of the estimated row-level weight data, with ID and time
            columns restored to the names defined by :class:`ccw.DataSpec`.
        """

        rename = {"id": self._data_spec.id, "t": self._data_spec.time}
        return self._weight_data.rename(columns=rename).copy(deep=True)

    def summary(self) -> pd.DataFrame:
        """Summarize the principal estimated causal quantities.

        Returns
        -------
        pandas.DataFrame
            One row per estimand, with ``estimand`` and ``estimate`` columns.
            A ``std_error`` column containing bootstrap standard errors is
            included only when the analysis has bootstrap results.
        """

        canonical = self.strategy_names == ("control", "intervention")
        rows = [
            {
                "estimand": f"{name}_risk" if canonical else f"risk:{name}",
                "estimate": self.risk(name),
            }
            for name in self.strategy_names
        ]
        reference = (
            "control" if "control" in self.strategy_names else self.strategy_names[0]
        )
        for comparison in self.strategy_names:
            if comparison == reference:
                continue
            contrast = self.contrast(comparison, reference)
            label = f"{comparison}_vs_{reference}"
            suffix = "" if canonical else f":{label}"
            rows.extend(
                [
                    {
                        "estimand": f"risk_difference{suffix}",
                        "estimate": contrast.risk_difference,
                    },
                    {
                        "estimand": f"risk_ratio{suffix}",
                        "estimate": contrast.risk_ratio,
                    },
                    {
                        "estimand": f"log_risk_ratio{suffix}",
                        "estimate": contrast.log_risk_ratio,
                    },
                    {
                        "estimand": f"odds_ratio{suffix}",
                        "estimate": contrast.odds_ratio,
                    },
                ]
            )
        result = pd.DataFrame.from_records(rows)
        if self.has_bootstrap:
            standard_errors = [
                self.risk_std_error(name) for name in self.strategy_names
            ]
            for comparison in self.strategy_names:
                if comparison == reference:
                    continue
                contrast = self.contrast(comparison, reference)
                standard_errors.extend(
                    [
                        contrast.risk_difference_std_error,
                        contrast.risk_ratio_std_error,
                        contrast.log_risk_ratio_std_error,
                        contrast.odds_ratio_std_error,
                    ]
                )
            result["std_error"] = standard_errors
        return result

    def weight_diagnostics(
        self,
        *,
        patterns: Iterable[str] = ("UNW", "NAT", "VAR", "HPREV2"),
        max_time: int | None = None,
        eps: float = 1e-12,
        min_m: int = 200,
        min_k_abs: int = 20,
        min_k_frac: float = 0.01,
        max_k_frac: float = 0.10,
        k_points: int = 20,
        tail_index_selector: str = "plateau",
        reiss_thomas_beta: float = 0.3,
        weight_cap_quantile: float | None = None,
        alpha_borderline: float = 1.2,
        alpha_bad: float = 1.0,
        tail_tiny: float = 1e-8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run tail diagnostics on the fitted censoring weights.

        Parameters
        ----------
        patterns : iterable of str, default=("UNW", "NAT", "VAR", "HPREV2")
            Tail-weighting patterns to evaluate. Available values are
            ``"UNW"``, ``"NAT"``, ``"VAR"``, ``"DIRECT"``, and
            ``"HPREV2"``.
        max_time : int, optional
            Largest analysis time to include. By default, use every available
            time.
        eps : float, default=1e-12
            Lower clipping bound applied to probability values.
        min_m : int, default=200
            Minimum number of valid observations required to estimate a tail
            index within a strategy and time.
        min_k_abs : int, default=20
            Absolute lower bound for candidate tail sizes.
        min_k_frac : float, default=0.01
            Sample-size fraction used as an additional lower bound for
            candidate tail sizes.
        max_k_frac : float, default=0.10
            Sample-size fraction used as the upper bound for candidate tail
            sizes.
        k_points : int, default=20
            Number of candidate tail sizes to evaluate.
        tail_index_selector : {"plateau", "reiss_thomas", "median"}, default="plateau"
            Rule used to select or aggregate the Hill tail-index path.
        reiss_thomas_beta : float, default=0.3
            Exponent used by the Reiss--Thomas selection rule.
        weight_cap_quantile : float, optional
            Quantile at which diagnostic weights are capped before estimation.
            By default, weights are not capped.
        alpha_borderline : float, default=1.2
            Tail-index threshold below which a result is flagged as
            borderline.
        alpha_bad : float, default=1.0
            Tail-index threshold below which a result is flagged as
            problematic.
        tail_tiny : float, default=1e-8
            Tolerance used to identify a negligible upper tail.

        Returns
        -------
        summary : pandas.DataFrame
            Tail-index summaries for each requested grouping and pattern.
        detail : pandas.DataFrame
            Hill-estimator values across candidate tail sizes.
        """

        from .diagnostics import _diagnose_weights

        return _diagnose_weights(
            self._weight_data,
            patterns=patterns,
            by_regime=True,
            by_run=False,
            max_time=max_time,
            eps=eps,
            min_m=min_m,
            min_k_abs=min_k_abs,
            min_k_frac=min_k_frac,
            max_k_frac=max_k_frac,
            k_points=k_points,
            tail_index_selector=tail_index_selector,
            reiss_thomas_beta=reiss_thomas_beta,
            weight_cap_quantile=weight_cap_quantile,
            alpha_borderline=alpha_borderline,
            alpha_bad=alpha_bad,
            tail_tiny=tail_tiny,
        )
