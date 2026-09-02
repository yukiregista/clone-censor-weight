"""Public treatment-strategy definitions for clone-censor-weight analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd


class TreatmentStrategy(ABC):
    """Base class for custom treatment strategies.

    Parameters
    ----------
    grace_period : int
        Non-negative last time governed by the treatment protocol.

    Notes
    -----
    Subclasses must implement :meth:`artificial_censor` and
    :meth:`censoring_prob_mask`. Only columns declared by
    :class:`ccw.DataSpec` are carried into the analysis. The subject and time
    columns are standardized internally, and their names are supplied to both
    methods through ``id_col`` and ``time_col``. All other declared columns
    retain their user-provided names and can be accessed directly. A custom
    strategy should accept the names of any covariates it needs as its own
    constructor arguments. A strategy should determine adherence using
    treatment and covariate information available at or before the current
    time, never future observations or outcomes. Within a time point, the
    assumed order is outcome, covariate measurement, treatment initiation, and
    any resulting protocol-deviation censoring.
    """

    __slots__ = ("_grace_period",)

    def __init__(self, grace_period: int) -> None:
        if (
            isinstance(grace_period, bool)
            or not isinstance(grace_period, int)
            or grace_period < 0
        ):
            raise ValueError("grace_period must be a non-negative integer.")
        self._grace_period = grace_period

    @property
    def grace_period(self) -> int:
        """Return the last time governed by the treatment protocol.

        Returns
        -------
        int
            Non-negative protocol deadline.
        """

        return self._grace_period

    @abstractmethod
    def artificial_censor(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        treatment_col: str,
    ) -> pd.DataFrame:
        """Mark the first row at which each subject deviates from the strategy.

        This method is called on one strategy clone before CCW adds the
        columns used to estimate censoring weights.

        Parameters
        ----------
        data : pandas.DataFrame
            Validated discrete-time long data with one row per subject and
            time point. It contains the subject, time, and treatment
            columns named by the arguments below, together with the outcome,
            censoring, baseline, time-varying, and sample-weight columns
            declared in :class:`ccw.DataSpec`.
        id_col : str
            Name of the subject-identifier column in ``data``.
        time_col : str
            Name of the discrete-time column in ``data``.
        treatment_col : str
            Name of the non-null binary treatment-initiation column in
            ``data``.

        Returns
        -------
        pandas.DataFrame
            A copy that preserves the input columns and retained row indices
            and adds both of the following columns:

            * ``"artificial_censor"``: a non-null binary indicator equal to
              one only on the first protocol-deviation row for a subject;
            * ``"time_to_artificial_censor"``: that subject's first deviation
              time on every retained row, or missing if no deviation occurs.

            The deviation row must be retained. Rows after it may be removed;
            the CCW pipeline removes them in either case.
        """

        raise NotImplementedError

    @abstractmethod
    def censoring_prob_mask(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        treatment_col: str,
    ) -> pd.Series:
        """Select rows where protocol-deviation censoring can occur.

        The returned mask defines the risk set for the artificial-censoring
        model. On a selected row, CCW models the probability of
        ``"artificial_censor_tstart"`` and includes the corresponding
        conditional probability of remaining artificially uncensored, and
        therefore adherent, in the cumulative weight. For an unselected row,
        this probability is set to one, so that interval does not change the
        inverse-probability weight.

        Consider a strategy requiring treatment initiation by day ``d``. Only
        subjects at day ``d`` who have not initiated treatment earlier are at
        risk of artificial censoring for failure to initiate. The mask should
        therefore select exactly those subjects' day-``d`` rows. Earlier rows
        are excluded because this form of censoring cannot yet occur.

        By contrast, for a strategy prohibiting initiation through day ``d``,
        initiation can cause deviation at every time through ``d``. Its mask
        therefore selects every row at or before ``d``.

        Parameters
        ----------
        data : pandas.DataFrame
            The clone returned by :meth:`artificial_censor`, after CCW has
            prepared it for weighting. The subject, time, and treatment
            columns named by the arguments below and all declared
            :class:`ccw.DataSpec` covariates remain available. Custom
            strategies do not need to access CCW's additional generated
            columns.
        id_col : str
            Name of the subject-identifier column in ``data``.
        time_col : str
            Name of the time column to use for protocol timing.
        treatment_col : str
            Name of the non-null binary treatment-initiation column in
            ``data``.

        Returns
        -------
        pandas.Series
            Non-null boolean Series with exactly ``data.index``. ``True``
            selects a row where the subject is at risk of deviating and the
            artificial-censoring model should contribute to the weight;
            ``False`` excludes that row from this model.

        """

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class InitiateBy(TreatmentStrategy):
    """Require treatment initiation on or before a deadline.

    Parameters
    ----------
    day : int
        Non-negative last day on which treatment may be initiated without
        protocol deviation.
    """

    day: int

    def __post_init__(self) -> None:
        if isinstance(self.day, bool) or not isinstance(self.day, int) or self.day < 0:
            raise ValueError("day must be a non-negative integer.")

    @property
    def grace_period(self) -> int:
        """Return the treatment-initiation deadline.

        Returns
        -------
        int
            Configured value of ``day``.
        """

        return self.day

    def transform_cpd(
        self,
        original_cpd: Callable[..., Any],
        treatment_variable: object,
    ) -> Callable[..., Any]:
        """Return a CPD that enforces initiation by the deadline.

        This method supports the package's research simulator. Before the
        deadline it retains the observed treatment process. At the deadline it
        initiates subjects who have not yet started treatment, and thereafter
        it prevents another initiation event.

        Parameters
        ----------
        original_cpd : callable
            Conditional probability distribution for treatment in the
            observational data-generating process.
        treatment_variable : object
            Key used to find treatment history in the CPD parent values.

        Returns
        -------
        callable
            Conditional probability distribution under this strategy.
        """

        def transformed(parent_values, time: int, seed: int):
            if time < self.day:
                return original_cpd(parent_values, time, seed)
            history = parent_values.get(treatment_variable, {})
            if time == self.day and sum(history.values()) == 0:
                return 1
            return 0

        return transformed

    def censoring_prob_mask(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        treatment_col: str,
    ) -> pd.Series:
        """Select subjects at risk of missing the initiation deadline.

        Parameters
        ----------
        data : pandas.DataFrame
            Canonical longitudinal clone data.
        id_col : str
            Subject-identifier column.
        time_col : str
            Analysis-time column.
        treatment_col : str
            Incident treatment-initiation indicator column.

        Returns
        -------
        pandas.Series
            Boolean mask selecting deadline rows for subjects who did not
            initiate treatment before the deadline.
        """

        first_treatment = (
            data.loc[data[treatment_col] == 1].groupby(id_col)[time_col].min()
        )
        treated_before = set(first_treatment[first_treatment < self.day].index)
        return (~data[id_col].isin(treated_before)) & (data[time_col] == self.day)

    def artificial_censor(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        treatment_col: str,
    ) -> pd.DataFrame:
        """Censor subjects who do not initiate treatment by the deadline.

        Parameters
        ----------
        data : pandas.DataFrame
            Canonical longitudinal clone data.
        id_col : str
            Subject-identifier column.
        time_col : str
            Discrete-time column.
        treatment_col : str
            Incident treatment-initiation indicator column.

        Returns
        -------
        pandas.DataFrame
            A censored copy containing protocol-deviation indicators and
            censoring times.
        """

        frame = data.copy().sort_values([id_col, time_col], kind="mergesort")
        frame["artificial_censor"] = 0
        initiated = frame.groupby(id_col, sort=False)[treatment_col].cumsum()
        at_deadline = frame[time_col] == self.day
        frame.loc[at_deadline, "artificial_censor"] = (
            initiated.loc[at_deadline] == 0
        ).astype(int)
        censored = (
            frame.groupby(id_col, sort=False)["artificial_censor"].cummax().astype(bool)
        )
        frame = frame.loc[~(censored & (frame[time_col] > self.day))].copy()
        censor_time = (
            frame.loc[frame["artificial_censor"] == 1]
            .groupby(id_col)[time_col]
            .min()
        )
        frame["time_to_artificial_censor"] = frame[id_col].map(censor_time)
        return frame

    def treatment_prob_mask(
        self,
        data: pd.DataFrame,
        time_col: str,
        treatment_col: str,
    ) -> pd.Series:
        """Select rows used to estimate treatment-initiation probabilities.

        Parameters
        ----------
        data : pandas.DataFrame
            Longitudinal data containing ``"id"`` and treatment columns.
        time_col : str
            Analysis-time column.
        treatment_col : str
            Incident treatment-initiation indicator column.

        Returns
        -------
        pandas.Series
            Boolean mask selecting untreated risk-set rows through ``day``.
        """

        treated_before = (
            data.groupby("id", sort=False)[treatment_col].cumsum()
            - data[treatment_col]
        )
        return (treated_before == 0) & (data[time_col] <= self.day)

    def convert_treatment_prob_to_ipcw(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        *,
        return_details: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """Convert initiation probabilities to protocol-adherence weights.

        Parameters
        ----------
        data : pandas.DataFrame
            Longitudinal data containing ``"id"``, ``"time"``,
            ``treatment_col``, and ``"treatment_prob"`` columns.
        treatment_col : str
            Incident treatment-initiation indicator column.
        return_details : bool, default=False
            Whether to return derivative terms used by variance estimation.

        Returns
        -------
        pandas.DataFrame or tuple of (pandas.DataFrame, dict)
            Data with ``"ipw.weights"`` or, when ``return_details=True``,
            that data together with derivative details.
        """

        frame = data.copy().sort_values(["id", "time"], kind="mergesort")
        treated_before = set(
            frame.loc[
                (frame["time"] < self.day) & (frame[treatment_col] == 1), "id"
            ]
        )
        at_risk = (~frame["id"].isin(treated_before)) & (
            frame["time"] == self.day
        )
        uncensor_probability = pd.Series(1.0, index=frame.index)
        uncensor_probability.loc[at_risk] = frame.loc[at_risk, "treatment_prob"]
        cumulative = uncensor_probability.groupby(frame["id"], sort=False).cumprod()
        frame["ipw.weights"] = 1.0 / cumulative
        if not return_details:
            return frame

        derivative = pd.Series(0.0, index=frame.index)
        derivative.loc[at_risk] = 1.0 - frame.loc[at_risk, "treatment_prob"]
        return frame, {"at_risk": at_risk, "dlogg_deta": derivative, "x": self.day}


@dataclass(frozen=True, slots=True)
class NoInitiationThrough(TreatmentStrategy):
    """Prohibit treatment initiation through a specified day.

    Parameters
    ----------
    day : int
        Non-negative last day through which treatment initiation constitutes
        protocol deviation.
    """

    day: int

    def __post_init__(self) -> None:
        if isinstance(self.day, bool) or not isinstance(self.day, int) or self.day < 0:
            raise ValueError("day must be a non-negative integer.")

    @property
    def grace_period(self) -> int:
        """Return the last day on which initiation is prohibited.

        Returns
        -------
        int
            Configured value of ``day``.
        """

        return self.day

    def transform_cpd(
        self,
        original_cpd: Callable[..., Any],
        treatment_variable: object,
    ) -> Callable[..., Any]:
        """Return a CPD that prevents initiation through the deadline.

        Parameters
        ----------
        original_cpd : callable
            Conditional probability distribution for treatment in the
            observational data-generating process.
        treatment_variable : object
            Key identifying treatment history. It is accepted for the common
            simulator strategy interface and is not needed by this strategy.

        Returns
        -------
        callable
            Conditional probability distribution under this strategy.
        """

        def transformed(parent_values, time: int, seed: int):
            if time <= self.day:
                return 0
            return original_cpd(parent_values, time, seed)

        return transformed

    def censoring_prob_mask(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        treatment_col: str,
    ) -> pd.Series:
        """Select rows within the no-initiation period.

        Parameters
        ----------
        data : pandas.DataFrame
            Canonical longitudinal clone data.
        id_col : str
            Subject-identifier column. This argument is accepted for the
            common strategy interface and is not used here.
        time_col : str
            Analysis-time column.
        treatment_col : str
            Incident treatment-initiation indicator column. This argument is
            accepted for compatibility with
            :class:`ccw.strategies.TreatmentStrategy`.

        Returns
        -------
        pandas.Series
            Boolean mask selecting observations through ``day``.
        """

        return data[time_col] <= self.day

    def artificial_censor(
        self,
        data: pd.DataFrame,
        *,
        id_col: str,
        time_col: str,
        treatment_col: str,
    ) -> pd.DataFrame:
        """Censor subjects who initiate during the prohibited period.

        Parameters
        ----------
        data : pandas.DataFrame
            Canonical longitudinal clone data.
        id_col : str
            Subject-identifier column.
        time_col : str
            Discrete-time column.
        treatment_col : str
            Incident treatment-initiation indicator column.

        Returns
        -------
        pandas.DataFrame
            A censored copy containing protocol-deviation indicators and
            censoring times.
        """

        frame = data.copy().sort_values([id_col, time_col], kind="mergesort")
        frame["artificial_censor"] = (
            (frame[treatment_col] == 1) & (frame[time_col] <= self.day)
        ).astype(int)
        censor_time = (
            frame.loc[frame["artificial_censor"] == 1]
            .groupby(id_col)[time_col]
            .min()
        )
        frame["time_to_artificial_censor"] = frame[id_col].map(censor_time)
        keep = frame["time_to_artificial_censor"].isna() | (
            frame[time_col] <= frame["time_to_artificial_censor"]
        )
        return frame.loc[keep].copy()

    def treatment_prob_mask(
        self,
        data: pd.DataFrame,
        time_col: str,
        treatment_col: str,
    ) -> pd.Series:
        """Select rows used to estimate treatment-initiation probabilities.

        Parameters
        ----------
        data : pandas.DataFrame
            Longitudinal data containing an analysis-time column.
        time_col : str
            Analysis-time column.
        treatment_col : str
            Incident treatment-initiation indicator column. It is accepted
            for the common strategy interface and is not used here.

        Returns
        -------
        pandas.Series
            Boolean mask selecting rows through ``day``.
        """

        return data[time_col] <= self.day

    def convert_treatment_prob_to_ipcw(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        *,
        return_details: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        """Convert non-initiation probabilities to protocol-adherence weights.

        Parameters
        ----------
        data : pandas.DataFrame
            Longitudinal data containing ``"id"``, ``"time"``, and
            ``"treatment_prob"`` columns.
        treatment_col : str
            Incident treatment-initiation indicator column. It is accepted
            for the common strategy interface and is not used here.
        return_details : bool, default=False
            Whether to return derivative terms used by variance estimation.

        Returns
        -------
        pandas.DataFrame or tuple of (pandas.DataFrame, dict)
            Data with ``"ipw.weights"`` or, when ``return_details=True``,
            that data together with derivative details.
        """

        frame = data.copy().sort_values(["id", "time"], kind="mergesort")
        at_risk = frame["time"] <= self.day
        uncensor_probability = pd.Series(1.0, index=frame.index)
        uncensor_probability.loc[at_risk] = (
            1.0 - frame.loc[at_risk, "treatment_prob"]
        )
        cumulative = uncensor_probability.groupby(frame["id"], sort=False).cumprod()
        frame["ipw.weights"] = 1.0 / cumulative
        if not return_details:
            return frame

        derivative = pd.Series(0.0, index=frame.index)
        derivative.loc[at_risk] = -frame.loc[at_risk, "treatment_prob"]
        return frame, {"at_risk": at_risk, "dlogg_deta": derivative, "x": self.day}


__all__ = ["InitiateBy", "NoInitiationThrough", "TreatmentStrategy"]
