"""Private input-schema implementation for the public CCW API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_GENERATED_COLUMNS = {
    "arm",
    "tstart",
    "tend",
    "outcome_tend",
    "artificial_censor",
    "artificial_censor_tstart",
    "time_to_artificial_censor",
    "time_to_outcome_cutoff_obs",
    "CENSOR_tstart",
    "ipw.weights",
    "analysis_weight",
    "G_t",
    "H_prev",
    "S_t",
    "t",
}


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Describe columns in a discrete-time longitudinal data set.

    Parameters
    ----------
    id : str
        Subject-identifier column.
    time : str
        Integer time column. Each subject must have consecutive observations
        beginning at zero.
    treatment : str
        Treatment-initiation indicator column.
    outcome : str
        Outcome indicator column.
    censoring : tuple of str, default=()
        Observed-censoring indicator columns.
    baseline : tuple of str, default=()
        Baseline covariate columns. Values must be constant within each
        subject.
    time_varying : tuple of str, default=()
        Time-varying covariate columns to make available to weight models.
        Values must be observed before the outcome. Values on or after the
        outcome row may be missing because those rows do not contribute to
        the at-risk intervals used by the analysis.
    sample_weight : str, optional
        Positive subject-level sampling-weight column. Values must be constant
        within each subject.

    Notes
    -----
    Treatment, outcome, and censoring columns must be binary indicators. A
    value of one means that the corresponding event occurs at that time, and
    each event can occur at most once per subject. During fitting, every
    time-varying covariate also receives a ``<name>_baseline`` column containing
    that subject's value at time zero. Within each time point, the assumed
    observation order is outcome, covariate measurement, treatment initiation,
    and any resulting protocol-deviation censoring.
    """

    id: str
    time: str
    treatment: str
    outcome: str
    censoring: tuple[str, ...] = ()
    baseline: tuple[str, ...] = ()
    time_varying: tuple[str, ...] = ()
    sample_weight: str | None = None

    def __post_init__(self) -> None:
        if any(
            isinstance(columns, str)
            for columns in (self.censoring, self.baseline, self.time_varying)
        ):
            raise TypeError(
                "censoring, baseline, and time_varying must be sequences of column names."
            )
        object.__setattr__(self, "censoring", tuple(self.censoring))
        object.__setattr__(self, "baseline", tuple(self.baseline))
        object.__setattr__(self, "time_varying", tuple(self.time_varying))
        primary_roles = [self.id, self.time, self.treatment, self.outcome, *self.censoring]
        covariates = [*self.baseline, *self.time_varying]
        all_roles = [*primary_roles, *covariates]
        if self.sample_weight is not None:
            all_roles.append(self.sample_weight)
        if any(not isinstance(name, str) or not name.strip() for name in all_roles):
            raise ValueError("Column names must be non-empty strings.")
        if len(set(primary_roles)) != len(primary_roles):
            raise ValueError("ID, time, treatment, outcome, and censoring roles must be distinct.")
        if len(set(self.baseline)) != len(self.baseline):
            raise ValueError("baseline column names must be unique.")
        if len(set(self.time_varying)) != len(self.time_varying):
            raise ValueError("time_varying column names must be unique.")
        primary_overlap = sorted(set(primary_roles) & set(covariates))
        if primary_overlap:
            raise ValueError(f"Covariate columns overlap other data roles: {primary_overlap}")
        covariate_overlap = sorted(set(self.baseline) & set(self.time_varying))
        if covariate_overlap:
            raise ValueError(
                "Columns cannot be both baseline and time_varying: "
                f"{covariate_overlap}"
            )
        if self.sample_weight is not None and self.sample_weight in {
            *primary_roles,
            *covariates,
        }:
            raise ValueError("sample_weight must be distinct from all other data roles.")

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the input columns required by this specification.

        Returns
        -------
        tuple of str
            Required columns in role-definition order, with duplicates
            removed.
        """

        columns = [
            self.id,
            self.time,
            self.treatment,
            self.outcome,
            *self.censoring,
            *self.baseline,
            *self.time_varying,
        ]
        if self.sample_weight is not None:
            columns.append(self.sample_weight)
        return tuple(dict.fromkeys(columns))

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data and create a canonical working copy.

        Parameters
        ----------
        data : pandas.DataFrame
            Discrete-time longitudinal observations using the configured
            column names.

        Returns
        -------
        pandas.DataFrame
            Deep copy sorted by subject and time, with the configured ID and
            time columns renamed internally to ``"id"`` and ``"time"``.

        Raises
        ------
        TypeError
            If ``data`` is not a pandas data frame.
        ValueError
            If required columns, identifiers, time values, event indicators,
            or sampling weights violate the schema.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one observation.")
        if not data.columns.is_unique:
            duplicates = sorted(
                set(data.columns[data.columns.duplicated()].tolist()), key=str
            )
            raise ValueError(f"DataFrame column names must be unique; duplicates: {duplicates}")
        missing = sorted(set(self.required_columns) - set(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        dynamic_columns = {
            *(f"time_to_{column}" for column in self.censoring),
            *(f"{column}_tstart" for column in self.censoring),
            *(f"{column}_baseline" for column in self.time_varying),
        }
        collisions = sorted(
            column for column in data.columns if column in _GENERATED_COLUMNS | dynamic_columns
        )
        if collisions:
            raise ValueError(f"Input columns collide with CCW-generated columns: {collisions}")
        if self.id != "id" and "id" in data.columns:
            raise ValueError("Input already contains 'id'; it conflicts with the canonical ID column.")
        if self.time != "time" and "time" in data.columns:
            raise ValueError("Input already contains 'time'; it conflicts with the canonical time column.")

        frame = data.loc[:, self.required_columns].copy(deep=True).rename(
            columns={self.id: "id", self.time: "time"}
        )
        if frame["id"].isna().any() or frame["time"].isna().any():
            raise ValueError("ID and time columns cannot contain missing values.")
        if frame.duplicated(["id", "time"]).any():
            raise ValueError("Data must contain at most one row per subject and time.")

        numeric_time = pd.to_numeric(frame["time"], errors="coerce")
        if (
            numeric_time.isna().any()
            or not np.isfinite(numeric_time).all()
            or not np.all(np.equal(numeric_time, np.floor(numeric_time)))
        ):
            raise ValueError("Time values must be finite integers.")
        frame["time"] = numeric_time.astype(int)
        if (frame["time"] < 0).any():
            raise ValueError("Time values must be non-negative.")
        frame.sort_values(["id", "time"], kind="mergesort", inplace=True)
        frame.reset_index(drop=True, inplace=True)

        for subject_id, times in frame.groupby("id", sort=False)["time"]:
            observed = times.to_numpy(dtype=int)
            expected = np.arange(observed[-1] + 1, dtype=int)
            if not np.array_equal(observed, expected):
                raise ValueError(
                    f"Subject {subject_id!r} must have consecutive times beginning at zero."
                )

        indicator_columns = [self.treatment, self.outcome, *self.censoring]
        for column in indicator_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise ValueError(f"{column!r} must be a non-null binary indicator.")
            frame[column] = values.astype(int)

        for column in (self.treatment, self.outcome, *self.censoring):
            event_counts = frame.groupby("id", sort=False)[column].sum()
            if (event_counts > 1).any():
                raise ValueError(f"{column!r} must occur at most once per subject.")
        baseline_outcomes = frame.loc[frame["time"] == 0, self.outcome]
        if (baseline_outcomes != 0).any():
            raise ValueError("The outcome indicator must be zero at baseline.")

        for column in self.baseline:
            values = frame[column]
            if values.isna().any():
                raise ValueError(f"Covariate column {column!r} cannot contain missing values.")
            if pd.api.types.is_numeric_dtype(values) and not np.isfinite(
                values.to_numpy(dtype=float)
            ).all():
                raise ValueError(f"Covariate column {column!r} must contain finite values.")

            distinct_values = frame.groupby("id", sort=False)[column].nunique(dropna=False)
            if (distinct_values > 1).any():
                raise ValueError(
                    f"Baseline column {column!r} must be constant within each subject."
                )

        at_or_after_outcome = (
            frame.groupby("id", sort=False)[self.outcome].cummax().astype(bool)
        )
        for column in self.time_varying:
            values_at_risk = frame.loc[~at_or_after_outcome, column]
            if values_at_risk.isna().any():
                raise ValueError(
                    f"Time-varying covariate column {column!r} cannot contain "
                    "missing values before the outcome."
                )
            if pd.api.types.is_numeric_dtype(values_at_risk) and not np.isfinite(
                values_at_risk.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Time-varying covariate column {column!r} must contain finite "
                    "values before the outcome."
                )

        if self.sample_weight is not None:
            weights = pd.to_numeric(frame[self.sample_weight], errors="coerce")
            if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Sample weights must be finite and strictly positive.")
            frame[self.sample_weight] = weights.astype(float)
            varying = frame.groupby("id", sort=False)[self.sample_weight].nunique(dropna=False)
            if (varying > 1).any():
                raise ValueError("Sample weights must be constant within subject.")

        return frame
