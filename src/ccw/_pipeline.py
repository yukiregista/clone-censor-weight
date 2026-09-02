"""Private analysis pipelines shared by the public and research workflows."""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import pandas as pd

from ccw._core.analysis import (
    calculate_strategy_outcomes,
    ccw_preprocessing,
    create_counting_process_format,
    integrate_censoring_columns,
    prepare_ccw_analysis_part,
    remove_rows_after_fup,
)
from ccw._logging import TagAdapter, temp_stderr_logging, to_level

## base logger
_base_logger = getLogger(__name__)


@dataclass(slots=True)
class _PipelineOutput:
    """Internal output shared by the estimator and research workflow."""

    estimates: dict[str, object]
    weights: pd.DataFrame | None = None
    cloned_data: dict[str, pd.DataFrame] | None = None


def _pairwise_variance(
    reference: str,
    comparison: str,
    *,
    hazard_if_by_arm: dict,
    hazard_cov_by_arm: dict,
    survival_by_arm: dict,
    hazards_by_arm: dict,
    var_survival_by_arm: dict,
    n_ids_by_arm: dict,
) -> tuple[float, float, float]:
    """Return risk-difference variance, log-risk-ratio variance, and covariance."""

    required_maps = (
        hazard_if_by_arm,
        hazard_cov_by_arm,
        survival_by_arm,
        hazards_by_arm,
        var_survival_by_arm,
        n_ids_by_arm,
    )
    if not all(reference in values and comparison in values for values in required_maps):
        return np.nan, np.nan, np.nan

    n_reference = n_ids_by_arm[reference]
    n_comparison = n_ids_by_arm[comparison]
    if n_reference != n_comparison:
        raise ValueError(
            "Strategies have different numbers of subjects: "
            f"{reference}={n_reference}, {comparison}={n_comparison}."
        )

    cross_covariance = (
        hazard_if_by_arm[reference] @ hazard_if_by_arm[comparison].T
    ) / (n_reference**2)
    reference_survival = survival_by_arm[reference]
    comparison_survival = survival_by_arm[comparison]
    reference_risk = 1.0 - reference_survival
    comparison_risk = 1.0 - comparison_survival
    reference_gradient = reference_survival / (1.0 - hazards_by_arm[reference])
    comparison_gradient = comparison_survival / (1.0 - hazards_by_arm[comparison])
    risk_covariance = reference_gradient @ cross_covariance @ comparison_gradient
    difference_variance = (
        var_survival_by_arm[reference]
        + var_survival_by_arm[comparison]
        - 2.0 * risk_covariance
    )
    if reference_risk <= 0 or comparison_risk <= 0:
        return difference_variance, np.nan, risk_covariance

    reference_log_gradient = reference_gradient / reference_risk
    comparison_log_gradient = comparison_gradient / comparison_risk
    log_ratio_variance = (
        reference_log_gradient
        @ hazard_cov_by_arm[reference]
        @ reference_log_gradient
        + comparison_log_gradient
        @ hazard_cov_by_arm[comparison]
        @ comparison_log_gradient
        - 2.0
        * (reference_log_gradient @ cross_covariance @ comparison_log_gradient)
    )
    return difference_variance, log_ratio_variance, risk_covariance

def run_ccw_weighting_and_analysis(
    cloned_dfs: dict[str, pd.DataFrame],
    strategies,
    ipw_explanatory_formula,
    configs,
    cutoff_time_of_intervention: float | Mapping[str, float],
    cutoff_time_of_observation_display: float = 10,
    logger_=None,
    verbose=None,
    weight_col: str | None = None,
    return_weight_df: bool = False,
):
    """
    既に clone & censor 済みの cloned_dfs に対して、
    IPCWの推定およびアウトカム解析のみを行うユーティリティ関数。

    Parameters
    ----------
    cloned_dfs : dict[str, pd.DataFrame]
        key='control','intervention', ... などのアーム名を持つ DataFrame の dict。
        各 DataFrame は ccw_preprocessing → artificial_censor →
        create_counting_process_format → remove_rows_after_fup →
        integrate_censoring_columns 済みであることを前提とする。
    その他の引数は cloning_censoring_weighting_analysis と同様。
    """
    lg = TagAdapter(logger_ or _base_logger, tag="run_ccw_weighting_and_analysis")
    with temp_stderr_logging(lg.logger, level = to_level(verbose), user_supplied=(logger_ is not None)):
        treatment_var = configs['treatment_var']
        censor_vars = configs.get('censor_vars', None)
        censor_day0 = configs.get('censor_day0', None)

        treatment_col = treatment_var.name
        censor_cols = []
        if censor_vars is not None:
            censor_cols = [var.name for var in censor_vars]

        lg.info("3. Weighting and Analysis...")

        # --- 3-1. IPCW ---
        for key, cloned_df in cloned_dfs.items():
            lg.debug(f"分割後 {key}: {len(cloned_df)} 行")

        if isinstance(cutoff_time_of_intervention, Mapping):
            grace_periods = dict(cutoff_time_of_intervention)
            if set(grace_periods) != set(cloned_dfs):
                raise ValueError(
                    "cutoff_time_of_intervention must define every strategy."
                )
        else:
            grace_periods = {
                key: cutoff_time_of_intervention for key in cloned_dfs
            }

        # (NEW) stash details if returned
        ipcw_details_by_arm = {}

        for key, cloned_df in cloned_dfs.items():
            formula_namespace = {"grace_end": grace_periods[key]}
            out = configs['ipcw_func'](
                cloned_df,
                treatment_col,
                ipw_explanatory_formula[key],
                strategies[key],
                censor_cols=censor_cols,
                censor_day0=censor_day0,
                return_details=True,  # only functions that support it will use it
                weight_col=weight_col,
                formula_namespace=formula_namespace,
            )

            if isinstance(out, tuple):
                res_df, ipcw_details = out
            else:
                res_df, ipcw_details = out, None

            cloned_df["ipw.weights"] = res_df["ipw.weights"]
            cloned_dfs[key] = cloned_df
            ipcw_details_by_arm[key] = ipcw_details

        analysis_weight_col = "ipw.weights"
        if weight_col is not None:
            for key, cloned_df in cloned_dfs.items():
                cloned_df['analysis_weight'] = cloned_df[weight_col] * cloned_df['ipw.weights']
                cloned_dfs[key] = cloned_df
            analysis_weight_col = "analysis_weight"

        # --- 3-2. Analysis: outcome metrics & モデル推定 ---
        weight_summary_frames: list[pd.DataFrame] = []
        weight_export_frames: list[pd.DataFrame] | None = [] if return_weight_df else None
        ipcw_details_export_by_arm = {}
        if return_weight_df:
            for key, details in ipcw_details_by_arm.items():
                if details is None:
                    continue
                g_t = details.get("g_t")
                h_prev = details.get("h_prev")
                if g_t is None or h_prev is None:
                    continue
                ipcw_details_export_by_arm[key] = {
                    "g_t": g_t,
                    "h_prev": h_prev,
                }
        for key, cloned_df in cloned_dfs.items():
            # keep the pre-filter rows for diagnostics export (S_t risk-set level)
            export_source_df = cloned_df
            cloned_df = prepare_ccw_analysis_part(cloned_df)
            cloned_dfs[key] = cloned_df
            weight_summary_frames.append(
                cloned_df[['tstart', 'tend', analysis_weight_col]].copy().assign(arm=key)
            )
            if return_weight_df:
                export_cols = ['id', 'tstart', 'tend', 'ipw.weights']
                if analysis_weight_col not in export_cols:
                    export_cols.append(analysis_weight_col)
                if weight_col is not None and weight_col not in export_cols:
                    export_cols.append(weight_col)
                # optional diagnostics-friendly columns
                for diag_col in ['CENSOR_tstart', 'outcome_tend']:
                    if diag_col in export_source_df.columns and diag_col not in export_cols:
                        export_cols.append(diag_col)
                export_df = export_source_df[export_cols].copy().assign(arm=key)
                export_df["S_t"] = 1
                export_df["t"] = export_df["tstart"]
                details = ipcw_details_export_by_arm.get(key)
                if details is not None:
                    export_df["G_t"] = details["g_t"].loc[export_df.index].to_numpy()
                    export_df["H_prev"] = details["h_prev"].loc[export_df.index].to_numpy()
                weight_export_frames.append(export_df)

        canonical_pair = set(cloned_dfs) == {"control", "intervention"}
        results_dict, km_details = calculate_strategy_outcomes(
            cloned_dfs,
            time_col="tend",
            weight_col=analysis_weight_col,
            outcome_col="outcome_tend",
            cutoff_time_of_observation_display=int(
                cutoff_time_of_observation_display
            ),
        )
        
        
        

        # Now construct the sandwich variance estimator from IPCW + KM details
        ## stash details for variance calculation
        hazard_if_by_arm = {}       # key -> (T, n_ids) influence for hazards only
        hazard_cov_by_arm = {}      # key -> (T, T) from var_matrix (reuse!)
        survival_by_arm = {}        # key -> scalar S
        hazards_by_arm = {}         # key -> (T,) hazards
        n_ids_by_arm = {}           # key -> int
        var_survival_by_arm = {}   # key -> scalar variance of survival
            
        # 1. construct joint M-estimating equations
        
        for key, cloned_df in cloned_dfs.items():
            if ipcw_details_by_arm.get(key) is not None and km_details is not None:
                cloned_df = cloned_df.copy()
                lg.info(f"Calculating sandwich variance estimator for arm '{key}'")

                km_times = np.asarray(km_details[f'km_{key}']['times'], dtype=float)
                hazards_all = np.asarray(km_details[f'km_{key}']['hazard'], dtype=float)

                # 1) 時間のチェック（1..cutoff_time_of_observation_displayの整数が全て観測されているか）
                display_time = int(cutoff_time_of_observation_display)
                expected_times = set(range(1, display_time + 1)) if display_time >= 1 else set()
                km_times_int = km_times.astype(int)
                observed_times = {
                    int(t) for t in km_times_int
                    if (1 <= t <= display_time)
                }
                missing_expected_times = sorted(expected_times ^ observed_times)
                results_dict.update({
                    f'variance_expected_time_support_ok_{key}': len(missing_expected_times) == 0,
                    f'variance_expected_time_support_missing_count_{key}': len(missing_expected_times),
                })
                if len(missing_expected_times) > 0:
                    lg.warning(
                        f"Arm '{key}': missing expected times within 1..{display_time}: {missing_expected_times}"
                    )
                    var_survival = np.nan
                    var_survival_by_arm[key] = var_survival
                    results_dict.update({
                        f'outcome_rate_{key}_var': var_survival,
                        f'outcome_rate_{key}_std': np.nan,
                    })
                    continue

                # estimating equations (stack horizontally here)         
                
                M = pd.concat([km_details[f'km_{key}']['eeq_by_id'], ipcw_details_by_arm[key]['score_by_id']], axis=1)
                M.fillna(0, inplace=True)  # fill NaN with 0
                M = M.T.values # shape (num_estimating_eqs, num_subjects)
                n_ids = M.shape[1]
                
                
                # construct \partial m / \partial \theta
                A = np.zeros((M.shape[0], M.shape[0]))  # shape (num_estimating_eqs, num_estimating_eqs)
                # fill the first diagnoal block using km_details
                dm_dhazard = km_details[f'km_{key}']['dm_dhazard']  # shape (T, T)
                A[:dm_dhazard.shape[0], :dm_dhazard.shape[1]] = dm_dhazard
                # fill the second diagonal block using ipcw_details
                hessian_gamma = ipcw_details_by_arm[key]['hessian_gamma']  # shape (P, P)
                A[dm_dhazard.shape[0]:, dm_dhazard.shape[1]:] = hessian_gamma
                
                # off-diagnoal block
                dW_dgamma = ipcw_details_by_arm[key]['dW_dgamma']  # shape (n_observations, P)
                # take remaining indices
                dW_dgamma = dW_dgamma.loc[cloned_df.index, :]  # align with cloned_df
                time_to_hazard = dict(
                    zip(
                        km_details[f"km_{key}"]["times"],
                        km_details[f"km_{key}"]["hazard"],
                        strict=True,
                    )
                )
                cloned_df.loc[:, 'marginal_hazard'] = cloned_df['tend'].map(time_to_hazard)  
                dm1dgamma = dW_dgamma * \
                (cloned_df.loc[:, ['marginal_hazard']].values - cloned_df.loc[:, ['outcome_tend']].values) # shape (n_observations, P)
                # now group by time to sum
                dm1dgamma['tend'] = cloned_df['tend']
                dm1_dgamma = dm1dgamma.groupby('tend').sum()  # shape (T, P)
                A[:dm_dhazard.shape[0], dm_dhazard.shape[1]:] = dm1_dgamma.values # shape (T, P)
                A = A / n_ids  # average over subjects

                if not np.isfinite(A).all():
                    lg.warning(
                        f"Arm '{key}': detected NaN/Inf in A before pinv; survival variance set to NaN."
                    )
                    var_survival = np.nan
                    var_survival_by_arm[key] = var_survival
                    results_dict.update({
                        f'outcome_rate_{key}_var': var_survival,
                        f'outcome_rate_{key}_std': np.nan,
                    })
                    continue
                                
                # calculate meat
                meat = M @ M.T / n_ids # shape (num_estimating_eqs, num_estimating_eqs), already averaged over subjects
                bread = np.linalg.pinv(A)  # shape (num_estimating_eqs, num_estimating_eqs)
                Phi = bread @ M                     # (num_eqs, n_ids)
                var_matrix = bread @ (meat) @ bread.T / n_ids  # shape (num_estimating_eqs, num_estimating_eqs)
                
                # 2) evaluate sandwich variance at display cutoff only (first T_eval times)
                T_eval = display_time

                hazard_if_by_arm[key] = Phi[:T_eval, :]  # (T_eval, n_ids)
                n_ids_by_arm[key] = n_ids
                hazard_cov_eval = var_matrix[:T_eval, :T_eval]
                hazard_cov_by_arm[key] = hazard_cov_eval
                
                # Now get the variance estimator for survival at display time
                hazards = hazards_all[:T_eval]  # shape (T_eval, )
                survival = np.prod(1 - hazards)
                survival_by_arm[key] = survival
                hazards_by_arm[key] = hazards

                dS_dhazard = -survival / (1 - hazards)  # shape (T_eval, )
                var_survival = dS_dhazard.dot(hazard_cov_eval).dot(dS_dhazard.T)
                var_survival_by_arm[key] = var_survival
                results_dict.update({
                    f'outcome_rate_{key}_var': var_survival,
                    f'outcome_rate_{key}_std': np.sqrt(var_survival),
                })
                lg.info(f"Variance estimator of survival at time {cutoff_time_of_observation_display} for arm '{key}': {var_survival}")
                # also print for debugging

        pairs = (
            (("control", "intervention"),)
            if canonical_pair
            else tuple(
                (reference, comparison)
                for reference in cloned_dfs
                for comparison in cloned_dfs
                if comparison != reference
            )
        )
        for reference, comparison in pairs:
            difference_variance, log_ratio_variance, risk_covariance = (
                _pairwise_variance(
                    reference,
                    comparison,
                    hazard_if_by_arm=hazard_if_by_arm,
                    hazard_cov_by_arm=hazard_cov_by_arm,
                    survival_by_arm=survival_by_arm,
                    hazards_by_arm=hazards_by_arm,
                    var_survival_by_arm=var_survival_by_arm,
                    n_ids_by_arm=n_ids_by_arm,
                )
            )
            if not np.isfinite(difference_variance):
                continue
            if canonical_pair:
                results_dict.update(
                    {
                        "rd_var": float(difference_variance),
                        "rd_std": float(np.sqrt(difference_variance)),
                        "log_rr_var": (
                            float(log_ratio_variance)
                            if np.isfinite(log_ratio_variance)
                            else np.nan
                        ),
                        "log_rr_std": (
                            float(np.sqrt(log_ratio_variance))
                            if np.isfinite(log_ratio_variance)
                            else np.nan
                        ),
                    }
                )
                control_variance = var_survival_by_arm["control"]
                intervention_variance = var_survival_by_arm["intervention"]
                denominator = np.sqrt(control_variance * intervention_variance)
                correlation = risk_covariance / denominator if denominator > 0 else 0
                independent_variance = control_variance + intervention_variance
                reduction = 2.0 * risk_covariance / independent_variance * 100
                lg.info("--- Variance Component Diagnostics (Asymptotic) ---")
                lg.info("  Var(Control Risk):      %.8f", control_variance)
                lg.info("  Var(Intervention Risk): %.8f", intervention_variance)
                lg.info("  Cov(Control, Interv):   %.8f", risk_covariance)
                lg.info("  Analytical Correlation: %.4f", correlation)
                lg.info("  Variance Reduction:     %.2f%%", reduction)
                lg.info("  Final RD Var:           %.8f", difference_variance)
                lg.info("--------------------------------------------------")
            else:
                suffix = f"{comparison}_vs_{reference}"
                results_dict[f"risk_difference_{suffix}_std"] = float(
                    np.sqrt(max(difference_variance, 0.0))
                )
                results_dict[f"log_risk_ratio_{suffix}_std"] = (
                    float(np.sqrt(max(log_ratio_variance, 0.0)))
                    if np.isfinite(log_ratio_variance)
                    else np.nan
                )
        
        def _weight_stats(series: pd.Series) -> dict[str, float]:
            if series.empty:
                return {
                    'min': np.nan,
                    'q25': np.nan,
                    'median': np.nan,
                    'q75': np.nan,
                    'q99': np.nan,
                    'max': np.nan,
                    'count': 0.0,
                    'mean': np.nan,
                    'std': np.nan,
                    'ess': np.nan,
                }
            # 有効サンプルサイズ (Kish's ESS): ESS = (Σw)² / Σ(w²)
            sum_w = series.sum()
            sum_w2 = (series ** 2).sum()
            ess = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else np.nan
            return {
                'min': float(series.min()),
                'q25': float(series.quantile(0.25)),
                'median': float(series.quantile(0.5)),
                'q75': float(series.quantile(0.75)),
                'q99': float(series.quantile(0.99)),
                'max': float(series.max()),
                'count': float(series.count()),
                'mean': float(series.mean()),
                'std': float(series.std(ddof=1)) if series.count() > 1 else 0.0,
                'ess': float(ess),
            }

        combined_weights = pd.concat(weight_summary_frames, ignore_index=True) if weight_summary_frames else pd.DataFrame()
        weight_export_df = None
        if return_weight_df:
            weight_export_df = pd.concat(weight_export_frames, ignore_index=True) if weight_export_frames else pd.DataFrame()

        def _filter_weights(
            df: pd.DataFrame,
            *,
            arm: str | None,
            col: str,
            threshold: float | Mapping[str, float],
            compare_col: str,
        ) -> pd.Series:
            if df.empty:
                return pd.Series(dtype=float)
            subset = df
            if arm is not None:
                subset = subset[subset['arm'] == arm]
            if isinstance(threshold, Mapping):
                thresholds = subset["arm"].map(threshold)
                mask = subset[compare_col] <= thresholds
            else:
                mask = subset[compare_col] <= threshold
            return subset.loc[mask, col]

        grace_weights = _filter_weights(
            combined_weights, arm=None, col=analysis_weight_col,
            threshold=cutoff_time_of_intervention, compare_col='tstart'
        )
        obs30_weights = _filter_weights(
            combined_weights, arm=None, col=analysis_weight_col,
            threshold=cutoff_time_of_observation_display, compare_col='tend'
        )

        grace_stats = _weight_stats(grace_weights)
        obs30_stats = _weight_stats(obs30_weights)

        results_dict.update({
            'ipw_weight_grace_min': grace_stats['min'],
            'ipw_weight_grace_q25': grace_stats['q25'],
            'ipw_weight_grace_median': grace_stats['median'],
            'ipw_weight_grace_q75': grace_stats['q75'],
            'ipw_weight_grace_q99': grace_stats['q99'],
            'ipw_weight_grace_max': grace_stats['max'],
            'ipw_weight_grace_count': grace_stats['count'],
            'ipw_weight_grace_mean': grace_stats['mean'],
            'ipw_weight_grace_std': grace_stats['std'],
            'ipw_weight_grace_ess': grace_stats['ess'],
            'ipw_weight_30day_min': obs30_stats['min'],
            'ipw_weight_30day_q25': obs30_stats['q25'],
            'ipw_weight_30day_median': obs30_stats['median'],
            'ipw_weight_30day_q75': obs30_stats['q75'],
            'ipw_weight_30day_q99': obs30_stats['q99'],
            'ipw_weight_30day_max': obs30_stats['max'],
            'ipw_weight_30day_count': obs30_stats['count'],
            'ipw_weight_30day_mean': obs30_stats['mean'],
            'ipw_weight_30day_std': obs30_stats['std'],
            'ipw_weight_30day_ess': obs30_stats['ess'],
            'ipw_weight_grace_values': [],
            'ipw_weight_30day_values': [],
        })

        result_arm_order = (
            ("control", "intervention") if canonical_pair else tuple(cloned_dfs)
        )
        for arm in result_arm_order:
            arm_grace = _filter_weights(
                combined_weights, arm=arm, col=analysis_weight_col,
                threshold=grace_periods[arm], compare_col='tstart'
            )
            arm_obs30 = _filter_weights(
                combined_weights, arm=arm, col=analysis_weight_col,
                threshold=cutoff_time_of_observation_display, compare_col='tend'
            )

            arm_grace_stats = _weight_stats(arm_grace)
            arm_obs30_stats = _weight_stats(arm_obs30)

            results_dict.update({
                f'ipw_weight_grace_{arm}_min': arm_grace_stats['min'],
                f'ipw_weight_grace_{arm}_q25': arm_grace_stats['q25'],
                f'ipw_weight_grace_{arm}_median': arm_grace_stats['median'],
                f'ipw_weight_grace_{arm}_q75': arm_grace_stats['q75'],
                f'ipw_weight_grace_{arm}_q99': arm_grace_stats['q99'],
                f'ipw_weight_grace_{arm}_max': arm_grace_stats['max'],
                f'ipw_weight_grace_{arm}_count': arm_grace_stats['count'],
                f'ipw_weight_grace_{arm}_mean': arm_grace_stats['mean'],
                f'ipw_weight_grace_{arm}_std': arm_grace_stats['std'],
                f'ipw_weight_grace_{arm}_ess': arm_grace_stats['ess'],
                f'ipw_weight_grace_{arm}_values': [],
                f'ipw_weight_30day_{arm}_min': arm_obs30_stats['min'],
                f'ipw_weight_30day_{arm}_q25': arm_obs30_stats['q25'],
                f'ipw_weight_30day_{arm}_median': arm_obs30_stats['median'],
                f'ipw_weight_30day_{arm}_q75': arm_obs30_stats['q75'],
                f'ipw_weight_30day_{arm}_q99': arm_obs30_stats['q99'],
                f'ipw_weight_30day_{arm}_max': arm_obs30_stats['max'],
                f'ipw_weight_30day_{arm}_count': arm_obs30_stats['count'],
                f'ipw_weight_30day_{arm}_mean': arm_obs30_stats['mean'],
                f'ipw_weight_30day_{arm}_std': arm_obs30_stats['std'],
                f'ipw_weight_30day_{arm}_ess': arm_obs30_stats['ess'],
                f'ipw_weight_30day_{arm}_values': [],
            })

        lg.info(f"{cutoff_time_of_observation_display}-d outcome:")
        for arm in cloned_dfs:
            risk = results_dict[f"outcome_rate_{arm}"]
            lg.info(f"virtual {arm} group: {risk:.4f} ({risk * 100:.2f}%)")
        if canonical_pair:
            lg.info("Group differences:")
            lg.info(f"  Rate Difference: {results_dict['outcome_rate_difference']:.4f}")
            lg.info(f"  Rate Ratio: {results_dict['outcome_rate_ratio']:.4f}")

    return _PipelineOutput(
        estimates=results_dict,
        weights=weight_export_df if return_weight_df else None,
    )

def cloning_censoring_weighting_analysis(
    df_joined,
    strategies,
    ipw_explanatory_formula,
    configs,
    cutoff_time_of_intervention,
    cutoff_time_of_observation=30,
    cutoff_time_of_observation_display=10,
    seed=4,
    logger_ = None, 
    verbose=None,
    weight_col: str | None = None,
    return_cloned_dfs: bool = False,
    return_weight_df: bool = False,
):
    lg = TagAdapter(logger_ or _base_logger, tag="cloning_censoring_weighting_analysis")
    with temp_stderr_logging(lg.logger, level = to_level(verbose), user_supplied=(logger_ is not None)):
        treatment_var = configs['treatment_var']
        outcome_var = configs['outcome_var']
        censor_vars = configs.get('censor_vars', None)
        time_varying_vars = configs.get('time_varying_vars', None)
                
        np.random.seed(seed)
        random.seed(seed)
        
        if len(strategies) < 2:
            raise ValueError("strategies must contain at least two entries")

        treatment_col = treatment_var.name
        outcome_col = outcome_var.name
        censor_cols = []
        if censor_vars is not None:
            censor_cols = [var.name for var in censor_vars]
        time_varying_cols = None
        if time_varying_vars is not None:
            time_varying_cols = [var.name for var in time_varying_vars]
        
        # --- 1. Preprocessing ---
        df_joined = ccw_preprocessing(
            df_joined,
            cutoff_time_of_observation,
            outcome_col,
            censor_cols,
        )

        # --- 2. Cloning & Censoring ---
        lg.info("=== Clone-Censor-Weight Analysis Started ===")
        lg.info("1. Cloning...")
        cloned_dfs = {}
        for key in strategies:
            cloned_dfs[key] = df_joined.copy()
            cloned_dfs[key]['arm'] = key

        lg.info("2. Censoring...")
        lg.info("2. 変数の設定...")
        for key, strategy in strategies.items():
            cloned_dfs[key] = strategy.artificial_censor(
                cloned_dfs[key],
                id_col="id",
                time_col="time",
                treatment_col=treatment_col,
            )
            cloned_dfs[key] = create_counting_process_format(
                cloned_dfs[key], outcome_col, censor_cols, time_varying_cols
            )
            cloned_dfs[key] = remove_rows_after_fup(cloned_dfs[key])
            cloned_dfs[key] = integrate_censoring_columns(cloned_dfs[key], censor_cols)

        cloned_dfs_original = {}
        if return_cloned_dfs:
            cloned_dfs_original = {key: df.copy() for key, df in cloned_dfs.items()}
        # --- 3. Weighting & Analysis を別関数に委譲 ---
        res_ccw = run_ccw_weighting_and_analysis(
            cloned_dfs=cloned_dfs,
            strategies=strategies,
            ipw_explanatory_formula=ipw_explanatory_formula,
            configs=configs,
            cutoff_time_of_intervention=cutoff_time_of_intervention,
            cutoff_time_of_observation_display=cutoff_time_of_observation_display,
            logger_=lg.logger,
            verbose=verbose,
            weight_col=weight_col,
            return_weight_df=return_weight_df,
        )
        if return_cloned_dfs:
            res_ccw.cloned_data = cloned_dfs_original
        return res_ccw
