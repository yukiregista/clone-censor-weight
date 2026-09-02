from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from patsy import EvalEnvironment, bs, cr
from patsy.builtins import Treatment
from statsmodels.api import GLM, families

from ccw._logging import LoggerLike, get_tagged


def grace_cat(x, grace_end):
    x_arr = np.asarray(x)
    return np.where(x_arr <= grace_end, x_arr, -1)


def ccw_preprocessing(
    df_joined: pd.DataFrame,
    cutoff_time_of_observation: int,
    outcome_col: str,
    censor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    データフレームをCCW解析に適した形式に変換する。
    X_{t-1} -> A_{t-1} -> D_t -> X_t -> A_tの順序を想定。
    Parameters
    ----------
    df_joined : pd.DataFrame
        入力データフレーム。列 'time', `outcome_col` を含む。
    """
    # create copy
    df = df_joined.copy() # deepcopy by default
    
    # データをcutoff_time_of_observationでフィルタリング
    df = df[df['time'] <= cutoff_time_of_observation].copy()
    
    # censor_cols - censor_col == 1 なる最小index + 1 の時点以降を削除 for each individual
    if censor_cols is not None:
        for censor_col in censor_cols:
            first_cens_time = (
                df.loc[df[censor_col] == 1]
                  .groupby('id')['time']
                  .min()
            )
            df[f'time_to_{censor_col}'] = df['id'].map(first_cens_time)
            df = df[df[f'time_to_{censor_col}'].isna() | (df['time'] <= df[f'time_to_{censor_col}'])].copy()     
            
    
    # time_to_outcome: stores the time when the outcome occurs, or nan.
    time_to_outcome = df.loc[df[outcome_col]==1].groupby('id')['time'].min()
    df['time_to_outcome_cutoff_obs'] = df['id'].map(time_to_outcome)

    return df


def create_counting_process_format(
    df: pd.DataFrame,
    outcome_col: str,
    censor_cols: list[str] | None = None,
    time_varying_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    データを時間軸で分割したカウンティングプロセス形式（開始-終了形式）に変換する
    1. 'tstart', 'tend' ('tstart'+1) を追加
    2. 'outcome_tend', 'artificial_censor_tstart' を追加
    """
    
    df['tstart'] = df['time']
    df['tend'] = df['time'] + 1
    
    df['outcome_tend'] = df[outcome_col].shift(-1, fill_value=0)
    df['artificial_censor_tstart'] = df['artificial_censor']
    
    if censor_cols is not None:
        for censor_col in censor_cols:
            df[f'{censor_col}_tstart'] = df[censor_col]
            
    if time_varying_cols is not None:
        # 各患者のベースライン値（時間0での値）を取得
        for col in time_varying_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
                
            # 各患者IDについて時間0での値を取得
            baseline_col_name = f'{col}_baseline'
            
            # 時間0での値を取得（患者ごと）
            time_zero_values = df[df['time'] == 0].set_index('id')[col]
            
            # 全ての行に対してベースライン値をマッピング
            df[baseline_col_name] = df['id'].map(time_zero_values)
    return df


def remove_rows_after_fup(df: pd.DataFrame):
    # remove any rows where tend > time_to_outcome_cutoff_obs or tstart > time_to_artificial_censor
    # replace NaN with np.inf for comparison
    df_copy = df.copy()
    df_copy['time_to_outcome_cutoff_obs_inf'] = df_copy['time_to_outcome_cutoff_obs'].fillna(np.inf)
    df_copy['time_to_artificial_censor_inf'] = df_copy['time_to_artificial_censor'].fillna(np.inf)
    df_copy = df_copy[df_copy['time_to_outcome_cutoff_obs_inf'] >= df_copy['tend']]
    # 本当はもうartificial_censorでやっているので以下は不要
    df_copy = df_copy[df_copy['time_to_artificial_censor_inf'] >= df_copy['tstart']]
    # remove the temporary columns
    df_copy = df_copy.drop(columns=['time_to_outcome_cutoff_obs_inf', 'time_to_artificial_censor_inf'])
    return df_copy

def integrate_censoring_columns(
    df: pd.DataFrame,
    censor_cols: list[str]
) -> pd.DataFrame:
    """
    複数のcensoring列を統合して、1つの列 'CENSOR_tstart' を作成する。
    """
    if censor_cols is None or len(censor_cols) == 0:
        # use artificial_censor_tstart 
        df['CENSOR_tstart'] = df['artificial_censor_tstart']
    else:
        # 複数のcensoring列を統合
        ## get or of all censoring columns
        df['CENSOR_tstart'] = df[['artificial_censor_tstart'] + [item + "_tstart" for item in censor_cols]].any(axis=1).astype(int)
    
    # any columns after any kind of censoring should have been removed already
    
    return df
    
    
    

def prepare_ccw_analysis_part(df: pd.DataFrame):
    """remove rows with=artificial_censor_tstart==1"""
    return df[df['CENSOR_tstart'] != 1]


def _dynamically_adjust_spline(formula: str, time_variable_name: str, unique_times: int, logger_: LoggerLike | None = None) -> str:
    """
    Dynamically adjusts the spline term in a formula based on the number of unique time points.
    
    1. If unique_times <= 3, converts spline to a categorical variable C(time_variable_name).
    2. If unique_times > 3, adjusts spline df to be unique_times - 1 if df is too high.
    """
    lg = get_tagged(logger_, "_dynamically_adjust_spline")
    if formula is None:
        return None

    spline_pattern = re.compile(r'(cr|bs)\(\s*' + re.escape(time_variable_name) + r'[^)]+\)')
    match = spline_pattern.search(formula)

    if not match:
        return formula

    # Case 1: Few unique time points, convert spline to categorical.
    if unique_times <= 3:
        lg.warning(f"Warning: Only {unique_times} unique time points. Converting spline for '{time_variable_name}' to categorical 'C({time_variable_name})'.")
        return spline_pattern.sub(f'C({time_variable_name})', formula)

    # Case 2: Enough time points, check and adjust df if necessary.
    df_pattern = re.compile(r'df\s*=\s*(\d+)')
    df_match = df_pattern.search(match.group(0))
    
    if not df_match:
        # If df is not specified, patsy uses a default. Assume it's okay.
        return formula

    df_val = int(df_match.group(1))

    if df_val >= unique_times:
        new_df = unique_times - 1
        lg.warning(f"Warning: Spline df={df_val} is too high for {unique_times} unique time points. Adjusting to df={new_df}.")
        
        # Replace only the df value within the spline term
        original_spline_term = match.group(0)
        adjusted_spline_term = df_pattern.sub(f'df={new_df}', original_spline_term)
        
        # Replace the original spline term in the whole formula
        return formula.replace(original_spline_term, adjusted_spline_term)

    return formula

def exp_prob(
    data: pd.DataFrame,
    exposure: str,
    family: str,
    link: str,
    denominator: str,
    id_col: str,
    timevar: str,
    treatment_col: str | None = None,
    mask_func: Callable[[pd.DataFrame, str, str], pd.Series] | None = None,
    logger_: LoggerLike | None = None,
    return_details: bool = False,
    weight_col: str | None = None,
    **kwargs
):
    lg = get_tagged(logger_, "exp_prob")
    # 0. Map 'ns(' and 'bs(' alias to natural spline 'cr('
    denominator = denominator.replace('ns(', 'cr(').replace('bs(', 'cr(')

    # 1. Basic checks
    allowed_families = {'binomial'}
    if family not in allowed_families:
        raise ValueError(f"Invalid family: {family}")
    df = data.copy()

    # 2. Sort and prepare
    df = df.sort_values([id_col, timevar])
    sel = pd.Series(1, index=df.index)

    # apply mask to sel
    sel = mask_func(df, timevar, treatment_col) if mask_func is not None else pd.Series(True, index=df.index)
    sel = sel.astype(int)

    # for debug
    lg.debug("Unique time for sel==1: %s", df[sel == 1][timevar].unique())

    # 4. Prepare evaluation environment
    eval_env = EvalEnvironment.capture(0)
    # expose spline functions
    eval_env.namespace['cr'] = cr
    eval_env.namespace['bs'] = bs

    # 5. Model fitting by family
    mod_den = None
    score = None
    hessian = None
    X_df = None
    score_by_row = None
    score_by_id = None
    prob_den = pd.Series(np.nan, index=df.index)

    if family == 'binomial':
        # supported link functions
        link_map = {
            'logit': families.links.Logit(),
            'probit': families.links.Probit(),
            'cloglog': families.links.CLogLog(),
            'log': families.links.Log()
        }
        if link not in link_map:
            raise ValueError(f"Unsupported link '{link}' for binomial family")
        fam = families.Binomial(link_map[link])

        df_sub = df[sel == 1]
        unique_times_for_fitting = df_sub[timevar].nunique()
        denominator = _dynamically_adjust_spline(denominator, 'tstart', unique_times_for_fitting, logger_=lg)
        # denominator model
        freq_weights = None
        if weight_col is not None:
            freq_weights = df_sub[weight_col].to_numpy()
        mod_den = GLM.from_formula(
            f"{exposure} ~ {denominator}", data=df_sub,
            family=fam, eval_env=eval_env, freq_weights=freq_weights, **kwargs
        ).fit()
        # only apply prob_den to rows where sel==1
        prob_den.loc[df_sub.index] = mod_den.predict(df_sub)

        # score and hessian for log-likelihood (gamma)
        X = mod_den.model.exog
        y = mod_den.model.endog
        mu = mod_den.fittedvalues.values
        weight_factor = freq_weights if freq_weights is not None else np.ones_like(mu)
        w = mu * (1.0 - mu) * weight_factor

        X_df = pd.DataFrame(X, index=df_sub.index, columns=mod_den.model.exog_names)
        score_by_row_sub = pd.DataFrame(
            X * (y - mu)[:, None] * weight_factor[:, None],
            index=df_sub.index,
            columns=mod_den.model.exog_names
        )
        
        score_by_row = pd.DataFrame(
            0.0,
            index=df.index,
            columns=mod_den.model.exog_names
        )
        score_by_row.loc[df_sub.index] = score_by_row_sub
        
        score_by_row_with_id = score_by_row.copy()
        score_by_row_with_id[id_col] = df[id_col]
        score_by_id = score_by_row_with_id.groupby(id_col)[mod_den.model.exog_names].sum()
        
        # Total score (sum over all ids)
        score = score_by_id.sum(axis=0).values
        
        # Hessian: use rows from df_sub but normalize by n_ids from df
        hessian = -(X.T @ (X * w[:, None]))

    if not return_details:
        return prob_den

    return {
        "prob": prob_den,
        "fit": mod_den,
        "score": score,              # normalized by n_ids
        "hessian": hessian,          # normalized by n_ids
        "sel": sel,
        "X": X_df,
        "score_by_id": score_by_id,        # all unique ids with zeros for missing
    }


def ipcwtm(
    data: pd.DataFrame,
    censor_indicator: str,
    family: str,
    link: str,
    denominator: str,
    id_col: str,
    timevar: str,
    numerator: str | None = None,
    trunc: float | None = None,
    treatment_col: str | None = None,
    mask_func: Callable[[pd.DataFrame, str, str], pd.Series] | None = None,
    logger_: LoggerLike | None = None,
    weight_col: str | None = None,
    formula_namespace: dict | None = None,
    **kwargs
) -> dict:
    """
    Calculate inverse probability of censoring weights (IPCW) for various outcome families.

    The argument 'denominator' (and 'numerator') may include 'ns(x, ...)' which will be
    automatically mapped to patsy.cr (natural spline) for compatibility.
    
    mask_func: Callable, optional
        A function that returns a boolean mask to filter the data before fitting the models.
    formula_namespace: dict, optional
        Extra symbols usable inside patsy formulas (e.g., {'grace_end': 2}).
    """
    lg = get_tagged(logger_, "ipcwtm")
    # 0. Map 'ns(' and 'bs(' alias to natural spline 'cr('
    denominator = denominator.replace('ns(', 'cr(').replace('bs(', 'cr(')
    if numerator is not None:
        numerator = numerator.replace('ns(', 'cr(').replace('bs(', 'cr(')

    # 1. Basic checks
    allowed_families = {'binomial'}
    if family not in allowed_families:
        raise ValueError(f"Invalid family: {family}")
    df = data.copy()
    
    # 3. Determine which rows to use for model fitting based on the strategy's mask
    sel = mask_func(df, timevar, treatment_col) if mask_func is not None else pd.Series(True, index=df.index)
    sel = sel.astype(int)
    
    # for debug
    lg.debug("[debug info] unique time for sel==1: %s", df[sel==1][timevar].unique())
    
    # 4. Prepare evaluation environment for formulas
    formula_symbols = {
        'cr': cr,
        'bs': bs,
        'Treatment': Treatment,
        'grace_cat': grace_cat,
    }
    if formula_namespace:
        formula_symbols.update(formula_namespace)
    eval_env = EvalEnvironment.capture(0).with_outer_namespace(formula_symbols)

    # Containers for probabilities, initialized to 1 (no effect on weight)
    p_num = pd.Series(1.0, index=df.index)
    p_den = pd.Series(1.0, index=df.index)

    score = None
    hessian = None
    score_by_id = None

    # 5. Model fitting by family
    if family == 'binomial':
        link_map = {
            'logit': families.links.Logit(),
            'probit': families.links.Probit(),
            'cloglog': families.links.CLogLog(),
            'log': families.links.Log()
        }
        if link not in link_map:
            raise ValueError(f"Unsupported link '{link}' for binomial family")
        fam = families.Binomial(link_map[link])
        
        # Fit model only on the subset of data defined by the mask (`sel`)
        df_sub = df[sel == 1]
        unique_times_for_fitting = df_sub[timevar].nunique()
        denominator = _dynamically_adjust_spline(denominator, 'tstart', unique_times_for_fitting)
        if numerator is not None:
            numerator = _dynamically_adjust_spline(numerator, 'tstart', unique_times_for_fitting)
        
        # Denominator model for censoring probability
        freq_weights = None
        if weight_col is not None:
            freq_weights = df_sub[weight_col].to_numpy()
        mod_den = GLM.from_formula(
            f"{censor_indicator} ~ {denominator}", data=df_sub,
            family=fam, eval_env=eval_env, freq_weights=freq_weights, **kwargs
        ).fit()
        # Predict censoring probability for subset data
        prob_den = mod_den.predict(df_sub)
        
        # The probability of *not* being censored is 1 - prob_den.
        # This is used for the denominator of the weights.
        # We only update the probabilities for the rows included by the mask.
        # prob_den is a numpy array or Series with consecutive index, so we use direct assignment
        p_den.loc[df_sub.index] = 1 - prob_den
        
        # for varaicne computation
        dlogg_deta = pd.Series(0.0, index=df.index)
        dlogg_deta.loc[df_sub.index] = prob_den 

        X = mod_den.model.exog
        y = mod_den.model.endog
        mu = mod_den.fittedvalues.values
        weight_factor = freq_weights if freq_weights is not None else np.ones_like(mu)
        w = mu * (1.0 - mu) * weight_factor
        
        X_df = pd.DataFrame(X, index=df_sub.index, columns=mod_den.model.exog_names)


        # Per-row score contribution
        score_by_row_sub = pd.DataFrame(
            X * (y - mu)[:, None] * weight_factor[:, None],
            index=df_sub.index,
            columns=mod_den.model.exog_names
        )
        
        # Expand to full df index with zeros for missing rows
        score_by_row = pd.DataFrame(
            0.0,
            index=df.index,
            columns=mod_den.model.exog_names
        )
        score_by_row.loc[df_sub.index] = score_by_row_sub
        
        # Aggregate score by id
        score_by_row_with_id = score_by_row.copy()
        score_by_row_with_id[id_col] = df[id_col]
        score_by_id = score_by_row_with_id.groupby(id_col)[mod_den.model.exog_names].sum()
        
        # Total score
        score = score_by_id.sum(axis=0).values
        
        # Hessian
        hessian = -(X.T @ (X * w[:, None]))

        # Numerator model (for stabilized weights) if provided
        if numerator:
            mod_num = GLM.from_formula(
                f"{censor_indicator} ~ {numerator}", data=df_sub,
                family=fam, eval_env=eval_env, freq_weights=freq_weights, **kwargs
            ).fit()
            # Predict for the same subset used for fitting
            prob_num = mod_num.predict(df_sub)
            p_num.loc[df_sub.index] = 1 - prob_num
        else:
            mod_num = None
        den_model = mod_den

    else:
        raise NotImplementedError(f"Family '{family}' not implemented yet")

    # 6. Ensure probabilities are pandas Series (already are, but good practice)
    if not isinstance(p_num, pd.Series):
        p_num = pd.Series(p_num, index=df.index)
    if not isinstance(p_den, pd.Series):
        p_den = pd.Series(p_den, index=df.index)

    # 7. Compute weights by taking the cumulative product of probabilities for each individual
    df['w_num'] = p_num.groupby(df[id_col]).cumprod()
    df['w_den'] = p_den.groupby(df[id_col]).cumprod()
    df['ipw'] = df['w_num'] / df['w_den']

    g_t = p_den.copy()
    h_full = g_t.groupby(df[id_col]).cumprod()
    h_prev = h_full.groupby(df[id_col]).shift(1, fill_value=1.0)

    # 8. Truncate weights if a quantile is specified
    if trunc is not None:
        low = df['ipw'].quantile(trunc)
        high = df['ipw'].quantile(1-trunc)
        df['ipw_trunc'] = df['ipw'].clip(low, high)

    # 9. Prepare and return results
    result = {
        'ipw.weights': df['ipw'].values,
        'selvar': sel.values,
        'den.mod': den_model,
        'score': score,
        'hessian': hessian,
        'score_by_id': score_by_id,
        "sel": sel,
        "X": X_df,
        "dlogg_deta": dlogg_deta,
        "g_t": g_t,
        "h_prev": h_prev,
    }
    if numerator:
        result['num.mod'] = mod_num
    if trunc is not None:
        result['weights.trunc'] = df['ipw_trunc'].values
    return result

def weighted_kaplan_meier(
    split_data: pd.DataFrame,
    time_col,
    weight_col,
    outcome_col,
    logger_: LoggerLike | None = None,
    return_details: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], dict]:
    """
    split_dataをいれて、各時点でのsurvival probabilityを返す
    ----------
    split_data: pd.DataFrame
        列 'tend', 'ipw.weights', 'event_outcome_tend' を含む。
        (time_col == tend)

    ----------
    """
    lg = get_tagged(logger_, "weighted_kaplan_meier")

    # 全ての時点を取得してソート
    unique_times = sorted(split_data[time_col].unique())

    times = []
    survival_probs = []
    survival_prob = 1.0

    # (NEW) collect minimal KM ingredients for variance later
    if return_details:
        risk_idx_list = []
        event_idx_list = []
        weighted_n_at_risk_list = []
        weighted_n_events_list = []
        hazard_list = []
        
    eeq_by_row = pd.DataFrame(
        0.0,
        index=split_data.index,
        columns = unique_times
    )
    
    dm_dhazard = np.zeros((len(unique_times), len(unique_times)))
    

    for t_idx, t in enumerate(unique_times):
        # 時点tでのリスクセット: tend == t となる全レコードのipw.weightsの合計
        at_risk_records = split_data[split_data[time_col] == t]
        weighted_n_at_risk = at_risk_records[weight_col].sum()

        # 時点tでのイベント: tend == t かつ event_outcome_tend == 1
        event_records = split_data[(split_data[time_col] == t) & (split_data[outcome_col] == 1)]
        weighted_n_events = event_records[weight_col].sum()

        lg.debug(
            f"Time {t}: At risk (weighted) = {weighted_n_at_risk:.2f}, "
            f"Events (weighted) = {weighted_n_events:.2f}"
        )

        hazard = 0.0
        # 生存確率の更新
        if weighted_n_at_risk > 0:
            hazard = weighted_n_events / weighted_n_at_risk
            survival_prob *= (1 - hazard)

        times.append(t)
        survival_probs.append(survival_prob)

        if return_details:
            # compute per-row estimating equation components
            eeq_by_row.loc[at_risk_records.index, t] = split_data.loc[at_risk_records.index, weight_col] * (hazard - split_data.loc[at_risk_records.index, outcome_col])
            
            risk_idx_list.append(at_risk_records.index)
            event_idx_list.append(event_records.index)
            weighted_n_at_risk_list.append(float(weighted_n_at_risk))
            weighted_n_events_list.append(float(weighted_n_events))
            hazard_list.append(float(hazard))
            dm_dhazard[t_idx, t_idx] = weighted_n_at_risk

        lg.debug(f"  -> Survival probability = {survival_prob:.4f}")

    # create per-individual estimating equation components
    
    if return_details:
        eeq_by_row['id'] = split_data['id']
        eeq_by_id = eeq_by_row.groupby('id').sum()
    
    if not return_details:
        return times, survival_probs

    details = {
        "times": times,
        "risk_idx": risk_idx_list,
        "event_idx": event_idx_list,
        "weighted_n_at_risk": weighted_n_at_risk_list,
        "weighted_n_events": weighted_n_events_list,
        "hazard": hazard_list,
        "eeq_by_id": eeq_by_id,
        "dm_dhazard": dm_dhazard,
    }
    return times, survival_probs, details


def calculate_strategy_outcomes(
    strategy_data: dict[str, pd.DataFrame],
    time_col: str,
    weight_col: str,
    outcome_col: str,
    cutoff_time_of_observation_display: int,
) -> tuple[dict[str, float], dict]:
    """Estimate risks and pairwise contrasts for arbitrary named strategies."""

    canonical_pair = set(strategy_data) == {"control", "intervention"}
    strategy_names = (
        ("control", "intervention") if canonical_pair else tuple(strategy_data)
    )
    details: dict[str, object] = {}
    risks: dict[str, float] = {}
    for name in strategy_names:
        frame = strategy_data[name]
        times, survival, km_details = weighted_kaplan_meier(
            frame,
            time_col=time_col,
            weight_col=weight_col,
            outcome_col=outcome_col,
            return_details=True,
        )
        if not times:
            survival_at_cutoff = np.nan
        elif cutoff_time_of_observation_display in times:
            survival_at_cutoff = survival[
                times.index(cutoff_time_of_observation_display)
            ]
        elif cutoff_time_of_observation_display > max(times):
            survival_at_cutoff = survival[-1]
        else:
            prior = [
                value
                for time, value in zip(times, survival, strict=True)
                if time < cutoff_time_of_observation_display
            ]
            survival_at_cutoff = prior[-1] if prior else 1.0
        risk = 1.0 - survival_at_cutoff
        risks[name] = risk
        details[f"km_{name}"] = km_details
    details["cutoff_time_of_observation_display"] = cutoff_time_of_observation_display
    for name in strategy_names:
        details[f"survival_prob_{name}_at_cutoff"] = 1.0 - risks[name]

    results: dict[str, float] = {
        f"outcome_rate_{name}": risks[name] for name in strategy_names
    }
    if canonical_pair:
        control_risk = risks["control"]
        intervention_risk = risks["intervention"]
        difference = intervention_risk - control_risk
        ratio = _risk_ratio(intervention_risk, control_risk)
        results.update(
            {
                "outcome_rate_difference": difference,
                "outcome_rate_ratio": ratio,
                "outcome_rate_log_ratio": _log_ratio(ratio),
            }
        )
        return results, details

    for reference, reference_risk in risks.items():
        for comparison, comparison_risk in risks.items():
            if comparison == reference:
                continue
            suffix = f"{comparison}_vs_{reference}"
            difference = comparison_risk - reference_risk
            ratio = _risk_ratio(comparison_risk, reference_risk)
            results[f"risk_difference_{suffix}"] = difference
            results[f"risk_ratio_{suffix}"] = ratio
            results[f"log_risk_ratio_{suffix}"] = _log_ratio(ratio)
    return results, details


def _risk_ratio(comparison_risk: float, reference_risk: float) -> float:
    if not (np.isfinite(comparison_risk) and np.isfinite(reference_risk)):
        return np.nan
    if reference_risk > 0:
        return comparison_risk / reference_risk
    return float("inf") if comparison_risk > 0 else 1.0


def _log_ratio(ratio: float) -> float:
    if np.isfinite(ratio) and ratio > 0:
        return np.log(ratio)
    if np.isinf(ratio) and ratio > 0:
        return np.inf
    return np.nan
