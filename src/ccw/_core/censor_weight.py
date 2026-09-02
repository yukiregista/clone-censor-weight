import numpy as np
import pandas as pd

from .analysis import exp_prob, ipcwtm


def _strategy_censoring_prob_mask(strategy):
    strategy_mask = getattr(strategy, "censoring_prob_mask", None)
    if strategy_mask is None:
        return None

    def mask(data: pd.DataFrame, time_col: str, treatment_col: str) -> pd.Series:
        return strategy_mask(
            data,
            id_col="id",
            time_col=time_col,
            treatment_col=treatment_col,
        )

    return mask


def day0_mask(df: pd.DataFrame, time_col: str, _intervention_col: str):
    keep_mask = df[time_col] > 0
    return pd.Series(keep_mask.values, index=df.index)


def simple_all_at_once(
    df,
    treatment_col,
    ipw_explanatory_formula,
    strategy,
    censor_cols=None,
    censor_day0=None,
    return_details: bool = True,
    weight_col: str | None = None,
    formula_namespace: dict | None = None,
):
    mask_func = _strategy_censoring_prob_mask(strategy) if not censor_cols else None
    res = ipcwtm(
        data=df,
        censor_indicator="CENSOR_tstart",
        family="binomial",
        link="logit",
        numerator=None,
        denominator=ipw_explanatory_formula["all"],
        id_col="id",
        timevar="tstart",
        treatment_col=treatment_col,
        mask_func=mask_func,
        weight_col=weight_col,
        formula_namespace=formula_namespace,
    )
    df["ipw.weights"] = res["ipw.weights"]

    dlogg_deta_censor = res["dlogg_deta"]
    X = pd.DataFrame(
        0.0, index=df.index, columns=["censor" + item for item in res["X"].columns]
    )
    assert set(res["X"].index).issubset(set(df.index)), (
        "res['X'] index is not subset of df index"
    )
    X.loc[res["X"].index, :] = res["X"].to_numpy()
    dlogg_dbeta_censor = dlogg_deta_censor.to_numpy()[:, None] * X.to_numpy()
    df.loc[:, X.columns] = dlogg_dbeta_censor
    dlogg_dbeta_censor_sum = df.groupby("id")[X.columns].cumsum().to_numpy()
    dW_dgamma = df.loc[:, "ipw.weights"].to_numpy()[:, None] * dlogg_dbeta_censor_sum
    if weight_col is not None:
        dW_dgamma = dW_dgamma * df.loc[:, weight_col].to_numpy()[:, None]

    df.drop(columns=X.columns, inplace=True)

    dW_dgamma = pd.DataFrame(dW_dgamma, index=df.index)

    details = {
        "score_gamma": res["score"],
        "score_by_id": res["score_by_id"],
        "hessian_gamma": res["hessian"],
        "dW_dgamma": dW_dgamma,
        "g_t": res["g_t"],
        "h_prev": res["h_prev"],
    }
    if not return_details:
        return df
    return df, details


def simple_separate(
    df,
    treatment_col,
    ipw_explanatory_formula,
    strategy,
    censor_cols=None,
    censor_day0=None,
    return_details: bool = True,
    weight_col: str | None = None,
    formula_namespace: dict | None = None,
):
    censor_cols = () if censor_cols is None else tuple(censor_cols)
    if censor_day0 is not None and len(censor_day0) != len(censor_cols):
        raise ValueError("censor_day0 must have one entry for each censor column")

    res = ipcwtm(
        data=df,
        censor_indicator="artificial_censor_tstart",
        family="binomial",
        link="logit",
        numerator=None,
        denominator=ipw_explanatory_formula["artificial_censor"],
        id_col="id",
        timevar="tstart",
        treatment_col=treatment_col,
        mask_func=_strategy_censoring_prob_mask(strategy),
        weight_col=weight_col,
        formula_namespace=formula_namespace,
    )
    df["ipw.weights.artificial_censor"] = res["ipw.weights"]
    g_t_total = res["g_t"].copy()

    dlogg_deta_censor = res["dlogg_deta"]
    X = pd.DataFrame(
        0.0,
        index=df.index,
        columns=["artificial_censor" + item for item in res["X"].columns],
    )
    assert set(res["X"].index).issubset(set(df.index)), (
        "res['X'] index is not subset of df index"
    )
    X.loc[res["X"].index, :] = res["X"].to_numpy()
    dlogg_dbeta_censor = dlogg_deta_censor.to_numpy()[:, None] * X.to_numpy()
    df.loc[:, X.columns] = dlogg_dbeta_censor
    dlogg_dbeta_censor_sum = df.groupby("id")[X.columns].cumsum().to_numpy()

    score = res["score"]
    hessian = res["hessian"]
    score_by_id = res["score_by_id"]

    censor_cols_res = {}
    for col_ind, censor_col in enumerate(censor_cols):
        mask_func = None
        if censor_day0 is not None and not censor_day0[col_ind]:
            mask_func = day0_mask
        res = ipcwtm(
            data=df,
            censor_indicator=f"{censor_col}_tstart",
            family="binomial",
            link="logit",
            numerator=None,
            denominator=ipw_explanatory_formula[censor_col],
            id_col="id",
            timevar="tstart",
            treatment_col=treatment_col,
            mask_func=mask_func,
            weight_col=weight_col,
            formula_namespace=formula_namespace,
        )
        df[f"ipw.weights.{censor_col}"] = res["ipw.weights"]
        censor_cols_res[censor_col] = res
        g_t_total = g_t_total * res["g_t"]

    df["ipw.weights"] = df[
        [f"ipw.weights.{censor_col}" for censor_col in censor_cols]
        + ["ipw.weights.artificial_censor"]
    ].prod(axis=1)
    dW_dgamma = df.loc[:, "ipw.weights"].to_numpy()[:, None] * dlogg_dbeta_censor_sum
    if weight_col is not None:
        dW_dgamma = dW_dgamma * df.loc[:, weight_col].to_numpy()[:, None]
    df.drop(columns=X.columns, inplace=True)

    if censor_cols:
        for censor_col in censor_cols:
            res_censor = censor_cols_res[censor_col]
            dlogg_deta_censor = res_censor["dlogg_deta"]
            X = pd.DataFrame(
                0.0,
                index=df.index,
                columns=[f"{censor_col}" + item for item in res_censor["X"].columns],
            )
            X.loc[res_censor["X"].index, :] = res_censor["X"].to_numpy()
            dlogg_dbeta_censor = dlogg_deta_censor.to_numpy()[:, None] * X.to_numpy()
            df.loc[:, X.columns] = dlogg_dbeta_censor
            dlogg_dbeta_censor_sum = df.groupby("id")[X.columns].cumsum().to_numpy()
            dW_dbeta_censor = (
                df.loc[:, "ipw.weights"].to_numpy()[:, None] * dlogg_dbeta_censor_sum
            )
            if weight_col is not None:
                dW_dbeta_censor = (
                    dW_dbeta_censor * df.loc[:, weight_col].to_numpy()[:, None]
                )
            dW_dgamma = np.hstack([dW_dgamma, dW_dbeta_censor])
            score = np.concatenate([score, res_censor["score"]])
            hessian = np.block(
                [
                    [
                        hessian,
                        np.zeros((hessian.shape[0], res_censor["score"].shape[0])),
                    ],
                    [
                        np.zeros((res_censor["score"].shape[0], hessian.shape[1])),
                        res_censor["hessian"],
                    ],
                ]
            )
            score_by_id = pd.concat([score_by_id, res_censor["score_by_id"]], axis=1)

            df.drop(columns=X.columns, inplace=True)

    dW_dgamma = pd.DataFrame(dW_dgamma, index=df.index)
    h_full = g_t_total.groupby(df["id"]).cumprod()
    h_prev = h_full.groupby(df["id"]).shift(1, fill_value=1.0)

    details = {
        "score_gamma": score,
        "score_by_id": score_by_id,
        "hessian_gamma": hessian,
        "dW_dgamma": dW_dgamma,
        "g_t": g_t_total,
        "h_prev": h_prev,
    }
    if not return_details:
        return df
    return df, details


def simple_ignore_censor_cols(
    df,
    treatment_col,
    ipw_explanatory_formula,
    strategy,
    censor_cols=None,
    censor_day0=None,
    return_details: bool = True,
    weight_col: str | None = None,
    formula_namespace: dict | None = None,
):
    res = ipcwtm(
        data=df,
        censor_indicator="artificial_censor_tstart",
        family="binomial",
        link="logit",
        numerator=None,
        denominator=ipw_explanatory_formula["artificial_censor"],
        id_col="id",
        timevar="tstart",
        treatment_col=treatment_col,
        mask_func=_strategy_censoring_prob_mask(strategy),
        weight_col=weight_col,
        formula_namespace=formula_namespace,
    )
    df["ipw.weights"] = res["ipw.weights"]

    dlogg_deta_censor = res["dlogg_deta"]
    X = pd.DataFrame(
        0.0,
        index=df.index,
        columns=["artificial_censor" + item for item in res["X"].columns],
    )
    X.loc[res["X"].index, :] = res["X"].to_numpy()
    dlogg_dbeta_censor = dlogg_deta_censor.to_numpy()[:, None] * X.to_numpy()
    df.loc[:, X.columns] = dlogg_dbeta_censor
    dlogg_dbeta_censor_sum = df.groupby("id")[X.columns].cumsum().to_numpy()
    dW_dgamma = df.loc[:, "ipw.weights"].to_numpy()[:, None] * dlogg_dbeta_censor_sum
    if weight_col is not None:
        dW_dgamma = dW_dgamma * df.loc[:, weight_col].to_numpy()[:, None]

    df.drop(columns=X.columns, inplace=True)

    dW_dgamma = pd.DataFrame(dW_dgamma, index=df.index)

    details = {
        "score_gamma": res["score"],
        "score_by_id": res["score_by_id"],
        "hessian_gamma": res["hessian"],
        "dW_dgamma": dW_dgamma,
        "g_t": res["g_t"],
        "h_prev": res["h_prev"],
    }
    if not return_details:
        return df
    return df, details


def use_treatment_prob(
    df,
    treatment_col,
    ipw_explanatory_formula,
    strategy,
    censor_cols=None,
    censor_day0=None,
    return_details: bool = True,
    weight_col: str | None = None,
    formula_namespace: dict | None = None,
):
    censor_cols = () if censor_cols is None else tuple(censor_cols)
    if censor_day0 is not None and len(censor_day0) != len(censor_cols):
        raise ValueError("censor_day0 must have one entry for each censor column")

    res_treat = exp_prob(
        data=df,
        exposure=treatment_col,
        family="binomial",
        link="logit",
        denominator=ipw_explanatory_formula["treatment"],
        id_col="id",
        timevar="tstart",
        treatment_col=treatment_col,
        mask_func=strategy.treatment_prob_mask,
        return_details=True,
        weight_col=weight_col,
    )
    df["treatment_prob"] = res_treat["prob"]

    censor_cols_res = {}
    for col_ind, censor_col in enumerate(censor_cols):
        mask_func = None
        if censor_day0 is not None and not censor_day0[col_ind]:
            mask_func = day0_mask
        res = ipcwtm(
            data=df,
            censor_indicator=f"{censor_col}_tstart",
            family="binomial",
            link="logit",
            numerator=None,
            denominator=ipw_explanatory_formula[censor_col],
            id_col="id",
            timevar="tstart",
            treatment_col=treatment_col,
            mask_func=mask_func,
            weight_col=weight_col,
            formula_namespace=formula_namespace,
        )
        df[f"ipw.weights.{censor_col}"] = res["ipw.weights"]
        censor_cols_res[censor_col] = res

    df, W_details = strategy.convert_treatment_prob_to_ipcw(
        df, treatment_col, return_details=True
    )

    if censor_cols:
        df["ipw.weights"] = df[
            [f"ipw.weights.{censor_col}" for censor_col in censor_cols]
            + ["ipw.weights"]
        ].prod(axis=1)

    at_risk = W_details["at_risk"]
    dlogg_deta = W_details["dlogg_deta"]

    X = pd.DataFrame(
        0.0,
        index=df.index,
        columns=["artificial_censor" + item for item in res_treat["X"].columns],
    )
    X.loc[res_treat["X"].index, :] = res_treat["X"].to_numpy()
    dlogg_dgamma = dlogg_deta.to_numpy()[:, None] * X.to_numpy()
    df.loc[:, X.columns] = dlogg_dgamma
    dlogg_dgamma_sum = df.groupby("id")[X.columns].cumsum().to_numpy()
    dW_dgamma = -df.loc[:, "ipw.weights"].to_numpy()[:, None] * dlogg_dgamma_sum
    if weight_col is not None:
        dW_dgamma = dW_dgamma * df.loc[:, weight_col].to_numpy()[:, None]
    df.drop(columns=X.columns, inplace=True)

    score = res_treat["score"]
    hessian = res_treat["hessian"]
    score_by_id = res_treat["score_by_id"]

    if censor_cols:
        for censor_col in censor_cols:
            res_censor = censor_cols_res[censor_col]
            dlogg_deta_censor = res_censor["dlogg_deta"]
            X = pd.DataFrame(
                0.0,
                index=df.index,
                columns=[f"{censor_col}" + item for item in res_censor["X"].columns],
            )
            X.loc[res_censor["X"].index, :] = res_censor["X"].to_numpy()
            dlogg_dbeta_censor = dlogg_deta_censor.to_numpy()[:, None] * X.to_numpy()
            df.loc[:, X.columns] = dlogg_dbeta_censor
            dlogg_dbeta_censor_sum = df.groupby("id")[X.columns].cumsum().to_numpy()
            dW_dbeta_censor = (
                -df.loc[:, "ipw.weights"].to_numpy()[:, None] * dlogg_dbeta_censor_sum
            )
            if weight_col is not None:
                dW_dbeta_censor = (
                    dW_dbeta_censor * df.loc[:, weight_col].to_numpy()[:, None]
                )
            dW_dgamma = np.hstack([dW_dgamma, dW_dbeta_censor])
            score = np.concatenate([score, res_censor["score"]])
            hessian = np.block(
                [
                    [
                        hessian,
                        np.zeros((hessian.shape[0], res_censor["score"].shape[0])),
                    ],
                    [
                        np.zeros((res_censor["score"].shape[0], hessian.shape[1])),
                        res_censor["hessian"],
                    ],
                ]
            )
            score_by_id = pd.concat([score_by_id, res_censor["score_by_id"]], axis=1)

            df.drop(columns=X.columns, inplace=True)

    dW_dgamma = pd.DataFrame(dW_dgamma, index=df.index)

    at_risk_mask = at_risk.astype(bool)
    g_t_total = pd.Series(1.0, index=df.index)
    treat_prob = df["treatment_prob"]
    pos_mask = at_risk_mask & (dlogg_deta >= 0)
    neg_mask = at_risk_mask & (dlogg_deta < 0)
    g_t_total.loc[pos_mask] = treat_prob.loc[pos_mask]
    g_t_total.loc[neg_mask] = 1.0 - treat_prob.loc[neg_mask]
    for censor_col in censor_cols_res:
        g_t_total = g_t_total * censor_cols_res[censor_col]["g_t"]
    h_full = g_t_total.groupby(df["id"]).cumprod()
    h_prev = h_full.groupby(df["id"]).shift(1, fill_value=1.0)

    if not return_details:
        return df

    details = {
        "score_gamma": score,
        "score_by_id": score_by_id,
        "hessian_gamma": hessian,
        "dW_dgamma": dW_dgamma,
        "g_t": g_t_total,
        "h_prev": h_prev,
    }
    return df, details
