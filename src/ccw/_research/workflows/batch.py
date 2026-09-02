import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import json
import logging
import warnings
from contextlib import nullcontext
import pickle
import shutil

import itertools

from ccw import CCW, DataSpec
from ccw._bootstrap import subject_level_bootstrap
from ccw._research.tracking.run_logger import (
    RunLogger,
    run_ground_truth_simulation as _run_ground_truth_simulation,
)
from ccw._research.evaluation import run_evaluation


from ccw._research.configs.load_variables import load_experiment_settings
from ccw._research.configs.config_override import set_config_override_dir, read_yaml_text
from ccw._research.analysis import analysis_with_immortal_time_bias, landmark_analysis
from ccw._research.utils import create_datasets_in_df
from ccw._research.data_generation.core import BayesianNetwork, VariableTypes
from scipy import stats

import multiprocessing as mp
from logging import getLogger
import threading

# iptw関連
from patsy import EvalEnvironment
from statsmodels.api import GLM
from statsmodels.genmod import families
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


CHECKPOINT_DIRNAME = "checkpoints"
GROUND_TRUTH_CACHE_NAME = "ground_truth.pkl"
BATCH_PARAMS_NAME = "batch_params.json"
COMPLETION_MARKER_NAME = "simulation_complete.txt"
GROUND_TRUTH_INFO_NAME = "ground_truth_cache_info.json"


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Programmatic configuration for one deterministic batch experiment."""

    experiment: str
    sample_size: int = 1000
    ground_truth_sample_size: int = 10000
    n_runs: int = 10
    n_time: int = 40
    cutoff_time_of_intervention: int = 2
    cutoff_time_of_observation: int = 31
    cutoff_time_of_observation_display: int = 30
    ground_truth_seed: int = 42
    base_seed: int = 1000
    n_workers: int = 1
    ground_truth_workers: int = 1
    config_dir: str | Path | None = None
    ground_truth_cache_dir: str | Path | None = None
    save_ipw_weights: bool = False
    skip_ground_truth: bool = False
    skip_figures: bool = True
    run_evaluation: bool = True
    verbose: int = 1
    run_IPTW: bool = False
    bootstrap: bool = False
    bootstrap_B: int = 100
    bootstrap_seed: int = 2025
    bootstrap_conf: float = 0.95

    def validate(self) -> None:
        if self.sample_size < 1 or self.ground_truth_sample_size < 1:
            raise ValueError("sample sizes must be positive")
        if self.n_runs < 1 or self.n_time < 1:
            raise ValueError("n_runs and n_time must be positive")
        if self.n_workers < 1 or self.ground_truth_workers < 1:
            raise ValueError("worker counts must be positive")
        if self.bootstrap_B < 0:
            raise ValueError("bootstrap_B cannot be negative")
    def to_namespace(self, output_dir: str | Path) -> argparse.Namespace:
        self.validate()
        values = dict(vars(self))
        values["experiment"] = self.experiment
        values["output_dir"] = str(Path(output_dir))
        values["config_dir"] = str(self.config_dir) if self.config_dir is not None else None
        values["ground_truth_cache_dir"] = (
            str(self.ground_truth_cache_dir) if self.ground_truth_cache_dir is not None else None
        )
        values["bootstrap_method"] = "basic"
        values["scenario"] = None
        return argparse.Namespace(**values)


@dataclass(frozen=True)
class RunArtifacts:
    """Stable locations and in-memory results from a completed experiment."""

    output_dir: Path
    completion_marker: Path
    analysis_results_csv: Path
    summary_statistics_csv: Path
    result: dict[str, Any]


def _atomic_write_bytes(data: bytes, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def _atomic_pickle_dump(obj, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_checkpoint_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "artifacts", CHECKPOINT_DIRNAME)


def _get_run_checkpoint_path(checkpoint_dir: str, run_id: int) -> str:
    return os.path.join(checkpoint_dir, f"run_{int(run_id)}.pkl")


def _get_ground_truth_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "artifacts", GROUND_TRUTH_CACHE_NAME)


def _get_batch_params_path(output_dir: str) -> str:
    return os.path.join(output_dir, BATCH_PARAMS_NAME)


def _get_completion_marker_path(output_dir: str) -> str:
    return os.path.join(output_dir, COMPLETION_MARKER_NAME)


def _get_ground_truth_info_path(output_dir: str) -> str:
    return os.path.join(output_dir, GROUND_TRUTH_INFO_NAME)


def _compute_config_hash() -> str:
    import hashlib as _hashlib
    parts = []
    for name in ("scenario1.yaml", "scenario2.yaml", "scenario3.yaml"):
        text = read_yaml_text(name, "ccw._research.configs")
        parts.append(f"--- {name} ---\n")
        parts.append(text)
        parts.append("\n")
    digest = _hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()
    return digest


def _build_gt_cache_meta(args, commit_hash: str | None) -> dict:
    return {
        "experiment": args.experiment,
        "ground_truth_sample_size": args.ground_truth_sample_size,
        "n_time": args.n_time,
        "cutoff_time_of_intervention": args.cutoff_time_of_intervention,
        "cutoff_time_of_observation": args.cutoff_time_of_observation,
        "cutoff_time_of_observation_display": args.cutoff_time_of_observation_display,
        "ground_truth_seed": getattr(args, "ground_truth_seed", None),
        "ground_truth_workers": getattr(args, "ground_truth_workers", None),
        "run_IPTW": bool(getattr(args, "run_IPTW", False)),
        "config_hash": _compute_config_hash(),
        "git_commit_hash": commit_hash,
    }


def _gt_cache_matches(cached_meta: dict, expected_meta: dict, logger=None) -> bool:
    if not isinstance(cached_meta, dict):
        return False
    mismatches = []
    for key, expected_val in expected_meta.items():
        cached_val = cached_meta.get(key, None)
        if cached_val != expected_val:
            mismatches.append((key, cached_val, expected_val))
    if mismatches and logger is not None:
        preview = ", ".join([f"{k}({cv}!= {ev})" for k, cv, ev in mismatches[:3]])
        logger.warning(f"Ground Truthキャッシュ不一致: {preview}")
    return not mismatches


def _write_ground_truth_info(output_dir: str, info: dict, logger=None) -> None:
    path = _get_ground_truth_info_path(output_dir)
    try:
        data = json.dumps(info, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        _atomic_write_bytes(data, path)
        if logger is not None:
            logger.info(f"Ground Truth情報保存: {path}")
    except Exception as e:
        if logger is not None:
            logger.warning(f"Ground Truth情報保存に失敗: {path} ({e})")


def load_run_checkpoints(checkpoint_dir: str, logger=None) -> dict[int, dict]:
    if not os.path.isdir(checkpoint_dir):
        return {}
    results_by_run: dict[int, dict] = {}
    for name in sorted(os.listdir(checkpoint_dir)):
        if not (name.startswith("run_") and name.endswith(".pkl")):
            continue
        path = os.path.join(checkpoint_dir, name)
        try:
            res = _load_pickle(path)
        except Exception as e:
            if logger is not None:
                logger.warning(f"チェックポイント読み込み失敗: {path} ({e})")
            continue
        run_id = res.get("run_id")
        if run_id is None:
            if logger is not None:
                logger.warning(f"run_idが見つからないためスキップ: {path}")
            continue
        results_by_run[int(run_id)] = res
    return results_by_run


def save_run_checkpoint(checkpoint_dir: str, run_id: int, result: dict, logger=None) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _get_run_checkpoint_path(checkpoint_dir, run_id)
    try:
        _atomic_pickle_dump(result, path)
    except Exception as e:
        if logger is not None:
            logger.warning(f"チェックポイント保存失敗: {path} ({e})")


def load_ground_truth_cache(output_dir: str, logger=None):
    path = _get_ground_truth_cache_path(output_dir)
    if not os.path.isfile(path):
        return None
    try:
        payload = _load_pickle(path)
        if isinstance(payload, dict) and "ground_truth_result" in payload and "simulated_or" in payload:
            return payload
        if logger is not None:
            logger.warning(f"Ground Truthキャッシュ形式が不正です: {path}")
        return None
    except Exception as e:
        if logger is not None:
            logger.warning(f"Ground Truthキャッシュ読み込み失敗: {path} ({e})")
        return None


def save_ground_truth_cache(output_dir: str, ground_truth_result, simulated_or, logger=None, meta: dict | None = None) -> None:
    os.makedirs(os.path.join(output_dir, "artifacts"), exist_ok=True)
    payload = {
        "ground_truth_result": ground_truth_result,
        "simulated_or": simulated_or,
        "meta": meta,
    }
    path = _get_ground_truth_cache_path(output_dir)
    try:
        _atomic_pickle_dump(payload, path)
        if logger is not None:
            logger.info(f"Ground Truthキャッシュ保存: {path}")
    except Exception as e:
        if logger is not None:
            logger.warning(f"Ground Truthキャッシュ保存失敗: {path} ({e})")


def save_batch_params(output_dir: str, args, commit_hash=None, logger=None) -> str:
    params = {
        k: v for k, v in vars(args).items()
        if not str(k).startswith("_") and isinstance(v, (str, int, float, bool, list))
    }
    if commit_hash:
        params["git_commit_hash"] = commit_hash
    data = json.dumps(params, indent=2).encode("utf-8")
    path = _get_batch_params_path(output_dir)
    try:
        _atomic_write_bytes(data, path)
        if logger is not None:
            logger.info(f"パラメータ保存: {path}")
    except Exception as e:
        if logger is not None:
            logger.warning(f"パラメータ保存失敗: {path} ({e})")
    return path


def logger_thread(q: mp.Queue, parent_logger_name: str, log_level=logging.INFO):
    # Use the logger BY NAME so it uses the parent’s existing handlers/formatters
    while True:
        record = q.get()
        if record is None:
            break
        assert record.name.startswith(
            parent_logger_name), f"Unexpected logger name: {record.name}"
        lg = getLogger(record.name)
        lg.setLevel(log_level)
        lg.handle(record)


def calculate_comprehensive_statistics(df_joined, scenario_vars, configs, grace_period, observation_period, experiment, logger=None, column_maps=None):
    """包括的な統計の計算（自動化版・ログ統合）"""
    if logger is None:
        # フォールバック用のシンプルなロガー
        logger = logging.getLogger(__name__)

    stats = {}

    # scenario_varsから変数情報を自動取得
    treatment_var = configs["treatment_var"]
    outcome_var = configs["outcome_var"]
    censor_vars = configs.get("censor_vars", [])

    # scenario_varsはVariable objectのリストなので、それに基づいて分類
    baseline_vars = {}
    time_varying_vars = {}

    for var in scenario_vars:
        var_name = var.id.name
        var_type = var.value_type  # VariableTypes enumオブジェクト
        var_info = {
            'type': var_type,
            'time_varying': var.is_time_varying,
            'col_name': column_maps.get(var_name, var_name) if column_maps else var_name
        }
        if var.id == treatment_var or var.id == outcome_var or var.id in censor_vars:
            continue
        if var.is_time_varying:
            time_varying_vars[var_name] = var_info
        else:
            baseline_vars[var_name] = var_info

    logger.debug(f"治療変数: {treatment_var}")
    logger.debug(f"アウトカム変数: {outcome_var}")
    logger.debug(f"ベースライン変数: {list(baseline_vars.keys())}")
    logger.debug(f"時間依存変数: {list(time_varying_vars.keys())}")

    # 変数名をそのまま使用（正規化不要）
    treatment_var_name = treatment_var.name
    outcome_var_name = outcome_var.name
    censor_var_names = [var.name for var in censor_vars]

    ##### 1. Interevention/Outcome/censor_vars の発生率計算 #####

    # grace period内のevent rate
    df_gp = df_joined[df_joined['time'] <= grace_period]
    _gp_event_res = df_gp.groupby(
        'id')[[treatment_var_name, outcome_var_name] + censor_var_names].max()

    # observation period 内の event rate
    df_op = df_joined[df_joined['time'] <= observation_period]
    _op_event_res = df_op.groupby(
        'id')[[treatment_var_name, outcome_var_name] + censor_var_names].max()

    # 患者レベル集約を用いた層別（Grace内介入の有無で層別）
    per_id = pd.DataFrame(index=_gp_event_res.index)
    per_id['treatment_in_grace'] = _gp_event_res[treatment_var_name].fillna(
        np.nan).astype(int)
    per_id['outcome_in_grace'] = _gp_event_res[outcome_var_name].fillna(
        np.nan).astype(int)
    per_id['outcome_in_observation'] = (
        _op_event_res.reindex(per_id.index)[
            outcome_var_name].fillna(np.nan).astype(int)
    )

    assert per_id.isna().sum().sum() == 0, "per_idにNaNが含まれています"

    with_mask = per_id['treatment_in_grace'] == 1
    without_mask = ~with_mask
    # 観測期間内アウトカム率（介入の有無で層別）
    outcome_op_rate_with_intervention = per_id.loc[with_mask, 'outcome_in_observation'].mean(
    )
    outcome_op_rate_without_intervention = per_id.loc[without_mask, 'outcome_in_observation'].mean(
    )
    # Grace期間内アウトカム率（介入の有無で層別）
    outcome_gp_rate_with_intervention = per_id.loc[with_mask, 'outcome_in_grace'].mean(
    )
    outcome_gp_rate_without_intervention = per_id.loc[without_mask, 'outcome_in_grace'].mean(
    )

    stats.update({
        'intervention_grace_rate': _gp_event_res[treatment_var_name].mean(),
        f'intervention_{observation_period}day_rate': _op_event_res[treatment_var_name].mean(),
        'outcome_grace_rate': _gp_event_res[outcome_var_name].mean(),
        f'outcome_{observation_period}day_rate': _op_event_res[outcome_var_name].mean(),
        f'outcome_{observation_period}day_with_intervention_rate': outcome_op_rate_with_intervention,
        f'outcome_{observation_period}day_without_intervention_rate': outcome_op_rate_without_intervention,
        'outcome_gp_with_intervention_rate': outcome_gp_rate_with_intervention,
        'outcome_gp_without_intervention_rate': outcome_gp_rate_without_intervention,
    })
    for censor_var_name in censor_var_names:
        per_id[f"{censor_var_name}_in_grace"] = _gp_event_res[censor_var_name].fillna(
            np.nan).astype(int)
        censor_mask = per_id[f"{censor_var_name}_in_grace"] == 1
        outcome_gp_rate_with_censor = per_id.loc[censor_mask, 'outcome_in_grace'].mean(
        )
        outcome_gp_rate_without_censor = per_id.loc[~censor_mask, 'outcome_in_grace'].mean(
        )
        stats.update({
            f'censor_{censor_var_name}_grace_rate': _gp_event_res[censor_var_name].mean(),
            f'censor_{censor_var_name}_{observation_period}day_rate': _op_event_res[censor_var_name].mean(),
            f'outcome_gp_with_{censor_var_name}_rate': outcome_gp_rate_with_censor,
            f'outcome_gp_without_{censor_var_name}_rate': outcome_gp_rate_without_censor,
        })

    #####  2. ベースライン特徴量分析 #####
    baseline_data = df_joined[df_joined['time'] == 0].copy()
    baseline_data = baseline_data.join(
        per_id[['treatment_in_grace']], on='id'
    )
    # use per_id to filter baesline_data with those with intervention in gp or those without
    baseline_with_intervention_gp = baseline_data.loc[baseline_data['treatment_in_grace'] == 1]
    baseline_without_intervention_gp = baseline_data.loc[baseline_data['treatment_in_grace'] == 0]

    for var_name, var_info in baseline_vars.items():
        # describe stats for baseline_data, baseline_with_intervention_gp, baseline_without_intervention_gp

        var_type = var_info.get('type')
        col_name = var_info['col_name']
        logger.debug(f"DEBUG: 処理中のベースライン変数: {var_name} (type: {var_type})")

        def _add_prefix_to_dict(d, prefix):
            return {f"{prefix}_{k}": v for k, v in d.items()}
        # VariableTypesのenumを使用して型を判定
        if var_type in [VariableTypes.CONTINUOUS, VariableTypes.ORDERED_CONTINUOUS]:
            # 連続変数の場合
            desc = baseline_data[col_name].describe().to_dict()
            desc_with = baseline_with_intervention_gp[col_name].describe(
            ).to_dict()
            desc_without = baseline_without_intervention_gp[col_name].describe(
            ).to_dict()
            # record
            stats.update(_add_prefix_to_dict(desc, f"baseline_{col_name}"))
            stats.update(_add_prefix_to_dict(
                desc_with, f"baseline_{col_name}_with_intervention_gp"))
            stats.update(_add_prefix_to_dict(
                desc_without, f"baseline_{col_name}_without_intervention_gp"))

        elif var_type in [VariableTypes.CATEGORICAL, VariableTypes.ORDERED_CATEGORICAL, VariableTypes.BINARY]:
            # just do value_counts with normalize=True
            value_counts = baseline_data[col_name].value_counts(
                normalize=True).to_dict()
            value_counts_with = baseline_with_intervention_gp[col_name].value_counts(
                normalize=True).to_dict()
            value_counts_without = baseline_without_intervention_gp[col_name].value_counts(
                normalize=True).to_dict()
            # record with prefix
            stats.update(_add_prefix_to_dict(
                value_counts, f"baseline_{col_name}"))
            stats.update(_add_prefix_to_dict(value_counts_with,
                         f"baseline_{col_name}_with_intervention_gp"))
            stats.update(_add_prefix_to_dict(value_counts_without,
                         f"baseline_{col_name}_without_intervention_gp"))

    ##### 3. 時間依存特徴量分析 #####
    for _var_name, var_info in time_varying_vars.items():
        # just do the same analysis with three timepoints: time=0, time=grace_period, time=observation_period
        var_type = var_info.get('type')
        col_name = var_info['col_name']
        if var_type in [VariableTypes.CONTINUOUS, VariableTypes.ORDERED_CONTINUOUS]:
            for time_point in [0, grace_period, observation_period]:
                tv_data = df_joined[df_joined['time'] == time_point].copy()
                tv_data = tv_data.join(
                    per_id[['treatment_in_grace']], on='id'
                )
                tv_with_intervention_gp = tv_data.loc[tv_data['treatment_in_grace'] == 1]
                tv_without_intervention_gp = tv_data.loc[tv_data['treatment_in_grace'] == 0]

                desc = tv_data[col_name].describe().to_dict()
                desc_with = tv_with_intervention_gp[col_name].describe(
                ).to_dict()
                desc_without = tv_without_intervention_gp[col_name].describe(
                ).to_dict()

                stats.update(_add_prefix_to_dict(
                    desc, f"time{time_point}_{col_name}"))
                stats.update(_add_prefix_to_dict(
                    desc_with, f"time{time_point}_{col_name}_with_intervention_gp"))
                stats.update(_add_prefix_to_dict(
                    desc_without, f"time{time_point}_{col_name}_without_intervention_gp"))

    logger.debug(
        f"DEBUG: 統計計算完了。総統計数: {len(stats)}, NaN値の数: {sum(1 for v in stats.values() if pd.isna(v))}")
    return stats


def check_variable_correlation(
    sample,
    var1_id,
    var2_id,
    var1_type,
    var2_type,
    var1_is_time_varying,
    var2_is_time_varying,
    time_point=0
):
    """
    シミュレーションサンプルから2つの変数の相関をチェック
    """
    # データを抽出
    var1_values = []
    var2_values = []

    for individual_sample in sample:
        # 変数1の値を取得
        if var1_id in individual_sample:
            time_key = time_point if var1_is_time_varying else 0
            if time_key in individual_sample[var1_id]:
                var1_val = individual_sample[var1_id][time_key]
            else:
                continue
        else:
            continue

        # 変数2の値を取得
        if var2_id in individual_sample:
            time_key = time_point if var2_is_time_varying else 0
            if time_key in individual_sample[var2_id]:
                var2_val = individual_sample[var2_id][time_key]
            else:
                continue
        else:
            continue

        # 二値変数を0/1に変換
        if var1_type in [VariableTypes.BINARY, VariableTypes.EVENT_BINARY]:
            var1_val = 1 if var1_val in [1, True] else 0
        if var2_type in [VariableTypes.BINARY, VariableTypes.EVENT_BINARY]:
            var2_val = 1 if var2_val in [1, True] else 0

        var1_values.append(var1_val)
        var2_values.append(var2_val)

    # データが不足している場合
    if len(var1_values) < 2:
        return {
            'n_pairs': len(var1_values),
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_r': np.nan,
            'spearman_p': np.nan
        }

    # 相関計算
    results = {
        'n_pairs': len(var1_values),
        'pearson_r': np.nan,
        'pearson_p': np.nan,
        'spearman_r': np.nan,
        'spearman_p': np.nan
    }

    if len(np.unique(var1_values)) > 1 and len(np.unique(var2_values)) > 1:
        # 連続変数同士の場合はPearson相関
        if (var1_type in [VariableTypes.CONTINUOUS, VariableTypes.ORDERED_CONTINUOUS] and
                var2_type in [VariableTypes.CONTINUOUS, VariableTypes.ORDERED_CONTINUOUS]):
            pearson_r, pearson_p = stats.pearsonr(var1_values, var2_values)
            results['pearson_r'] = pearson_r
            results['pearson_p'] = pearson_p

        # 常にSpearman相関も計算
        spearman_r, spearman_p = stats.spearmanr(var1_values, var2_values)
        results['spearman_r'] = spearman_r
        results['spearman_p'] = spearman_p

    return results


def _add_outcome_results(result, results_dict, prefix=""):
    for strat in ["intervention", "control"]:
        result[f'{prefix}outcome_rate_{strat}'] = results_dict.get(
            f'outcome_rate_{strat}', np.nan)

    # risk difference, risk_ratio
    result[f"{prefix}rd"] = results_dict.get(
        'outcome_rate_intervention', np.nan) - results_dict.get('outcome_rate_control', np.nan)
    if results_dict.get('outcome_rate_control', np.nan) > 0:
        result[f"{prefix}rr"] = results_dict.get(
            'outcome_rate_intervention', np.nan) / results_dict.get('outcome_rate_control', np.nan)
    else:
        result[f"{prefix}rr"] = np.nan
    return result


def _as_incident_censoring(
    data: pd.DataFrame,
    censoring_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Convert simulated absorbing censoring states to incident indicators."""
    frame = data.copy()
    for column in censoring_columns:
        first_event = (
            frame.loc[frame[column] == 1]
            .groupby("id", sort=False)["time"]
            .min()
        )
        frame[column] = (
            frame["time"] == frame["id"].map(first_event)
        ).astype(int)
    return frame


# to be called from run_single_simulation_analysis
def analyze_methods_on_df(
    df: pd.DataFrame,
    configs,
    intervention_strategies,
    column_maps,
    baseline_cols: tuple[str, ...],
    cutoff_time_of_intervention: int,
    cutoff_time_of_observation: int,
    cutoff_time_of_observation_display: int,
    seed: int,
    logger,
    verbose: int,
    run_IPTW: bool = False,
    logger_=None,
    external_weight_col: str | None = None,
    return_ccw_weights_df: bool = False,
) -> dict:
    """
    単一データフレーム df に対して CCW/ITB/Landmark を実行し、主要結果を返す。
    run_IPTW=True の場合は各回でIPTWを再学習する。
    CCW is fitted through the public :class:`ccw.CCW` API. The research-only
    ITB and landmark comparators remain in this workflow.
    """
    work_df = df
    weight_col = None
    ccw_weights_df = None

    if run_IPTW:
        work_df = df.copy()
        iptw_expl = configs["iptw_explanatory_formula"](column_maps)
        df_gp = work_df[work_df['time'] <= cutoff_time_of_intervention]
        id_treated = df_gp.groupby('id')[configs["treatment_var"].name].max()
        work_df['treatment_in_grace'] = work_df['id'].map(
            id_treated).fillna(np.nan)

        eval_env = EvalEnvironment.capture(0)
        fam = families.Binomial(link=families.links.Logit())
        iptw_freq_weights = None
        if external_weight_col is not None:
            iptw_freq_weights = work_df[external_weight_col].to_numpy()
        model = GLM.from_formula(
            "treatment_in_grace ~ " + iptw_expl, data=work_df,
            family=fam, eval_env=eval_env, freq_weights=iptw_freq_weights
        ).fit()
        work_df['treatment_prob_gp'] = model.predict(work_df)
        work_df['iptw_weight'] = np.where(
            work_df['treatment_in_grace'] == 1,
            1 / work_df['treatment_prob_gp'],
            1 / (1 - work_df['treatment_prob_gp'])
        )
        weight_col = 'iptw_weight'

    if external_weight_col is not None:
        if work_df is df:
            work_df = df.copy()
        if weight_col is None:
            weight_col = external_weight_col
        else:
            work_df['analysis_weight_input'] = work_df[weight_col] * work_df[external_weight_col]
            weight_col = 'analysis_weight_input'

    censor_cols = tuple(
        variable.name for variable in (configs.get("censor_vars") or ())
    )
    ccw_input = _as_incident_censoring(work_df, censor_cols)

    censor_day0 = configs.get("censor_day0") or [False] * len(censor_cols)
    ccw_model = CCW(
        spec=DataSpec(
            id="id",
            time="time",
            treatment=configs["treatment_var"].name,
            outcome=configs["outcome_var"].name,
            censoring=censor_cols,
            baseline=baseline_cols,
            time_varying=tuple(
                variable.name
                for variable in (configs.get("time_varying_vars") or ())
            ),
            sample_weight=weight_col,
        ),
        strategies=intervention_strategies,
        weight_models=configs["ipw_explanatory_formula"](column_maps),
        followup_end=cutoff_time_of_observation,
        estimate_at=cutoff_time_of_observation_display,
        random_state=seed,
        censoring_model=configs["censoring_model"],
        censoring_at_baseline=dict(zip(censor_cols, censor_day0, strict=True)),
        verbose=verbose,
    )
    logger.info("Fitting CCW analysis through the public ccw.CCW.fit API")
    fitted_ccw = ccw_model.fit(ccw_input)
    ccw_results_dict = fitted_ccw.estimates
    ccw_or_plr = np.nan
    ccw_or_km = fitted_ccw.odds_ratio
    if return_ccw_weights_df:
        ccw_weights_df = fitted_ccw.weights.rename(columns={"time": "t"})
        
    itb_results_dict, itb_or_km = analysis_with_immortal_time_bias(
        work_df,
        configs=configs,
        cutoff_time_of_intervention=cutoff_time_of_intervention,
        cutoff_time_of_observation=cutoff_time_of_observation,
        cutoff_time_of_observation_display=cutoff_time_of_observation_display,
        logger_=logger,
        verbose=verbose,
        weight_col=weight_col,
    )
    itb_or_plr = itb_or_km

    landmark_results_dict, landmark_or_km = landmark_analysis(
        work_df,
        configs=configs,
        cutoff_time_of_intervention=cutoff_time_of_intervention,
        cutoff_time_of_observation=cutoff_time_of_observation,
        cutoff_time_of_observation_display=cutoff_time_of_observation_display,
        logger_=logger,
        verbose=verbose,
        weight_col=weight_col,
    )
    landmark_or_plr = landmark_or_km

    if logger_ is not None:
        logger_.info(
            f"  CCW - PLR OR: {ccw_or_plr:.4f}, KM OR: {ccw_or_km:.4f}")
        logger_.info(
            f"  ITB - PLR OR: {itb_or_plr:.4f}, KM OR: {itb_or_km:.4f}")
        logger_.info(
            f"  Landmark - PLR OR: {landmark_or_plr:.4f}, KM OR: {landmark_or_km:.4f}")

    result = {
        'ccw_results_dict': ccw_results_dict,
        'ccw_or_plr': ccw_or_plr,
        'ccw_or_km': ccw_or_km,
        'itb_results_dict': itb_results_dict,
        'itb_or_plr': itb_or_plr,
        'itb_or_km': itb_or_km,
        'landmark_results_dict': landmark_results_dict,
        'landmark_or_plr': landmark_or_plr,
        'landmark_or_km': landmark_or_km,
    }
    result = _add_outcome_results(result, ccw_results_dict, prefix="ccw_")
    result = _add_outcome_results(result, itb_results_dict, prefix="itb_")
    result = _add_outcome_results(
        result, landmark_results_dict, prefix="landmark_")
    if return_ccw_weights_df:
        result['ccw_weights_df'] = ccw_weights_df
    return result


def run_single_simulation_analysis(args_dict, q, parent_logger_name: str | None = None):
    """単一のシミュレーション分析を実行する関数（ログ統合・複数分析手法対応）"""
    warnings.filterwarnings(
        "ignore",
        message="overflow encountered in exp",
        category=RuntimeWarning,
        module=r"statsmodels\.genmod\.families\.links",
    )
    warnings.filterwarnings(
        "ignore",
        message="All-NaN slice encountered",
        category=RuntimeWarning,
        module=r"numpy\.lib\._nanfunctions_impl",
    )
    warnings.filterwarnings("ignore", category=PerfectSeparationWarning)

    # プロセス固有のロガーを設定
    from logging.handlers import QueueHandler
    logger_name = parent_logger_name + \
        f".run_{args_dict.get('run_id', 'unknown')}" if parent_logger_name else f"CCWSimulation.run_{args_dict.get('run_id', 'unknown')}"
    logger = logging.getLogger(logger_name)
    if q is not None:
        logger.handlers.clear()
        logger.addHandler(QueueHandler(q))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f"Run {args_dict.get('run_id', 'unknown')}: 分析開始")

    try:
        # 引数を展開
        experiment = args_dict['experiment']
        sample_size = args_dict['sample_size']
        n_time = args_dict['n_time']
        cutoff_time_of_intervention = args_dict['cutoff_time_of_intervention']
        cutoff_time_of_observation = args_dict['cutoff_time_of_observation']
        cutoff_time_of_observation_display = args_dict['cutoff_time_of_observation_display']
        seed = args_dict['seed']
        run_id = args_dict['run_id']
        run_IPTW = args_dict.get('run_IPTW', False)
        save_ipw_weights = bool(args_dict.get('save_ipw_weights', False))
        output_dir = args_dict.get('output_dir')
        config_dir = args_dict.get('config_dir')

        # Ensure per-run scenario override is applied inside each worker process.
        # With multiprocessing spawn, module-level globals are not inherited.
        set_config_override_dir(config_dir)

        bootstrap_in_run: bool = bool(args_dict.get('bootstrap_in_run', False))
        bootstrap_B: int = int(args_dict.get('bootstrap_B', 0) or 0)
        bootstrap_seed: int = int(args_dict.get('bootstrap_seed', 2025))
        bootstrap_conf: float = float(args_dict.get('bootstrap_conf', 0.95))
        bootstrap_method = "basic"

        logger.debug(f"Run {run_id}: 分析開始 - 実験: {experiment}, シード: {seed}")

        # args_dictをNamespaceオブジェクトに変換（strategy_creator用）
        args_namespace = argparse.Namespace(**args_dict)

        # 実験設定をロード
        scenario_params, scenario_vars, configs = load_experiment_settings(
            experiment)

        intervention_strategies = {
            key: item(args_namespace) for key, item in configs["strategy_creator"].items()
        }

        logger.debug(f"Run {run_id}: 実験設定ロード完了")

        # データ生成
        censor_vars = configs.get('censor_vars', None)
        bn = BayesianNetwork(scenario_vars, n_time + 1)
        sample = bn.sample(sample_size=sample_size, seed=seed)

        for var1, var2 in itertools.combinations(scenario_vars, 2):
            corr_result = check_variable_correlation(
                sample=sample,
                var1_id=var1.id,
                var2_id=var2.id,
                var1_type=var1.value_type,
                var2_type=var2.value_type,
                var1_is_time_varying=var1.is_time_varying,
                var2_is_time_varying=var2.is_time_varying,
                time_point=0
            )

            logger.debug(f"Run {run_id}: 相関チェック ({var1.id.name} vs {var2.id.name}): "
                         f"Pearson r={corr_result['pearson_r']:.3f}, "
                         f"Spearman r={corr_result['spearman_r']:.3f}, "
                         f"n={corr_result['n_pairs']}")

        df_baseline, df_time_varying, df_intervention_outcome, df_joined = create_datasets_in_df(
            bn, sample,
            treatment_var=configs["treatment_var"],
            outcome_var=configs["outcome_var"],
            cut_data_after_outcome=configs["cut_data_after_outcome"],
            cutoff_time_of_observation=cutoff_time_of_observation
        )

        logger.debug(f"Run {run_id}: データ生成完了")

        # 前処理
        df_joined, column_maps = configs["preprocess_pipeline"](df_joined)
        baseline_cols = tuple(
            column_maps.get(variable.id.name, variable.id.name)
            for variable in scenario_vars
            if not variable.is_time_varying
        )

        # 詳細統計の計算（自動化版）
        detailed_stats = calculate_comprehensive_statistics(
            df_joined=df_joined,
            scenario_vars=scenario_vars,
            configs=configs,
            grace_period=cutoff_time_of_intervention,
            observation_period=cutoff_time_of_observation_display,
            experiment=experiment,
            logger=logger,
            column_maps=column_maps
        )

        result = detailed_stats.copy()

        logger.debug(f"Run {run_id}: 統計計算完了")

        # 複数の分析手法を実行
        analysis_weight_col = None
        ##### 1. full data で分析 #####
        analysis_out = analyze_methods_on_df(
            df=df_joined,
            configs=configs,
            intervention_strategies=intervention_strategies,
            column_maps=column_maps,
            baseline_cols=baseline_cols,
            cutoff_time_of_intervention=cutoff_time_of_intervention,
            cutoff_time_of_observation=cutoff_time_of_observation,
            cutoff_time_of_observation_display=cutoff_time_of_observation_display,
            seed=seed,
            logger=logger,
            verbose=args_dict['verbose'],
            run_IPTW=run_IPTW,
            logger_=logger,
            return_ccw_weights_df=save_ipw_weights,
        )
        
        ccw_weights_df = analysis_out.pop('ccw_weights_df', None)
        if save_ipw_weights and ccw_weights_df is not None and output_dir:
            weights_dir = os.path.join(output_dir, "artifacts", "ipw_weights")
            os.makedirs(weights_dir, exist_ok=True)
            for arm, arm_df in ccw_weights_df.groupby("arm", sort=False):
                if arm == "intervention":
                    safe_arm = "treated"
                elif arm == "control":
                    safe_arm = "control"
                else:
                    safe_arm = str(arm)
                weights_path = os.path.join(
                    weights_dir, f"ipw_weights_{safe_arm}_run_{int(run_id)}.csv"
                )
                arm_df.to_csv(weights_path, index=False)
                logger.info(f"Run {run_id}: ipw weights saved: {weights_path}")

        ccw_results_dict = analysis_out.pop('ccw_results_dict')
        itb_results_dict = analysis_out.pop('itb_results_dict')
        landmark_results_dict = analysis_out.pop('landmark_results_dict')

        # 詳細統計を追加
        result.update(analysis_out)

        # 各戦略のアウトカム率を追加（手法別）
        for strategy in intervention_strategies:
            # CCW
            result[f'ccw_outcome_rate_{strategy}'] = ccw_results_dict.get(
                f'outcome_rate_{strategy}', np.nan)
            # ITB
            result[f'itb_outcome_rate_{strategy}'] = itb_results_dict.get(
                f'outcome_rate_{strategy}', np.nan)
            # Landmark
            result[f'landmark_outcome_rate_{strategy}'] = landmark_results_dict.get(
                f'outcome_rate_{strategy}', np.nan)

        # その他の分析結果を追加（手法別プレフィックス付き）
        # CCW結果
        for key, value in ccw_results_dict.items():
            if key not in [f'outcome_rate_{s}' for s in intervention_strategies]:
                result[f'ccw_{key}'] = value

        # ITB結果
        for key, value in itb_results_dict.items():
            if key not in [f'outcome_rate_{s}' for s in intervention_strategies]:
                result[f'itb_{key}'] = value

        # Landmark結果
        for key, value in landmark_results_dict.items():
            if key not in [f'outcome_rate_{s}' for s in intervention_strategies]:
                result[f'landmark_{key}'] = value

        # full data の推定値
        theta_hat_ctrl = float(result.get("ccw_outcome_rate_control", np.nan))
        theta_hat_intv = float(result.get("ccw_outcome_rate_intervention", np.nan))
        se_hat_ctrl = float(result.get("ccw_outcome_rate_control_std", np.nan))
        se_hat_intv = float(result.get("ccw_outcome_rate_intervention_std", np.nan))
        se_hat_rd = float(result.get("ccw_rd_std", np.nan))
        se_hat_log_rr = float(result.get("ccw_log_rr_std", np.nan))
        
        ##### 2. bootstrap #####
        if bootstrap_in_run and bootstrap_B > 0:
            logger.info(
                f"Run {run_id}: bootstrap開始 (method={bootstrap_method}, B={bootstrap_B}, conf={bootstrap_conf:.2f})")

            def _bootstrap_statistic(df_bs: pd.DataFrame) -> np.ndarray:
                try:
                    df_bs, _ = configs["preprocess_pipeline"](df_bs, bootstrap=True)
                    df_bs = df_bs.reset_index(drop=True)

                    out = analyze_methods_on_df(
                        df=df_bs,
                        configs=configs,
                        intervention_strategies=intervention_strategies,
                        column_maps=column_maps,
                        baseline_cols=baseline_cols,
                        cutoff_time_of_intervention=cutoff_time_of_intervention,
                        cutoff_time_of_observation=cutoff_time_of_observation,
                        cutoff_time_of_observation_display=cutoff_time_of_observation_display,
                        seed=seed,
                        logger=logger,
                        verbose=0,
                        run_IPTW=run_IPTW,
                    )
                    return np.array([
                        out['ccw_or_plr'], out['ccw_or_km'],
                        out['itb_or_plr'], out['itb_or_km'],
                        out['landmark_or_plr'], out['landmark_or_km'],
                        out['ccw_rd'], out['itb_rd'], out['landmark_rd'],
                        out['ccw_rr'], out['itb_rr'], out['landmark_rr'],
                        out['ccw_results_dict']['outcome_rate_control'], out['ccw_results_dict']['outcome_rate_intervention'],
                        out['ccw_results_dict'].get('outcome_rate_control_std', np.nan), out['ccw_results_dict'].get('outcome_rate_intervention_std', np.nan),
                        out['ccw_results_dict'].get('rd_std', np.nan), out['ccw_results_dict'].get('log_rr_std', np.nan),  
                    ], dtype=float)
                except Exception:
                    logger.error(
                        f"Run {run_id}: ブートストラップサンプルの分析中にエラーが発生しました", exc_info=True)
                    return np.array([np.nan]*18, dtype=float)

            bootstrap_distribution = subject_level_bootstrap(
                df_joined,
                id_col="id",
                n_resamples=bootstrap_B,
                seed=bootstrap_seed + int(run_id),
                statistic=_bootstrap_statistic,
                confidence_level=bootstrap_conf,
            )
            bs_dist = bootstrap_distribution.values
            bootstrap_method_label = "basic(naomit)"

            theta_hat_vec = np.array([
                result['ccw_or_plr'], result['ccw_or_km'],
                result['itb_or_plr'], result['itb_or_km'],
                result['landmark_or_plr'], result['landmark_or_km'],
                result['ccw_rd'], result['itb_rd'], result['landmark_rd'],
                result['ccw_rr'], result['itb_rr'], result['landmark_rr'],
                result['ccw_outcome_rate_control'], result['ccw_outcome_rate_intervention'],
                result.get('ccw_outcome_rate_control_std', np.nan),
                result.get('ccw_outcome_rate_intervention_std', np.nan),
                result.get('ccw_rd_std', np.nan),
                result.get('ccw_log_rr_std', np.nan),
            ], dtype=float)

            lo, hi = bootstrap_distribution.basic_interval(
                theta_hat_vec,
                confidence_level=bootstrap_conf,
            )
            valid_B_per_stat = bootstrap_distribution.valid_per_statistic
            valid_B_all_stats = bootstrap_distribution.valid_all_statistics
            
            bs_results = pd.DataFrame(bs_dist[:, :], columns=[
                'ccw_or_plr', 'ccw_or_km',
                'itb_or_plr', 'itb_or_km',
                'landmark_or_plr', 'landmark_or_km',
                'ccw_rd', 'itb_rd', 'landmark_rd',
                'ccw_rr', 'itb_rr', 'landmark_rr',
                'ccw_outcome_rate_control', 'ccw_outcome_rate_intervention',
                'ccw_outcome_rate_control_std', 'ccw_outcome_rate_intervention_std',
                'ccw_rd_std', 'ccw_log_rr_std',
            ])
            bs_results['run_id'] = run_id
            bs_results['iteration'] = np.arange(bootstrap_B, dtype=int)
            bs_results['bootstrap_method'] = bootstrap_method_label
            bs_results_path = None
            if output_dir:
                bs_dir = os.path.join(output_dir, "artifacts", "bootstrap_results")
                os.makedirs(bs_dir, exist_ok=True)
                bs_results_path = os.path.join(bs_dir, f"bootstrap_results_run_{int(run_id)}.csv")
                bs_results.to_csv(bs_results_path, index=False)
                logger.info(f"Run {run_id}: bootstrap results saved: {bs_results_path}")

            result.update({
                "bootstrap_in_run": True,
                "bootstrap_B": bootstrap_B,
                "bootstrap_conf": bootstrap_conf,
                "bootstrap_method": bootstrap_method_label,
                "bootstrap_valid_B_all_stats": valid_B_all_stats,
                # OR
                "ccw_or_plr_ci_lower": float(lo[0]),  "ccw_or_plr_ci_upper": float(hi[0]),
                "ccw_or_km_ci_lower":  float(lo[1]),  "ccw_or_km_ci_upper":  float(hi[1]),
                "itb_or_plr_ci_lower": float(lo[2]),  "itb_or_plr_ci_upper": float(hi[2]),
                "itb_or_km_ci_lower":  float(lo[3]),  "itb_or_km_ci_upper":  float(hi[3]),
                "landmark_or_plr_ci_lower": float(lo[4]), "landmark_or_plr_ci_upper": float(hi[4]),
                "landmark_or_km_ci_lower":  float(lo[5]), "landmark_or_km_ci_upper":  float(hi[5]),
                # RD
                "ccw_rd_ci_lower": float(lo[6]), "ccw_rd_ci_upper": float(hi[6]),
                "itb_rd_ci_lower": float(lo[7]), "itb_rd_ci_upper": float(hi[7]),
                "landmark_rd_ci_lower": float(lo[8]), "landmark_rd_ci_upper": float(hi[8]),
                # RR
                "ccw_rr_ci_lower": float(lo[9]),  "ccw_rr_ci_upper": float(hi[9]),
                "itb_rr_ci_lower": float(lo[10]), "itb_rr_ci_upper": float(hi[10]),
                "landmark_rr_ci_lower": float(lo[11]), "landmark_rr_ci_upper": float(hi[11]),
                "ccw_outcome_rate_control_ci_lower": float(lo[12]),  "ccw_outcome_rate_control_ci_upper": float(hi[12]),
                "ccw_outcome_rate_intervention_ci_lower": float(lo[13]),  "ccw_outcome_rate_intervention_ci_upper": float(hi[13]),
                # 列ごとの有効反復数も保存
                "bootstrap_valid_B_ccw_or_plr": int(valid_B_per_stat[0]),
                "bootstrap_valid_B_ccw_or_km": int(valid_B_per_stat[1]),
                "bootstrap_valid_B_itb_or_plr": int(valid_B_per_stat[2]),
                "bootstrap_valid_B_itb_or_km": int(valid_B_per_stat[3]),
                "bootstrap_valid_B_landmark_or_plr": int(valid_B_per_stat[4]),
                "bootstrap_valid_B_landmark_or_km": int(valid_B_per_stat[5]),
                "bootstrap_valid_B_ccw_rd": int(valid_B_per_stat[6]),
                "bootstrap_valid_B_itb_rd": int(valid_B_per_stat[7]),
                "bootstrap_valid_B_landmark_rd": int(valid_B_per_stat[8]),
                "bootstrap_valid_B_ccw_rr": int(valid_B_per_stat[9]),
                "bootstrap_valid_B_itb_rr": int(valid_B_per_stat[10]),
                "bootstrap_valid_B_landmark_rr": int(valid_B_per_stat[11]),
                "bootstrap_valid_B_ccw_outcome_rate_control": int(valid_B_per_stat[12]),
                "bootstrap_valid_B_ccw_outcome_rate_intervention": int(valid_B_per_stat[13]),
                "bs_results_dataframe": None if bs_results_path else bs_results,
                "bs_results_path": bs_results_path,
            })
            
            ctrl_t_lo, ctrl_t_hi, valid_t_ctrl = (
                bootstrap_distribution.studentized_interval(
                    point_estimate=theta_hat_ctrl,
                    standard_error=se_hat_ctrl,
                    estimate_column=12,
                    standard_error_column=14,
                    confidence_level=bootstrap_conf,
                )
            )
            intv_t_lo, intv_t_hi, valid_t_intv = (
                bootstrap_distribution.studentized_interval(
                    point_estimate=theta_hat_intv,
                    standard_error=se_hat_intv,
                    estimate_column=13,
                    standard_error_column=15,
                    confidence_level=bootstrap_conf,
                )
            )
            rd_t_lo, rd_t_hi, valid_t_rd = (
                bootstrap_distribution.studentized_interval(
                    point_estimate=theta_hat_intv - theta_hat_ctrl,
                    standard_error=se_hat_rd,
                    estimate_column=6,
                    standard_error_column=16,
                    confidence_level=bootstrap_conf,
                )
            )
            log_rr_t_lo, log_rr_t_hi, valid_t_log_rr = (
                bootstrap_distribution.studentized_interval(
                    point_estimate=np.log(theta_hat_intv / theta_hat_ctrl),
                    standard_error=se_hat_log_rr,
                    estimate_column=9,
                    standard_error_column=17,
                    confidence_level=bootstrap_conf,
                    transform=np.log,
                )
            )
            result.update({
                "bootstrap_t_conf": bootstrap_conf,
                "ccw_outcome_rate_control_t_ci_lower": float(ctrl_t_lo) if np.isfinite(ctrl_t_lo) else np.nan,
                "ccw_outcome_rate_control_t_ci_upper": float(ctrl_t_hi) if np.isfinite(ctrl_t_hi) else np.nan,
                "ccw_outcome_rate_intervention_t_ci_lower": float(intv_t_lo) if np.isfinite(intv_t_lo) else np.nan,
                "ccw_outcome_rate_intervention_t_ci_upper": float(intv_t_hi) if np.isfinite(intv_t_hi) else np.nan,
                "bootstrap_t_valid_B_control": valid_t_ctrl,
                "bootstrap_t_valid_B_intervention": valid_t_intv,
            })
            if np.isfinite(rd_t_lo) and np.isfinite(rd_t_hi):
                result.update({
                    "ccw_rd_t_ci_lower": float(rd_t_lo),
                    "ccw_rd_t_ci_upper": float(rd_t_hi),
                    "bootstrap_t_valid_B_rd": valid_t_rd,
                })
            if np.isfinite(log_rr_t_lo) and np.isfinite(log_rr_t_hi):
                result.update({
                    "ccw_rr_t_ci_lower": float(np.exp(log_rr_t_lo)),
                    "ccw_rr_t_ci_upper": float(np.exp(log_rr_t_hi)),
                    "bootstrap_t_valid_B_log_rr": valid_t_log_rr,
                })
            logger.info(f"Run {run_id}: bootstrap完了 (method={bootstrap_method})")

        # メモリクリーンアップ
        del df_baseline, df_time_varying, df_intervention_outcome

        result.update({
            'run_id': run_id,
            'seed': seed,
            'sample_size': sample_size,
        })

        logger.info(f"Run {run_id}: 分析完了")

        return result

    except Exception as e:
        logger.error(
            f"Run {run_id if 'run_id' in locals() else 'unknown'}: エラー - {str(e)}", 
            exc_info=True)
        return {
            'run_id': run_id if 'run_id' in locals() else args_dict.get('run_id', -1),
            'seed': seed if 'seed' in locals() else args_dict.get('seed', -1),
            'error': str(e),
            'ccw_or_plr': np.nan,
            'ccw_or_km': np.nan,
            'itb_or_plr': np.nan,
            'itb_or_km': np.nan,
            'landmark_or_plr': np.nan,
            'landmark_or_km': np.nan,
        }


def run_ground_truth_simulation(args, logger):
    """Ground truthシミュレーションを実行（統一版を使用）"""
    logger.info("=== Running Ground Truth Simulation ===")

    # 共通関数を使用してGround truthシミュレーションを実行
    result_df, simulated_or, figs, scenario_params, scenario_vars, configs, intervention_strategies = _run_ground_truth_simulation(
        args, logger
    )

    return result_df, simulated_or, figs, scenario_params, scenario_vars, configs, intervention_strategies


def process_ccw_results_and_save_statistics(ccw_results_list, output_dir, logger, n_runs):
    """CCW結果の処理と統計の保存（シンプル版・列順序改善）"""

    artifacts_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    all_bootstrap_dfs = []
    all_bootstrap_paths = []
    for res in ccw_results_list:
        if "bs_results_dataframe" in res:
            bs_df = res.pop("bs_results_dataframe")
            if bs_df is not None:
                all_bootstrap_dfs.append(bs_df)
        bs_path = res.get("bs_results_path")
        if bs_path:
            all_bootstrap_paths.append(bs_path)
    # 保存
    combined_bs_path = None
    wrote_header = False
    if all_bootstrap_dfs:
        combined_bs_df = pd.concat(all_bootstrap_dfs, ignore_index=True)
        combined_bs_path = os.path.join(artifacts_dir, "bootstrap_results_combined.csv")
        combined_bs_df.to_csv(combined_bs_path, index=False)
        wrote_header = True
        logger.info(f"ブートストラップ結果を保存: {combined_bs_path}")
    if all_bootstrap_paths:
        combined_bs_path = combined_bs_path or os.path.join(artifacts_dir, "bootstrap_results_combined.csv")
        for path in sorted(set(all_bootstrap_paths)):
            if not os.path.isfile(path):
                logger.warning(f"ブートストラップ結果ファイルが見つかりません: {path}")
                continue
            df = pd.read_csv(path)
            df.to_csv(
                combined_bs_path,
                mode='a' if wrote_header else 'w',
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
        if wrote_header:
            logger.info(f"ブートストラップ結果を保存: {combined_bs_path}")

    # リスト値を持つ列を展開（重み配列など）
    processed_results = []
    for result in ccw_results_list:
        processed_result = {}
        for key, value in result.items():
            if isinstance(value, (list, np.ndarray)) and len(value):
                # リスト/配列の場合は統計量に変換
                arr = np.array(value)
                processed_result[f'{key}_mean'] = float(np.mean(arr))
                processed_result[f'{key}_std'] = float(np.std(arr))
                processed_result[f'{key}_min'] = float(np.min(arr))
                processed_result[f'{key}_max'] = float(np.max(arr))
                processed_result[f'{key}_median'] = float(np.median(arr))
            else:
                processed_result[key] = value
        processed_results.append(processed_result)
    
    # 結果を直接DataFrameに変換
    results_df = pd.DataFrame(processed_results)
    
    # エラー行を除外
    valid_results_df = results_df[~results_df['error'].notna()].copy() if 'error' in results_df.columns else results_df.copy()
    
    # 列の順序を調整：run_id, seed, sample_sizeを最初に
    id_cols = ['run_id', 'seed', 'sample_size']
    other_cols = [col for col in valid_results_df.columns if col not in id_cols]
    # ID列が存在する場合のみ並び替え
    existing_id_cols = [col for col in id_cols if col in valid_results_df.columns]
    valid_results_df = valid_results_df[existing_id_cols + other_cols]
    
    # 保存
    results_path = os.path.join(artifacts_dir, "analysis_results_detailed.csv")
    valid_results_df.to_csv(results_path, index=False)
    logger.info(f"分析結果を保存: {results_path}")

    # 基本統計の計算
    summary_stats = pd.DataFrame()
    if not valid_results_df.empty:
        numeric_cols = valid_results_df.select_dtypes(include=[np.number]).columns
        # run_id, seedなどのID列は除外
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        
        if len(numeric_cols) > 0:
            summary_stats = valid_results_df[numeric_cols].describe()

    # 基本統計を保存
    summary_path = os.path.join(artifacts_dir, "summary_statistics.csv")
    summary_stats.T.to_csv(summary_path)
    logger.info(f"基本統計を保存: {summary_path}")

    # 成功/失敗した実行数の計算
    successful_runs = [r for r in ccw_results_list if 'error' not in r]
    failed_runs = [r for r in ccw_results_list if 'error' in r]

    logger.info(f"成功した実行: {len(successful_runs)}/{len(ccw_results_list)}")

    # 主要メトリクスの基本統計を表示
    if not valid_results_df.empty:
        logger.info("分析結果サマリー")
        for method in ['ccw', 'itb', 'landmark']:
            for metric in ['or_plr', 'or_km', 'rd', 'rr']:
                col = f'{method}_{metric}'
                if col in valid_results_df.columns:
                    values = valid_results_df[col].dropna()
                    if not values.empty:
                        logger.info(f"{col}:")
                        logger.info(f"  平均: {values.mean():.4f}")
                        logger.info(f"  標準偏差: {values.std():.4f}")
                        logger.info(f"  中央値: {values.median():.4f}")

    # 実行メトリクスをローカルログへ保存
    if hasattr(logger, 'log_metric'):
        success_rate = len(successful_runs) / n_runs
        logger.log_metrics({
            'success_rate': success_rate,
            'n_successful_runs': len(successful_runs),
            'n_failed_runs': len(failed_runs)
        })

    # 既存のローカルアーティファクトを記録
    if hasattr(logger, 'log_artifacts'):
        logger.log_artifacts(artifacts_dir)

    return {
        'analysis_results': valid_results_df,
        'summary_statistics': summary_stats,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs
    }
    
    
    
def run_comprehensive_evaluation_and_log_metrics(results, simulated_or, logger, args):
    """包括的な評価を実行し、メトリクスをローカルに記録する。"""
    # stats_resultsから分析結果のDataFrameを取得
    analysis_df = results['analysis_results']
    # Ground truthとの比較とメトリクスログ
    if simulated_or is not None:
        logger.info("Ground Truthとの比較")
        logger.info(f"Ground Truth OR: {simulated_or:.4f}")

        # 各分析手法について比較
        for method in ['ccw', 'itb', 'landmark']:
            for metric in ['or_plr', 'or_km']:
                col = f'{method}_{metric}'
                if col in analysis_df.columns:
                    values = analysis_df[col].dropna()
                    if not values.empty:
                        bias = values.mean() - simulated_or
                        rmse = np.sqrt(((values - simulated_or) ** 2).mean())
                        logger.info(f"{col}:")
                        logger.info(f"  バイアス: {bias:.4f}")
                        logger.info(f"  RMSE: {rmse:.4f}")

                        # 主要メトリクスをログ
                        if hasattr(logger, 'log_metric'):
                            logger.log_metrics({
                                f"{col}_bias": bias,
                                f"{col}_rmse": rmse
                            })

    # 詳細評価の実行
    evaluation_results = None
    has_gt = results.get('ground_truth_result') is not None
    if args.run_evaluation and has_gt and 'analysis_results' in results:
        logger.info("詳細評価実行開始")

        try:
            # CCWのみの結果を抽出して評価に渡す
            ccw_results = []
            for _, row in analysis_df.iterrows():
                ccw_result = {
                    'run_id': row.get('run_id'),
                    'seed': row.get('seed'),
                    'sample_size': row.get('sample_size'),
                    'or_plr': row.get('ccw_or_plr'),
                    'or_km': row.get('ccw_or_km'),
                }
                ccw_results.append(ccw_result)

            evaluation_results = run_evaluation(
                ccw_results=ccw_results,
                ground_truth_result=results.get('ground_truth_result'),
                ground_truth_or=results.get('ground_truth_or'),
                experiment=args.experiment,
                output_dir=os.path.join(
                    logger._get_current_output_dir(), "evaluation"),
                grace_period=args.cutoff_time_of_intervention,
                observation_period=args.cutoff_time_of_observation_display,
                logger_=logger.current_logger,
                verbose=args.verbose,
                stats_df=results.get('analysis_results')
            )

            if evaluation_results and 'results' in evaluation_results:
                # 評価結果をローカルログへ保存
                eval_metrics = evaluation_results['results']
                eval_metrics_to_log = {}
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float, bool)) and hasattr(logger, 'log_metric'):
                        eval_metrics_to_log[f"eval_{key}"] = value
                logger.log_metrics(eval_metrics)

                logger.info("詳細評価が正常に完了しました")
            else:
                logger.warning("詳細評価の結果が不完全です")

        except Exception as e:
            logger.error(f"詳細評価の実行に失敗しました: {e}")
            evaluation_results = None

    return evaluation_results


def get_git_commit_hash():
    """現在のGitコミットハッシュを取得"""
    source_commit = os.environ.get("CCW_SOURCE_COMMIT")
    if source_commit:
        return source_commit[:8]
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # 短縮版
        else:
            return "unknown"
    except Exception:
        return "unknown"


def extract_main_outcome_metrics(stats_results, ground_truth_result, cutoff_time_of_observation_display, intervention_strategies):
    """メインアウトカムメトリクスを抽出する（stdメトリクス除外）。"""
    metrics = {}

    # Ground Truth結果を抽出
    if ground_truth_result is not None and not ground_truth_result.empty:
        cutoff_data = ground_truth_result[ground_truth_result['time']
                                          == cutoff_time_of_observation_display]

        for strategy in intervention_strategies:
            strategy_data = cutoff_data[cutoff_data['strategy'] == strategy]
            if not strategy_data.empty:
                row = strategy_data.iloc[0]
                metrics[f'GT_{strategy}_outcome_rate_{cutoff_time_of_observation_display}d'] = row['incident_rate']
                metrics[f'GT_{strategy}_outcome_ci_lower_{cutoff_time_of_observation_display}d'] = row['ci_lower']
                metrics[f'GT_{strategy}_outcome_ci_upper_{cutoff_time_of_observation_display}d'] = row['ci_upper']

    # 各分析手法の結果を抽出（meanのみ、stdは除外）
    if 'analysis_results' in stats_results and not stats_results['analysis_results'].empty:
        comp_stats = stats_results['analysis_results']

        # 各戦略について、各手法の平均のみを計算
        for strategy in intervention_strategies:
            for method in ['ccw', 'itb', 'landmark']:
                outcome_rate_col = f'{method}_outcome_rate_{strategy}'
                if outcome_rate_col in comp_stats.columns:
                    values = comp_stats[outcome_rate_col].dropna()
                    if not values.empty:
                        metrics[f'{method.upper()}_{strategy}_outcome_rate_{cutoff_time_of_observation_display}d'] = values.mean(
                        )

    # 全体のアウトカム率統計を抽出（meanのみ）
    if 'analysis_results' in stats_results and not stats_results['analysis_results'].empty:
        comp_stats = stats_results['analysis_results']

        # 全体のアウトカム率
        outcome_rate_col = f'outcome_{cutoff_time_of_observation_display}day_rate'
        if outcome_rate_col in comp_stats.columns:
            values = comp_stats[outcome_rate_col].dropna()
            if not values.empty:
                metrics[f'overall_outcome_rate_{cutoff_time_of_observation_display}d'] = values.mean(
                )

        # 介入別アウトカム率
        with_intervention_col = f'outcome_{cutoff_time_of_observation_display}day_with_intervention_rate'
        without_intervention_col = f'outcome_{cutoff_time_of_observation_display}day_without_intervention_rate'

        if with_intervention_col in comp_stats.columns:
            values = comp_stats[with_intervention_col].dropna()
            if not values.empty:
                metrics[f'with_intervention_outcome_rate_{cutoff_time_of_observation_display}d'] = values.mean(
                )

        if without_intervention_col in comp_stats.columns:
            values = comp_stats[without_intervention_col].dropna()
            if not values.empty:
                metrics[f'without_intervention_outcome_rate_{cutoff_time_of_observation_display}d'] = values.mean(
                )

    return metrics


def main_single_experiment(args, existing_logger=None):
    """単一実験のメイン処理"""
    # 出力ディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if existing_logger is None and args.output_dir is None:
        args.output_dir = f"output/batch_{timestamp}"
        os.makedirs(args.output_dir, exist_ok=True)

    # ワーカー数の設定
    if args.n_workers is None:
        args.n_workers = min(multiprocessing.cpu_count()//2, args.n_runs)
    if getattr(args, "ground_truth_workers", None) is None:
        args.ground_truth_workers = 1

    profile_mode = os.getenv("PROFILE_MODE", "0") == "1"

    owns_logger = existing_logger is None
    logger_ctx = (RunLogger(
                  output_dir=args.output_dir,
                  append_logs=getattr(args, "_resume_from_output_dir", False)) if owns_logger else existing_logger)

    with (logger_ctx if owns_logger else nullcontext(logger_ctx)) as logger:
        logger.info("バッチシミュレーション開始")
        logger.info(f"実験: {args.experiment}")
        logger.info(f"サンプルサイズ: {args.sample_size}")
        logger.info(f"Ground truthサンプルサイズ: {args.ground_truth_sample_size}")
        logger.info(f"Ground truthワーカー数: {args.ground_truth_workers}")
        logger.info(f"実行回数: {args.n_runs}")
        logger.info(f"ワーカー数: {args.n_workers}")
        logger.info(f"出力ディレクトリ: {logger._get_current_output_dir()}")
        logger.info(f"評価実行: {args.run_evaluation}")
        logger.info(f"IPTW使用: {args.run_IPTW}")
        if args.config_dir:
            logger.info(f"config_dir: {args.config_dir}")
        if args.ground_truth_cache_dir:
            logger.info(f"ground_truth_cache_dir: {args.ground_truth_cache_dir}")
        if getattr(args, "_resume_from_output_dir", False):
            logger.info("再開モード: 既存のoutput_dirからパラメータを復元しました")

        # Gitコミット番号を取得してログ
        commit_hash = get_git_commit_hash()
        logger.info(f"Gitコミット: {commit_hash}")

        # パラメータ保存（早期保存・再開用）
        if not getattr(args, "_resume_from_output_dir", False):
            save_batch_params(
                output_dir=logger._get_current_output_dir(),
                args=args,
                commit_hash=commit_hash,
                logger=logger,
            )

        results = {}
        intervention_strategies = None

        # メインの処理を開始
        nested = not owns_logger
        with logger.start_run(run_name=f"batch_{args.experiment}_{timestamp}", nested=nested):

            # Ground truthシミュレーション
            simulated_or = None
            ground_truth_result = None
            gt_cache_base_dir = args.ground_truth_cache_dir or getattr(
                logger, "base_output_dir", None) or args.output_dir
            gt_cache_expected_meta = _build_gt_cache_meta(args, commit_hash)
            gt_cache_used = False
            gt_cache_meta = None
            gt_source = None

            # 共有の結果があればそれを使う
            shared_gt = getattr(args, 'shared_ground_truth_result', None)
            shared_or = getattr(args, 'shared_simulated_or', None)
            if shared_gt is None or shared_or is None:
                # try first to load from cache
                gt_cache = load_ground_truth_cache(gt_cache_base_dir, logger=logger)
                if gt_cache is not None:
                    cached_meta = gt_cache.get("meta", None)
                    gt_cache_meta = cached_meta
                    if cached_meta is None:
                        logger.warning("Ground Truthキャッシュにmetaが無いため再計算します")
                    elif not _gt_cache_matches(cached_meta, gt_cache_expected_meta, logger=logger):
                        logger.warning("Ground Truthキャッシュが現在の設定と一致しないため再計算します")
                    else:
                        shared_gt = gt_cache.get("ground_truth_result")
                        shared_or = gt_cache.get("simulated_or")
                        if shared_gt is not None and shared_or is not None:
                            args.shared_ground_truth_result = shared_gt
                            args.shared_simulated_or = shared_or
                            gt_cache_used = True
                            gt_source = "cache"
                            logger.info(
                                f"Ground Truthキャッシュを使用します: {_get_ground_truth_cache_path(gt_cache_base_dir)}")
                        else:
                            if args.skip_ground_truth:
                                logger.warning("Ground Truthキャッシュが不完全です。skip_ground_truthのため再計算しません")
                            else:
                                logger.warning("Ground Truthキャッシュが不完全なため再計算します")

            if shared_gt is not None and shared_or is not None:
                logger.info("共有Ground Truthを使用します（シナリオ実行から供給）")
                ground_truth_result = shared_gt
                simulated_or = float(shared_or)
                results['ground_truth_or'] = simulated_or
                results['ground_truth_result'] = ground_truth_result
                if gt_source is None:
                    gt_source = "shared"

                scenario_params, scenario_vars, configs = load_experiment_settings(
                    args.experiment)
                intervention_strategies = {
                    key: item(args) for key, item in configs["strategy_creator"].items()}

            elif not args.skip_ground_truth:
                logger.info("Ground truthシミュレーション開始")
                with logger.start_run(run_name="ground_truth_simulation", nested=True):
                    # Ground truth用のargsを作成（専用サンプルサイズを使用）
                    gt_args = argparse.Namespace(**vars(args))
                    gt_args.sample_size = args.ground_truth_sample_size

                    ground_truth_result, simulated_or, figs, scenario_params, scenario_vars, configs, intervention_strategies = run_ground_truth_simulation(
                        gt_args, logger
                    )
                    results['ground_truth_or'] = simulated_or
                    results['ground_truth_result'] = ground_truth_result
                    save_ground_truth_cache(
                        output_dir=gt_cache_base_dir,
                        ground_truth_result=ground_truth_result,
                        simulated_or=simulated_or,
                        logger=logger,
                        meta=gt_cache_expected_meta,
                    )
                    gt_source = "computed"
            else:
                # Ground truthをスキップする場合でも、intervention_strategiesを取得
                scenario_params, scenario_vars, configs = load_experiment_settings(
                    args.experiment)
                intervention_strategies = {
                    key: item(args) for key, item in configs["strategy_creator"].items()
                }
                gt_source = "skipped"

            # Ground Truth情報の保存
            base_output_dir = getattr(logger, "base_output_dir", None) or args.output_dir
            gt_info = {
                "ground_truth_source": gt_source,
                "cache_dir": gt_cache_base_dir,
                "cache_path": _get_ground_truth_cache_path(gt_cache_base_dir),
                "cache_used": bool(gt_cache_used),
                "cache_meta_expected": gt_cache_expected_meta,
                "cache_meta_used": gt_cache_meta,
                "ground_truth_or": float(simulated_or) if simulated_or is not None else None,
            }
            _write_ground_truth_info(base_output_dir, gt_info, logger=logger)

            # 複数分析手法のバッチ実行
            logger.info(f"{args.n_runs}回の分析開始（CCW、ITB、Landmark）")
            with logger.start_run(run_name="multi_method_batch_analysis", nested=True):
                run_output_dir = logger._get_current_output_dir()
                # 各実行のパラメータを準備
                analysis_args_list = [
                    {
                        'experiment': args.experiment,
                        'sample_size': args.sample_size,  # バッチ分析用のサンプルサイズ
                        'n_time': args.n_time,
                        'cutoff_time_of_intervention': args.cutoff_time_of_intervention,
                        'cutoff_time_of_observation': args.cutoff_time_of_observation,
                        'cutoff_time_of_observation_display': args.cutoff_time_of_observation_display,
                        'seed': args.base_seed + run_id,
                        'run_id': run_id,
                        'verbose': args.verbose,
                        'run_IPTW': args.run_IPTW,
                        'bootstrap_in_run': getattr(args, 'bootstrap', False),
                        'bootstrap_B': getattr(args, 'bootstrap_B', 0),
                        'bootstrap_seed': getattr(args, 'bootstrap_seed', 2025),
                        'bootstrap_conf': getattr(args, 'bootstrap_conf', 0.95),
                        'bootstrap_method': 'basic',
                        'save_ipw_weights': getattr(args, 'save_ipw_weights', False),
                        'output_dir': run_output_dir,
                        'config_dir': getattr(args, 'config_dir', None),
                    }
                    for run_id in range(args.n_runs)
                ]

                # 並列実行
                # log のための queue

                checkpoint_dir = _get_checkpoint_dir(
                    logger._get_current_output_dir())
                completed_results = load_run_checkpoints(
                    checkpoint_dir, logger=logger)
                completed_run_ids = set(completed_results.keys())
                if completed_run_ids:
                    logger.info(
                        f"再開検出: {len(completed_run_ids)}件の完了runをスキップします")
                analysis_results_list = list(completed_results.values())
                analysis_args_list = [
                    args_dict for args_dict in analysis_args_list
                    if args_dict.get('run_id') not in completed_run_ids
                ]
                use_mp = (args.n_workers > 1)

                if profile_mode:
                    def progress_iter(iterable): return iterable
                else:
                    def progress_iter(iterable): return tqdm(
                        iterable, desc="Multi-Method Analysis")

                if use_mp:
                    mgr = mp.Manager()
                    q = mgr.Queue(-1)
                    # threading
                    t = threading.Thread(target=logger_thread, args=(
                        q, logger.current_logger_name), daemon=True)
                    t.start()
                    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                        futures = {
                            executor.submit(run_single_simulation_analysis, args_dict, q,  logger.current_logger_name): i
                            for i, args_dict in enumerate(analysis_args_list)
                        }

                        for future in progress_iter(as_completed(futures)):
                            result = future.result()
                            run_id = result.get('run_id')
                            if run_id is not None:
                                save_run_checkpoint(
                                    checkpoint_dir, int(run_id), result, logger=logger)
                            analysis_results_list.append(result)
                        q.put(None)
                        t.join()
                else:
                    for arg_dict in progress_iter(analysis_args_list):
                        result = run_single_simulation_analysis(
                            arg_dict, None, logger.current_logger_name)
                        run_id = result.get('run_id')
                        if run_id is not None:
                            save_run_checkpoint(
                                checkpoint_dir, int(run_id), result, logger=logger)
                        analysis_results_list.append(result)

                current_output_dir = logger._get_current_output_dir()

                # 統合された統計処理
                stats_results = process_ccw_results_and_save_statistics(
                    analysis_results_list, current_output_dir, logger, args.n_runs
                )

                if len(stats_results['successful_runs']) > 0:
                    results.update(stats_results)

            # メインアウトカムメトリクスをログ
            if intervention_strategies:
                logger.info("メインアウトカムメトリクスのログ開始")
                extract_main_outcome_metrics(
                    stats_results=results,
                    ground_truth_result=ground_truth_result,
                    cutoff_time_of_observation_display=args.cutoff_time_of_observation_display,
                    intervention_strategies=intervention_strategies
                )

            # 包括的評価とメトリクスログ
            if 'analysis_results' in results and len(results['analysis_results']) > 0:
                logger.info("包括的評価とメトリクスログ開始")
                evaluation_results = run_comprehensive_evaluation_and_log_metrics(
                    results=results,
                    simulated_or=simulated_or,
                    logger=logger,
                    args=args,
                )

                if evaluation_results:
                    results['evaluation_results'] = evaluation_results
            else:
                logger.warning("包括的評価をスキップしました（分析結果がありません）")

            # 完了マーカーの作成 + チェックポイント掃除
            base_output_dir = getattr(logger, "base_output_dir", None) or args.output_dir
            completion_path = _get_completion_marker_path(base_output_dir)
            try:
                payload = f"completed_at={datetime.now().isoformat()}\n"
                _atomic_write_bytes(payload.encode("utf-8"), completion_path)
                logger.info(f"完了マーカー作成: {completion_path}")
            except Exception as e:
                logger.warning(f"完了マーカー作成に失敗: {completion_path} ({e})")

            checkpoint_dir = _get_checkpoint_dir(base_output_dir)
            if os.path.isdir(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"チェックポイント削除: {checkpoint_dir}")
                except Exception as e:
                    logger.warning(f"チェックポイント削除に失敗: {checkpoint_dir} ({e})")

            logger.info(f"全結果保存完了: {args.output_dir}")

    return results


def run_experiment(config: ExperimentRunConfig, output_dir: str | Path) -> RunArtifacts:
    """Run one experiment through the package API used by CLI wrappers and tests."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    args = config.to_namespace(destination)
    set_config_override_dir(args.config_dir)
    result = main_single_experiment(args)
    artifacts_dir = destination / "multi_method_batch_analysis" / "artifacts"
    return RunArtifacts(
        output_dir=destination,
        completion_marker=destination / COMPLETION_MARKER_NAME,
        analysis_results_csv=artifacts_dir / "analysis_results_detailed.csv",
        summary_statistics_csv=artifacts_dir / "summary_statistics.csv",
        result=result,
    )


SCENARIO_EXPERIMENTS = {
    "scenario1": ["experimentA", "experimentB"],
    "scenario2": ["experimentC"],
    "scenario3": ["experimentD", "experimentE", "experimentF"],
}

# 追加: シナリオ一括実行エントリ


def main_multi_experiments_for_scenario(args):
    """指定シナリオに紐づく複数実験を、データ/GT共有で実行"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario = args.scenario

    experiments = list(dict.fromkeys(SCENARIO_EXPERIMENTS[scenario]))
    scenario_output_dir = args.output_dir or f"output/{timestamp}_{scenario}"
    os.makedirs(scenario_output_dir, exist_ok=True)

    # シナリオ共通の Ground Truth を1回だけ生成
    with RunLogger(output_dir=scenario_output_dir,
                      append_logs=getattr(args, "_resume_from_output_dir", False)) as scenario_logger:
        scenario_logger.info(f"シナリオ実行開始: {scenario} -> 実験: {experiments}")
        with scenario_logger.start_run(run_name=f"{scenario}_{timestamp}") as _:
            # 代表実験の設定で GT を生成（同一シナリオであれば同一になる想定）
            rep_exp = experiments[0]
            gt_args = argparse.Namespace(**vars(args))
            gt_args.experiment = rep_exp
            gt_args.output_dir = scenario_output_dir
            gt_args.sample_size = args.ground_truth_sample_size
            with scenario_logger.start_run(run_name="ground_truth_simulation", nested=True):
                ground_truth_result, simulated_or, figs, scenario_params, scenario_vars, _, _ = run_ground_truth_simulation(
                    gt_args, scenario_logger
                )
                scenario_logger.info(f"共有 Ground Truth OR: {simulated_or:.4f}")

            # 各実験を共有GTで個別実行（シードは base_seed + run_id で一致）
            all_results = {}
            for exp in experiments:
                exp_args = argparse.Namespace(**vars(args))
                exp_args.experiment = exp
                exp_args.output_dir = os.path.join(scenario_output_dir, exp)
                # os.makedirs(exp_args.output_dir, exist_ok=True)

                # 各実験では GT をスキップし、共有GTを注入
                exp_args.skip_ground_truth = True
                exp_args.shared_ground_truth_result = ground_truth_result
                exp_args.shared_simulated_or = simulated_or

                results = main_single_experiment(
                    exp_args, existing_logger=scenario_logger)
                all_results[exp] = results

    return {
        "scenario": scenario,
        "experiments": experiments,
        "output_dir": scenario_output_dir,
        "shared_ground_truth_or": simulated_or,
        "shared_ground_truth_result": ground_truth_result,
        "experiment_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch simulation for CCW analysis")

    # 基本パラメータ
    parser.add_argument("--experiment", type=str, nargs='+',
                        choices=["experimentA", "experimentB", "experimentC",
                                 "experimentD", "experimentE", "experimentF"],
                        default=["experimentA"], help="Experiment name(s)")

    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Sample size for each CCW analysis")

    parser.add_argument("--ground_truth_sample_size", type=int, default=10000,
                        help="Sample size for ground truth simulation (default: 10000)")

    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of CCW analysis runs (default: 3)")

    parser.add_argument("--n_time", type=int, default=40,
                        help="Number of time steps")

    parser.add_argument("--cutoff_time_of_intervention", type=int, default=2,
                        help="Cutoff time of intervention")

    parser.add_argument("--cutoff_time_of_observation", type=int, default=31,
                        help="Cutoff time of observation")

    parser.add_argument("--cutoff_time_of_observation_display", type=int, default=30,
                        help="Cutoff time of observation for display")

    # シード設定
    parser.add_argument("--ground_truth_seed", type=int, default=42,
                        help="Seed for ground truth simulation")

    parser.add_argument("--base_seed", type=int, default=1000,
                        help="Base seed for CCW analysis runs")

    # 並列処理設定
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--ground_truth_workers", type=int, default=None,
                        help="Number of parallel workers for ground truth simulation (default: 1)")

    # 出力設定
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/batch_TIMESTAMP)")

    parser.add_argument("--save_ipw_weights", action='store_true',
                        help="Save CCW IPW weights from the main run to CSV in output_dir/artifacts")

    parser.add_argument("--config_dir", type=str, default=None,
                        help="Override directory containing scenario*.yaml files")
    parser.add_argument("--ground_truth_cache_dir", type=str, default=None,
                        help="Shared directory to cache ground truth across runs")

    parser.add_argument("--skip_ground_truth", action='store_true',
                        help="Skip ground truth simulation")

    parser.add_argument("--skip_figures", action='store_true',
                        help="Skip plotting figures")

    # 評価設定
    parser.add_argument("--run_evaluation", action='store_true',
                        help="Run comprehensive evaluation after batch analysis")

    # verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")

    parser.add_argument("--run_IPTW", action='store_true',
                        help="Run IPTW analysis")

    # bootstrap 系
    parser.add_argument("--bootstrap", action='store_true',
                        help="各 run 内で個体（id）レベルのブートストラップを実施")
    parser.add_argument("--bootstrap_B", type=int, default=100,
                        help="ブートストラップ反復数（0以下で無効、デフォルト: 100）")
    parser.add_argument("--bootstrap_seed", type=int, default=2025,
                        help="ブートストラップの乱数シード（デフォルト: 2025）")
    parser.add_argument("--bootstrap_conf", type=float, default=0.95,
                        help="信頼区間の信頼係数（デフォルト: 0.95）")
    # Preserve the historical field in parameter files while exposing only
    # ordinary subject-level bootstrap.
    parser.set_defaults(bootstrap_method="basic")

    parser.add_argument("--scenario", type=str,
                        choices=list(SCENARIO_EXPERIMENTS.keys()),
                        help="シナリオ名。対応する複数実験をデータ/GT共有で実行")

    args = parser.parse_args()

    # 完了済みなら何もしない
    if args.output_dir:
        completion_path = _get_completion_marker_path(args.output_dir)
        if os.path.isfile(completion_path):
            print(f"既に完了しています: {completion_path}")
            return None

    # output_dir に過去の設定があれば読み込み（再開用）
    if args.output_dir:
        params_path = _get_batch_params_path(args.output_dir)
        if os.path.isfile(params_path):
            try:
                cli_n_workers = args.n_workers
                cli_ground_truth_workers = args.ground_truth_workers
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                resume_args = argparse.Namespace(**vars(args))
                for k, v in params.items():
                    setattr(resume_args, k, v)
                # 再開時は保存済みパラメータを基本とし、ワーカー数のみCLI指定を優先
                if cli_n_workers is not None:
                    resume_args.n_workers = cli_n_workers
                if cli_ground_truth_workers is not None:
                    resume_args.ground_truth_workers = cli_ground_truth_workers
                resume_args.output_dir = args.output_dir
                if isinstance(resume_args.experiment, str):
                    resume_args.experiment = [resume_args.experiment]
                args = resume_args
                setattr(args, "_resume_from_output_dir", True)
            except Exception:
                logging.getLogger(__name__).warning(
                    "再開用設定の読み込みに失敗しました。指定された引数で続行します。",
                    exc_info=True,
                )

    if args.config_dir:
        set_config_override_dir(args.config_dir)

    # run_evaluationはいつもtrueにしてしまう
    args.run_evaluation = True
    if args.ground_truth_workers is None:
        args.ground_truth_workers = 1

    if args.scenario:
        return main_multi_experiments_for_scenario(args)

    # 複数の実験が指定された場合
    if len(args.experiment) > 1:
        raise NotImplementedError("複数実験の一括実行は現在サポートされていません。単一実験を指定してください。")
    else:
        # 単一実験の場合
        args.experiment = args.experiment[0]
        return main_single_experiment(args)


if __name__ == "__main__":
    main()
