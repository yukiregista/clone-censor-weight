import re
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Optional
from ccw._logging import TagAdapter, temp_stderr_logging, to_level
from logging import getLogger

_base_logger = getLogger(__name__)


def calculate_ci_coverage(
    results_df,
    ground_truth_or: float,
    ground_truth_result: Optional[pd.DataFrame] = None,
    observation_period: Optional[int] = None
) -> Dict[str, float]:
    """ブートストラップCIに対するカバレッジ（真値がCIに入っている割合）を計算する。
    対応列:
      - {method}_{stat}_ci_lower / {method}_{stat}_ci_upper
        method ∈ {ccw, itb, landmark, ...}
        stat   ∈ {or_plr, or_km, rd, rr, outcome_rate_control, outcome_rate_intervention, ...}
    真値:
      - OR(PLR/KM): ground_truth_or
      - RD/RR: ground_truth_result から observation_period の incident_rate で算出
      - それ以外: truths 辞書を拡張することで対応（デフォルトでは None = カバレッジ計算しない）
    """
    # DataFrame化
    if isinstance(results_df, list):
        df = pd.DataFrame(results_df)
    else:
        df = results_df

    coverage: Dict[str, float] = {}
    if df is None or df.empty:
        return coverage

    # ---- 真値の準備 ----
    truths: Dict[str, Optional[float]] = {
        "or_plr": ground_truth_or if ground_truth_or is not None and not pd.isna(ground_truth_or) else None,
        "or_km":  ground_truth_or if ground_truth_or is not None and not pd.isna(ground_truth_or) else None,
        "rd": None,
        "rr": None,
        "outcome_rate_intervention": None,
        "outcome_rate_control": None,
    }

    # RD/RR と outcome_rate_(intervention/control) の真値を ground_truth_result から算出
    if ground_truth_result is not None and not ground_truth_result.empty and observation_period is not None:
        try:
            gt_cut = ground_truth_result[ground_truth_result["time"]
                                         == observation_period]
            p1 = gt_cut.loc[gt_cut["strategy"] ==
                            "intervention", "incident_rate"].iloc[0]
            p0 = gt_cut.loc[gt_cut["strategy"] ==
                            "control", "incident_rate"].iloc[0]
            if pd.notna(p1) and pd.notna(p0):
                p1 = float(p1)
                p0 = float(p0)
                truths["rd"] = p1 - p0
                truths["rr"] = p1 / p0 if p0 != 0.0 else np.nan
                truths["outcome_rate_intervention"] = p1
                truths["outcome_rate_control"] = p0
        except Exception:
            # ground_truth_result が期待通りでなければ黙ってスキップ
            pass

    def add_coverage_for(method: str, stat: str, true_val: Optional[float]):
        if true_val is None or pd.isna(true_val):
            return

        lo_col = f"{method}_{stat}_ci_lower"
        hi_col = f"{method}_{stat}_ci_upper"
        if lo_col not in df.columns or hi_col not in df.columns:
            return

        valid = df[[lo_col, hi_col]].dropna()
        if valid.empty:
            return

        within = (valid[lo_col] <= true_val) & (true_val <= valid[hi_col])
        key = f"ci_coverage_{method}_{stat}"
        coverage[key] = float(within.mean())
        coverage[f"{key}_count"] = int(within.sum())
        coverage[f"{key}_total"] = int(len(valid))
        
        
        # also check if t bootstrap result is there
        t_lo_col = f"{method}_{stat}_t_ci_lower"
        t_hi_col = f"{method}_{stat}_t_ci_upper"
        if t_lo_col in df.columns and t_hi_col in df.columns:
            valid_t = df[[t_lo_col, t_hi_col]].dropna()
            if not valid_t.empty:
                within_t = (valid_t[t_lo_col] <= true_val) & (true_val <= valid_t[t_hi_col])
                key_t = f"ci_coverage_{method}_{stat}_t"
                coverage[key_t] = float(within_t.mean())
                coverage[f"{key_t}_count"] = int(within_t.sum())
                coverage[f"{key_t}_total"] = int(len(valid_t))

    # ---- 列名から CI を自動抽出 ----
    # パターン: {method}_{stat}_ci_lower / _ci_upper
    ci_pattern = re.compile(
        r"^(?P<method>[^_]+)_(?P<stat>.+)_ci_(lower|upper)$")

    ci_specs: set[tuple[str, str]] = set()
    for col in df.columns:
        m = ci_pattern.match(col)
        if not m:
            continue
        method = m.group("method")
        stat = m.group("stat")
        ci_specs.add((method, stat))

    # 見つかった全ての (method, stat) について coverage を計算
    for method, stat in sorted(ci_specs):
        true_val = truths.get(stat)
        # 真値が定義されていない stat はスキップ（OR/RD/RR/outcome_rate_* 以外）
        add_coverage_for(method, stat, true_val)

    return coverage


def calculate_basic_evaluation_metrics(ccw_results,
                                       ground_truth_or: float,
                                       stats_df: Optional[pd.DataFrame] = None,
                                       ground_truth_result: Optional[pd.DataFrame] = None,
                                       observation_period: Optional[int] = None) -> Dict[str, float]:
    """基本的な評価指標の計算（CI coverage追加）"""

    # ccw_resultsがリストの場合はDataFrameに変換
    if isinstance(ccw_results, list):
        ccw_results_df = pd.DataFrame(ccw_results)
    else:
        ccw_results_df = ccw_results

    metrics: Dict[str, float] = {
        "n_total_runs": int(len(ccw_results_df)) if ccw_results_df is not None else 0,
    }

    # CI coverage は stats_df（bootstrap結果など）を優先
    coverage_source = stats_df if stats_df is not None else ccw_results_df

    coverage_metrics = calculate_ci_coverage(
        results_df=coverage_source,
        ground_truth_or=ground_truth_or,
        ground_truth_result=ground_truth_result,
        observation_period=observation_period
    )
    metrics.update(coverage_metrics)

    # ---- 後方互換: 成功数/成功率のキーを埋める ----
    # coverage で "total" が取れているものがあれば、それを成功母数の代表として利用。
    totals = [v for k, v in coverage_metrics.items() if isinstance(v, int) and k.endswith("_total")]

    # total が取れない場合は、とりあえず全件を母数とする（ログ用）。
    n_total_for_success = int(max(totals)) if totals else metrics["n_total_runs"]

    # successful_runs の定義を「coverage判定に使えた行数」とする
    metrics["n_successful_runs"] = int(n_total_for_success)
    metrics["success_rate"] = (metrics["n_successful_runs"] / metrics["n_total_runs"]) if metrics["n_total_runs"] > 0 else 0.0

    return metrics


def run_evaluation(ccw_results,
                   ground_truth_result: pd.DataFrame,
                   ground_truth_or: float,
                   experiment: str,
                   output_dir: str,
                   grace_period: int = 2,
                   observation_period: int = 30,
                   logger_=None,
                   verbose=None,
                   stats_df: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    簡素化された評価実行関数
    """
    lg = TagAdapter(logger_ or _base_logger, tag="run_evaluation")
    with temp_stderr_logging(lg.logger, level=to_level(verbose), user_supplied=(logger_ is not None)):
        try:
            lg.info(f"=== {experiment} 評価開始 ===")

            # ccw_resultsがリストの場合はDataFrameに変換
            if isinstance(ccw_results, list):
                ccw_results_df = pd.DataFrame(ccw_results)
            else:
                ccw_results_df = ccw_results

            # 基本評価指標を計算
            evaluation_metrics = calculate_basic_evaluation_metrics(
                ccw_results_df,
                ground_truth_or,
                stats_df=stats_df,
                ground_truth_result=ground_truth_result,
                observation_period=observation_period
            )

            # 評価結果を準備
            evaluation_results = {
                'experiment': experiment,
                'evaluation_date': datetime.now().isoformat(),
                'ground_truth_or': ground_truth_or,
                'grace_period': grace_period,
                'observation_period': observation_period,
                'results': evaluation_metrics
            }

            # 結果保存
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JSON形式で保存
            json_path = os.path.join(
                output_dir, f"{experiment}_evaluation_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f,
                          ensure_ascii=False, indent=2, default=str)

            # CSV形式でも保存
            metrics_df = pd.DataFrame([evaluation_metrics])
            csv_path = os.path.join(
                output_dir, f"{experiment}_evaluation_metrics_{timestamp}.csv")
            metrics_df.to_csv(csv_path, index=False)

            lg.info(f"評価結果を保存しました:")
            lg.info(f"  JSON: {json_path}")
            lg.info(f"  CSV:  {csv_path}")

            # サマリー表示
            lg.info(f"=== {experiment} 評価サマリー ===")
            lg.info(
                f"成功実行数: {evaluation_metrics.get('n_successful_runs', 0)}/{evaluation_metrics.get('n_total_runs', 0)}")
            lg.info(f"成功率: {evaluation_metrics.get('success_rate', 0):.1%}")

            if not np.isnan(ground_truth_or):
                lg.info(f"Ground Truth OR: {ground_truth_or:.4f}")

                for method in ['plr', 'km']:
                    bias_key = f'mortality_bias_{method}'
                    rmse_key = f'mortality_rmse_{method}'

                    if bias_key in evaluation_metrics and rmse_key in evaluation_metrics:
                        lg.info(f"{method.upper()}:")
                        lg.info(f"  バイアス: {evaluation_metrics[bias_key]:.4f}")
                        lg.info(f"  RMSE: {evaluation_metrics[rmse_key]:.4f}")

            return evaluation_results

        except Exception as e:
            lg.error(f"評価実行中にエラーが発生しました: {e}")
            return None


def main():
    """コマンドライン実行用のメイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="バッチ実行結果の簡素化評価")
    parser.add_argument("--ccw_results", type=str, required=True,
                        help="CCW分析結果のCSVファイルパス")
    parser.add_argument("--ground_truth_or", type=float, required=True,
                        help="Ground truthのオッズ比")
    parser.add_argument("--experiment", type=str, required=True,
                        help="実験名")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="評価結果出力ディレクトリ")
    parser.add_argument("--grace_period", type=int, default=2,
                        help="Grace periodの日数")
    parser.add_argument("--observation_period", type=int, default=30,
                        help="観察期間の日数")

    args = parser.parse_args()

    # CSVファイルを読み込み
    ccw_results = pd.read_csv(args.ccw_results)

    # 評価実行（ground_truth_resultは使用しないためダミー）
    results = run_evaluation(
        ccw_results=ccw_results,
        ground_truth_result=pd.DataFrame(),  # 未使用
        ground_truth_or=args.ground_truth_or,
        experiment=args.experiment,
        output_dir=args.output_dir,
        grace_period=args.grace_period,
        observation_period=args.observation_period,
        logger_=None,
        verbose=None
    )

    return results


if __name__ == "__main__":
    main()
