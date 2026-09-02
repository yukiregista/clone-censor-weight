import os
import subprocess
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import matplotlib.pyplot as plt
from ccw._research.configs.load_variables import load_experiment_settings
from ccw._research.analysis import simulate_true_effect
import pandas as pd
import shutil
import json
import numpy as np
from datetime import datetime
from ccw._research.utils import flatten_dict


def _split_sample_size(total_size: int, n_chunks: int) -> list[int]:
    n_chunks = max(1, min(int(n_chunks), int(total_size)))
    base = int(total_size) // n_chunks
    remainder = int(total_size) % n_chunks
    return [base + (1 if idx < remainder else 0) for idx in range(n_chunks)]


def _ground_truth_worker_args(args) -> dict:
    allowed_types = (str, int, float, bool, type(None), list, tuple)
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, allowed_types) and not str(key).startswith("_")
    }


def _simulate_ground_truth_chunk(payload: dict) -> dict:
    from argparse import Namespace

    from ccw._research.configs.config_override import set_config_override_dir

    config_dir = payload.get("config_dir")
    if config_dir:
        set_config_override_dir(config_dir)

    chunk_args = Namespace(**payload["args"])
    chunk_args.sample_size = int(payload["sample_size"])
    chunk_args.ground_truth_seed = int(payload["seed"])

    scenario_params, scenario_vars, configs = load_experiment_settings(chunk_args.experiment)
    intervention_strategies = {
        key: item(chunk_args) for key, item in configs["strategy_creator"].items()
    }
    result_df, simulated_or, _figs = simulate_true_effect(
        base_variables=scenario_vars,
        n_time=chunk_args.n_time,
        strategies=intervention_strategies,
        treatment_var=configs["treatment_var"],
        cutoff_time_of_observation_display=chunk_args.cutoff_time_of_observation_display,
        sample_size=chunk_args.sample_size,
        seed=chunk_args.ground_truth_seed,
        plot_figures=False,
        logger_=None,
        verbose=0,
    )
    return {
        "index": int(payload["index"]),
        "sample_size": int(payload["sample_size"]),
        "seed": int(payload["seed"]),
        "simulated_or": float(simulated_or),
        "result_df": result_df,
    }


def _combine_ground_truth_chunks(chunk_results: list[dict], cutoff_time: int) -> tuple[pd.DataFrame, float]:
    frames = []
    for result in chunk_results:
        df = result["result_df"].copy()
        df["chunk_sample_size"] = int(result["sample_size"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    rows = []
    for (time, strategy), group in all_df.groupby(["time", "strategy"], sort=True):
        sample_sizes = pd.to_numeric(group["chunk_sample_size"], errors="raise").to_numpy(dtype=float)
        rates = pd.to_numeric(group["incident_rate"], errors="raise").to_numpy(dtype=float)
        total_n = float(sample_sizes.sum())
        incident_rate = float(np.sum(rates * sample_sizes) / total_n)
        se = np.sqrt(incident_rate * (1.0 - incident_rate) / max(total_n, 1.0))
        rows.append(
            {
                "time": int(time),
                "strategy": strategy,
                "incident_rate": incident_rate,
                "ci_lower": incident_rate - 1.96 * se,
                "ci_upper": incident_rate + 1.96 * se,
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["time", "strategy"]).reset_index(drop=True)
    cutoff_data = result_df[result_df["time"] == int(cutoff_time)]
    p1 = cutoff_data.loc[cutoff_data["strategy"] == "intervention", "incident_rate"].iloc[0]
    p0 = cutoff_data.loc[cutoff_data["strategy"] == "control", "incident_rate"].iloc[0]
    simulated_or = (p1 / (1.0 - p1)) / (p0 / (1.0 - p0))
    return result_df, float(simulated_or)


def _run_parallel_ground_truth_simulation(args, logger, sample_size: int, seed: int) -> tuple[pd.DataFrame, float, dict]:
    requested_workers = int(getattr(args, "ground_truth_workers", 1) or 1)
    chunk_sizes = _split_sample_size(sample_size, requested_workers)
    n_workers = min(requested_workers, len(chunk_sizes))
    logger.info(
        "Ground truth parallel chunks: "
        f"total_sample_size={sample_size}, chunks={len(chunk_sizes)}, workers={n_workers}"
    )

    payloads = [
        {
            "index": idx,
            "sample_size": size,
            "seed": int(seed) + idx,
            "args": _ground_truth_worker_args(args),
            "config_dir": getattr(args, "config_dir", None),
        }
        for idx, size in enumerate(chunk_sizes)
    ]

    chunk_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_simulate_ground_truth_chunk, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            result = future.result()
            chunk_results.append(result)
            logger.info(
                "Ground truth chunk finished: "
                f"{len(chunk_results)}/{len(payloads)} "
                f"(chunk={payload['index']}, size={payload['sample_size']}, seed={payload['seed']})"
            )

    chunk_results.sort(key=lambda item: item["index"])
    result_df, simulated_or = _combine_ground_truth_chunks(
        chunk_results,
        cutoff_time=args.cutoff_time_of_observation_display,
    )
    return result_df, simulated_or, {}

class RunContext:
    def __init__(self, logger, nested, run_name):
        self.logger = logger
        self.nested = nested
        self.run_name = run_name
        
    def __enter__(self):
        # 常に自分自身を返す
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        # 常にend_runを呼ぶ
        self.logger.end_run()


def get_git_commit_hash():
    """現在のGitコミットハッシュを取得"""
    source_commit = os.environ.get("CCW_SOURCE_COMMIT")
    if source_commit:
        return source_commit[:8]
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # 短縮版
        else:
            return "unknown"
    except Exception:
        return "unknown"

class RunLogger:
    """Hierarchical local logger for experiment parameters, metrics, and files."""

    def __init__(self, log_level=logging.DEBUG, logger_name="CCWSimulation", output_dir=None, append_logs: bool = False):
        self.log_file_path = None
        self.base_output_dir = output_dir
        self.output_dir = output_dir
        self.append_logs = append_logs
        
        # ローカルログ用の変数
        self.local_metrics = {}
        self.local_params = {}
        self.local_tags = {}
        self.nested_runs = {}
        
        # 階層ロガー管理用の新しい属性を追加
        self.base_logger_name = logger_name
        self.run_stack = []  # ランのスタック [run_name1, run_name2, ...]
        self.current_logger_name = logger_name  # 現在のロガー名
        self.root_logger = None  # ルートロガー（ハンドラーを持つ）
        self.current_logger = None  # 現在使用中のロガー
        
        # ネストラン専用ファイルハンドラー管理用
        self.run_file_handlers = {}  # {run_path: file_handler}
        self.run_log_files = {}  # {run_path: log_file_path}
        
        # Gitコミット番号を取得
        self.git_commit_hash = get_git_commit_hash()
        
        # ローカルメトリクス用ディレクトリを設定
        if self.output_dir:
            self.local_logs_dir = self.output_dir #os.path.join(self.output_dir, "experiment_logs")
        else:
            self.local_logs_dir = os.path.join(tempfile.gettempdir(), "experiment_logs")
        
        os.makedirs(self.local_logs_dir, exist_ok=True)
        
        # ルートロガーを設定（既存のロガー設定を階層化）
        self._setup_root_logger(log_level)
        
        # 現在のロガーを初期化
        self.current_logger = self.root_logger

    def _setup_root_logger(self, log_level):
        """ルートロガーを設定（既存のロガー設定をベースに）"""
        self.root_logger = logging.getLogger(self.base_logger_name)
        self.root_logger.setLevel(log_level)
        
        # 既存のハンドラーを削除
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        # フォーマッター（コミット番号を含む）
        formatter = logging.Formatter(
            f'%(asctime)s - %(name)s - %(levelname)s - [{self.git_commit_hash}] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.root_logger.addHandler(console_handler)
        
        # ファイルハンドラー（常に作成）
        if not self.output_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="ccw_logs_")
            self.log_file_path = os.path.join(self.temp_dir, "simulation.log")
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            self.log_file_path = os.path.join(self.output_dir, "simulation.log")
        
        if self.log_file_path:
            file_mode = 'a' if self.append_logs else 'w'
            file_handler = logging.FileHandler(self.log_file_path, mode=file_mode, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.root_logger.addHandler(file_handler)
            
            self.root_logger.info(f"ログファイル作成: {self.log_file_path}")
    
    def _add_nested_run_file_handler(self, run_name, nested_output_dir):
        """ネストラン専用のファイルハンドラーを追加する。"""
        if not nested_output_dir:
            return None
        
        run_path = self._get_current_run_path()
        if not run_path:
            return None
        
        try:
            # ネストラン専用のログファイルパスを作成
            os.makedirs(nested_output_dir, exist_ok=True)
            log_file_name = f"{run_name}.log"
            nested_log_file_path = os.path.join(nested_output_dir, log_file_name)
            
            # フォーマッター（ネストラン専用）
            formatter = logging.Formatter(
                f'%(asctime)s - %(name)s - %(levelname)s - [{self.git_commit_hash}] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # ファイルハンドラーを作成
            file_mode = 'a' if self.append_logs else 'w'
            file_handler = logging.FileHandler(nested_log_file_path, mode=file_mode, encoding='utf-8')
            file_handler.setLevel(self.root_logger.level)
            file_handler.setFormatter(formatter)
            
            # 現在のロガーに追加（伝播はしないので、このロガー専用）
            self.current_logger.addHandler(file_handler)
            self.current_logger.propagate = False  # ルートロガーへの伝播を停止
            
            # 管理用辞書に保存
            self.run_file_handlers[run_path] = file_handler
            self.run_log_files[run_path] = nested_log_file_path
            self.current_logger.info(f"ネストラン専用ログファイル作成: {nested_log_file_path}")
            return file_handler
            
        except Exception as exc:
            self.root_logger.error(f"ネストラン専用ログファイル作成エラー: {exc}")
            return None
        
    def _remove_nested_run_file_handler(self, run_path):
        """ネストラン専用のファイルハンドラーを削除する。"""
        if run_path in self.run_file_handlers:
            try:
                file_handler = self.run_file_handlers[run_path]
                
                # ハンドラーをフラッシュして閉じる
                file_handler.flush()
                file_handler.close()
                
                # 現在のロガーから削除
                if file_handler in self.current_logger.handlers:
                    self.current_logger.removeHandler(file_handler)
                
                # 管理用辞書から削除
                del self.run_file_handlers[run_path]
                
                # ログファイルパスも記録しておく（アーティファクトアップロード用）
                log_file_path = self.run_log_files.get(run_path)
                if log_file_path:
                    self.root_logger.debug(f"ネストラン専用ログファイル終了: {log_file_path}")
                
                # ルートロガーへの伝播を再開
                self.current_logger.propagate = True
                
            except Exception as e:
                self.root_logger.error(f"ネストラン専用ログファイル削除エラー: {e}")


    def _push_run(self, run_name):
        """ランをスタックにプッシュし、新しい階層ロガーを作成する。"""
        self.run_stack.append(run_name)
        self.current_logger_name = ".".join([self.base_logger_name] + self.run_stack)
        
        # 新しい階層ロガーを作成（ハンドラーなし、ルートロガーに伝播）
        self.current_logger = logging.getLogger(self.current_logger_name)
        self.current_logger.setLevel(self.root_logger.level)
        
        # ハンドラーを持たず、ルートロガーに伝播
        self.current_logger.handlers = []
        self.current_logger.propagate = True
        
        self.root_logger.debug(f"階層ロガー作成: {self.current_logger_name}")
        return self.current_logger_name
    
    def _pop_run(self):
        """ランをスタックからポップし、親ロガーに戻る。"""
        if self.run_stack:
            ended_run = self.run_stack.pop()
            
            run_path = "/".join(self.run_stack + [ended_run])
            
            # ネストラン専用ファイルハンドラーを削除
            self._remove_nested_run_file_handler(run_path)
            
            if self.run_stack:
                # 親の階層ロガーに戻る
                self.current_logger_name = ".".join([self.base_logger_name] + self.run_stack)
                self.current_logger = logging.getLogger(self.current_logger_name)
            else:
                # ルートロガーに戻る
                self.current_logger_name = self.base_logger_name
                self.current_logger = self.root_logger
            
            self.root_logger.debug(f"階層ロガー終了: {ended_run}, 現在: {self.current_logger_name}")
            return ended_run
        else:
            self.root_logger.debug("[_pop_run]: ポップするランがありません")
            return None
        
           
    def _get_current_run_path(self):
        """現在のランの階層パスを取得 (例: "run1/run2/run3")"""
        if not self.run_stack:
            return None
        return "/".join(self.run_stack)

    def _get_current_output_dir(self):
        """現在のランコンテキストに応じた出力ディレクトリを取得"""
        if not self.base_output_dir:
            return None
        
        # ネストランがある場合は階層構造でディレクトリを作成
        if self.run_stack:
            nested_path = os.path.join(self.base_output_dir, *self.run_stack)
            return nested_path
        else:
            return self.base_output_dir


    def start_run(self, run_name=None, nested=False, run_id=None, create_log_file=True):
        """Start a local main or nested logging context."""
        # 階層ロガー管理
        if nested and run_name:
            # ネストランの場合
            self._push_run(run_name)
            run_path = self._get_current_run_path()
            
            # ローカルログ用の辞書にエントリ作成
            if run_path not in self.nested_runs:
                self.nested_runs[run_path] = {'metrics': {}, 'params': {}, 'tags': {}}
            
            self.output_dir = self._get_current_output_dir()
            
            self.current_logger.info(f"ネストラン開始: {run_name}")
            if self.output_dir:
                self.current_logger.info(f"ネストラン出力ディレクトリ: {self.output_dir}")
                os.makedirs(self.output_dir, exist_ok=True)
                
                # ネストラン専用ログファイルを作成（リクエストされた場合）
                if create_log_file:
                    self._add_nested_run_file_handler(run_name, self.output_dir)
                    
        elif not nested:
            # メインランの場合はスタックをクリア
            self.run_stack = []
            self.current_logger_name = self.base_logger_name
            self.current_logger = self.root_logger
            self.output_dir = self.base_output_dir
            self.current_logger.info(f"メインラン開始: {run_name or 'main'}")
        
        if run_name:
            self.current_logger.info(f"ラン開始: {run_name}")
        
        return RunContext(self, nested, run_name)

    def end_run(self):
        """End the current local logging context."""
        if self.run_stack:
            ended_run = self._pop_run()
            self.output_dir = self._get_current_output_dir()
            self.current_logger.info(f"ランネスト終了: {ended_run}")
            if self.output_dir:
                self.current_logger.info(f"出力ディレクトリを戻す: {self.output_dir}")
        else:
            self.root_logger.debug("Root Logger を終了します")
       
    def log_artifact(self, local_path, artifact_path=None, copy_local=False, run_id=None):
        """Optionally copy one artifact into the current local run directory."""
        if copy_local:
            current_output_dir = self._get_current_output_dir()
            if current_output_dir and os.path.exists(local_path):
                artifacts_dir = os.path.join(current_output_dir, "artifacts")
                os.makedirs(artifacts_dir, exist_ok=True)
                
                filename = os.path.basename(local_path)
                if artifact_path:
                    target_dir = os.path.join(artifacts_dir, artifact_path)
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, filename)
                else:
                    target_path = os.path.join(artifacts_dir, filename)
                
                shutil.copy2(local_path, target_path)
                self.current_logger.debug(f"アーティファクトをローカルにコピー: {target_path}")

    def log_artifacts(self, local_dir, artifact_path=None, copy_local=False, run_id=None):
        """Optionally copy an artifact directory into the current local run."""
        if copy_local:
            current_output_dir = self._get_current_output_dir()
            if current_output_dir and os.path.exists(local_dir):
                artifacts_dir = os.path.join(current_output_dir, "artifacts")
                if artifact_path:
                    target_dir = os.path.join(artifacts_dir, artifact_path)
                else:
                    target_dir = artifacts_dir
                
                os.makedirs(target_dir, exist_ok=True)
                
                for item in os.listdir(local_dir):
                    source = os.path.join(local_dir, item)
                    target = os.path.join(target_dir, item)
                    if os.path.isfile(source):
                        shutil.copy2(source, target)
                    elif os.path.isdir(source):
                        shutil.copytree(source, target, dirs_exist_ok=True)
                
                self.current_logger.debug(f"アーティファクト群をローカルにコピー: {target_dir}")

    def log_params(self, params):
        """Record parameters locally."""
        local_params = {}
        for key, value in params.items():
            try:
                converted_value = self._convert_numpy_types(value)
                if isinstance(converted_value, (dict, list)):
                    local_params[key] = str(converted_value)
                elif isinstance(converted_value, (int, float, bool, str)):
                    local_params[key] = converted_value
                else:
                    local_params[key] = str(converted_value)
            except Exception as e:
                self.current_logger.debug(f"パラメータ変換エラー ({key}): {e}")
                local_params[key] = str(value)[:100]
        for key, value in local_params.items():
            self._log_to_local('params', key, value)
            
    def log_metrics(self, metrics):
        """Record metrics locally."""
        for key, value in metrics.items():
            self._log_to_local('metrics', key, value)

    def log_metric(self, key, value):
        """Record one metric locally."""
        self._log_to_local('metrics', key, value)
        
    def _log_to_local(self, data_type, key, value, nested_run=None, step=None):
        """ローカルログに記録（階層構造対応）"""
        target_run_path = nested_run or self._get_current_run_path()
        
        try:
            converted_value = self._convert_numpy_types(value)
            
            if target_run_path:
                if target_run_path not in self.nested_runs:
                    self.nested_runs[target_run_path] = {'metrics': {}, 'params': {}, 'tags': {}}
                
                # ステップ付きメトリクスの場合はリスト形式で保存
                if data_type == 'metrics' and step is not None:
                    if key not in self.nested_runs[target_run_path][data_type]:
                        self.nested_runs[target_run_path][data_type][key] = []
                    
                    # (step, value, timestamp) のタプルとして保存
                    self.nested_runs[target_run_path][data_type][key].append({
                        'step': step,
                        'value': converted_value,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    # 通常のメトリクス、パラメータ、タグは単一値として保存
                    self.nested_runs[target_run_path][data_type][key] = converted_value
            else:
                # メインランに記録
                if data_type == 'metrics':
                    if step is not None:
                        # ステップ付きメトリクスはリスト形式
                        if key not in self.local_metrics:
                            self.local_metrics[key] = []
                        
                        self.local_metrics[key].append({
                            'step': step,
                            'value': converted_value,
                            'timestamp': datetime.now().isoformat()
                        })
                    else:
                        # 通常のメトリクス
                        self.local_metrics[key] = converted_value
                elif data_type == 'params':
                    self.local_params[key] = converted_value
                elif data_type == 'tags':
                    self.local_tags[key] = converted_value
        except Exception as exc:
            self.current_logger.debug(f"ローカルログ記録エラー ({key}): {exc}")
            # エラーが発生した場合は文字列化して記録
            safe_value = str(value)[:100]
            
            if target_run_path:
                if target_run_path not in self.nested_runs:
                    self.nested_runs[target_run_path] = {'metrics': {}, 'params': {}, 'tags': {}}
                self.nested_runs[target_run_path][data_type][key] = safe_value
            else:
                if data_type == 'metrics':
                    self.local_metrics[key] = safe_value
                elif data_type == 'params':
                    self.local_params[key] = safe_value
                elif data_type == 'tags':
                    self.local_tags[key] = safe_value
    def _convert_numpy_types(self, value):
        """numpy型をJSON serializable形式に変換"""
        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()
                else:
                    return value.tolist()
            elif hasattr(value, 'item') and hasattr(value, 'size') and value.size == 1:
                return value.item()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, (list, tuple)):
                return [self._convert_numpy_types(item) for item in value]
            elif isinstance(value, dict):
                return {k: self._convert_numpy_types(v) for k, v in value.items()}
            elif callable(value):
                # 関数やメソッドの場合は文字列表現を返す
                return str(value)
            elif hasattr(value, '__module__') and hasattr(value, '__name__'):
                # クラスオブジェクトなどの場合
                return f"{value.__module__}.{value.__name__}"
            else:
                # 基本型やその他のオブジェクト
                return value
        except Exception:
            # 変換に失敗した場合は文字列化
            self.current_logger.debug(f"型変換に失敗したため文字列化: {type(value)} -> {str(value)[:100]}")
            return str(value)

    def save_local_logs(self):
        """ローカルログをファイルに保存"""
        if not os.path.exists(self.local_logs_dir):
            return
        
        timestamp = datetime.now().isoformat()
        
        try:
            # メインランのログを保存
            main_run_data = {
                'run_name': 'main_run',
                'timestamp': timestamp,
                'git_commit_hash': self.git_commit_hash,
                'metrics': self.local_metrics,
                'params': self.local_params,
                'tags': self.local_tags
            }
            
            main_run_path = os.path.join(self.local_logs_dir, "main_run.json")
            with open(main_run_path, 'w', encoding='utf-8') as f:
                json.dump(main_run_data, f, indent=2, ensure_ascii=False, default=str)
            
            # ネストされたランのログを保存
            for nested_run_name, nested_data in self.nested_runs.items():
                try:
                    nested_run_data = {
                        'run_name': nested_run_name,
                        'parent_run': 'main_run',
                        'timestamp': timestamp,
                        'git_commit_hash': self.git_commit_hash,
                        'metrics': nested_data['metrics'],
                        'params': nested_data['params'],
                        'tags': nested_data['tags']
                    }
                    
                    nested_run_path = os.path.join(self.local_logs_dir, f"{nested_run_name}", "params_and_metrics.json")
                    os.makedirs(os.path.dirname(nested_run_path), exist_ok=True)
                    with open(nested_run_path, 'w', encoding='utf-8') as f:
                        json.dump(nested_run_data, f, indent=2, ensure_ascii=False, default=str)
                except Exception as e:
                    self.current_logger.error(f"ネストランログ保存エラー ({nested_run_name}): {e}")
            
            # すべてのメトリクスをCSV形式でも保存
            self._save_local_metrics_csv()
            self._save_local_params_csv()
            
            self.current_logger.info(f"ローカル実行ログを保存: {self.local_logs_dir}")
            
        except Exception as e:
            self.current_logger.error(f"ローカルログ保存エラー: {e}")
            
    def _save_local_metrics_csv(self):
        """メトリクスをCSV形式で保存（step対応版）"""
        try:
            metrics_data = []
            
            # メインランのメトリクス
            for key, value in self.local_metrics.items():
                try:
                    # ステップ付きメトリクス（リスト）の場合
                    if isinstance(value, list):
                        for entry in value:
                            if isinstance(entry, dict):
                                metrics_data.append({
                                    'run_name': 'main_run',
                                    'run_type': 'main',
                                    'metric_name': key,
                                    'metric_value': entry.get('value'),
                                    'step': entry.get('step'),
                                    'timestamp': entry.get('timestamp')
                                })
                    # 通常のメトリクス（単一値）の場合
                    else:
                        metrics_data.append({
                            'run_name': 'main_run',
                            'run_type': 'main',
                            'metric_name': key,
                            'metric_value': value,
                            'step': None,
                            'timestamp': None
                        })
                except Exception as e:
                    self.current_logger.debug(f"メトリクス処理エラー ({key}): {e}")
                    continue
            
            # ネストされたランのメトリクス
            for nested_run_name, nested_data in self.nested_runs.items():
                for key, value in nested_data['metrics'].items():
                    try:
                        # ステップ付きメトリクス（リスト）の場合
                        if isinstance(value, list):
                            for entry in value:
                                if isinstance(entry, dict):
                                    metrics_data.append({
                                        'run_name': nested_run_name,
                                        'run_type': 'nested',
                                        'parent_run': 'main_run',
                                        'metric_name': key,
                                        'metric_value': entry.get('value'),
                                        'step': entry.get('step'),
                                        'timestamp': entry.get('timestamp')
                                    })
                        # 通常のメトリクス（単一値）の場合
                        else:
                            metrics_data.append({
                                'run_name': nested_run_name,
                                'run_type': 'nested',
                                'parent_run': 'main_run',
                                'metric_name': key,
                                'metric_value': value,
                                'step': None,
                                'timestamp': None
                            })
                    except Exception as e:
                        self.current_logger.debug(f"ネストメトリクス処理エラー ({nested_run_name}.{key}): {e}")
                        continue
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_path = os.path.join(self.local_logs_dir, "all_metrics.csv")
                metrics_df.to_csv(metrics_path, index=False)
                self.current_logger.debug(f"メトリクスCSV保存完了: {len(metrics_data)}行")
        except Exception as e:
            self.current_logger.error(f"メトリクスCSV保存エラー: {e}")
        
    def _save_local_params_csv(self):
        """パラメータをCSV形式で保存"""
        try:
            params_data = []
            
            # メインランのパラメータ
            for key, value in self.local_params.items():
                try:
                    params_data.append({
                        'run_name': 'main_run',
                        'run_type': 'main',
                        'param_name': key,
                        'param_value': str(value)
                    })
                except Exception:
                    continue
            
            # ネストされたランのパラメータ
            for nested_run_name, nested_data in self.nested_runs.items():
                for key, value in nested_data['params'].items():
                    try:
                        params_data.append({
                            'run_name': nested_run_name,
                            'run_type': 'nested',
                            'parent_run': 'main_run',
                            'param_name': key,
                            'param_value': str(value)
                        })
                    except Exception:
                        continue
            
            if params_data:
                params_df = pd.DataFrame(params_data)
                params_path = os.path.join(self.local_logs_dir, "all_params.csv")
                params_df.to_csv(params_path, index=False)
        except Exception as e:
            self.current_logger.error(f"パラメータCSV保存エラー: {e}")

    def cleanup(self):
        """一時ファイルのクリーンアップ"""
        # ネストラン専用ファイルハンドラーをすべて閉じる
        for _run_path, file_handler in list(self.run_file_handlers.items()):
            try:
                file_handler.flush()
                file_handler.close()
                if file_handler in self.current_logger.handlers:
                    self.current_logger.removeHandler(file_handler)
            except Exception as e:
                self.root_logger.error(f"ファイルハンドラークリーンアップエラー: {e}")
        
        self.run_file_handlers.clear()
        self.run_log_files.clear()
        
        # 既存のクリーンアップ処理
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.current_logger.debug(f"一時ディレクトリを削除: {self.temp_dir}")
            except Exception as e:
                self.current_logger.error(f"一時ディレクトリの削除に失敗: {e}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        # ローカルログを保存
        self.save_local_logs()
            
        
        self.cleanup()

    # Python標準ロガーメソッドを直接公開
    def info(self, message):
        self.current_logger.info(message)

    def debug(self, message):
        self.current_logger.debug(message)

    def warning(self, message):
        self.current_logger.warning(message)

    def error(self, message):
        self.current_logger.error(message)

    def critical(self, message):
        self.current_logger.critical(message)


def extract_ground_truth_metrics_at_cutoff(ground_truth_result, cutoff_time_of_observation_display):
    """Ground truth結果からcutoff時点のCI情報を抽出"""
    
    metrics = {}
    
    # cutoff_time_of_observation_display時点のデータを抽出
    cutoff_data = ground_truth_result[ground_truth_result['time'] == cutoff_time_of_observation_display]
    
    if not cutoff_data.empty:
        # 各戦略について情報を抽出
        for _, row in cutoff_data.iterrows():
            strategy = row['strategy']
            
            metrics.update({
                f'GT_{strategy}_IR__ci_lower_at_{cutoff_time_of_observation_display}d': row['ci_lower'],
                f'GT_{strategy}_IR_ci_upper_at_{cutoff_time_of_observation_display}d': row['ci_upper'],
                f'GT_{strategy}_IR_mean_at_{cutoff_time_of_observation_display}d': row['incident_rate']  # meanはincident_rateとして記録されている
            })
    
    return metrics


def run_ground_truth_simulation(args, logger):
    """
    Ground truthシミュレーションを実行（ファイル保存・ローカルログ対応）
    simulate.pyとsimulate_batch.pyで共通で使用
    """
    
    logger.info("Ground truthシミュレーション設定を開始")
    
    # 実験設定をロード
    scenario_params, scenario_vars, configs = load_experiment_settings(args.experiment)
    intervention_strategies = {
        key: item(args) for key, item in configs["strategy_creator"].items()
    }
    
    logger.info(f"実験設定ロード完了: {args.experiment}")
    logger.info(f"介入戦略数: {len(intervention_strategies)}")
    
    # パラメータと設定をログ
    try:
        # 引数パラメータをログ
        args_params = {k: v for k, v in vars(args).items() 
                      if not callable(v) and not k.startswith('_')}
        logger.log_params(args_params)
        logger.info("引数パラメータをログ")
        
        # シナリオパラメータをログ（フラット化）
        flattened_scenario_params = flatten_dict(scenario_params)
        logger.log_params(flattened_scenario_params)
        logger.info("シナリオパラメータをログ")
        
        # 設定をログ（安全な形式で）
        safe_config_params = {}
        for k, v in configs.items():
            if isinstance(v, (str, int, float, bool)):
                safe_config_params[k] = v
            elif v is None:
                safe_config_params[k] = "None"
            else:
                safe_config_params[k] = str(type(v).__name__)
        
        logger.log_params(safe_config_params)
        logger.info("設定をログ")
        
        # 介入戦略をログ（簡略化された形式で）
        strategy_params = {
            f"strategy_{key}": str(type(item).__name__) 
            for key, item in intervention_strategies.items()
        }
        logger.log_params(strategy_params)
        logger.info("介入戦略をログ")
        
    except Exception as e:
        logger.error(f"パラメータログ中にエラー: {e}")
        logger.info("パラメータログをスキップして続行")
    
    plot_figures = not getattr(args, 'skip_figures', False)
    
    # Ground truthシミュレーション実行
    sample_size = args.sample_size[0] if isinstance(args.sample_size, list) else args.sample_size
    seed = getattr(args, 'ground_truth_seed', getattr(args, 'seed', 42))
    if isinstance(seed, list):
        seed = seed[0]
    ground_truth_workers = int(getattr(args, "ground_truth_workers", 1) or 1)
    
    logger.info(
        "Ground truthシミュレーション開始 - "
        f"サンプルサイズ: {sample_size}, シード: {seed}, workers: {ground_truth_workers}"
    )

    if ground_truth_workers > 1:
        if plot_figures:
            logger.warning("並列Ground Truthでは図の生成をスキップします")
        result_df, simulated_or, figs = _run_parallel_ground_truth_simulation(
            args=args,
            logger=logger,
            sample_size=int(sample_size),
            seed=int(seed),
        )
    else:
        result_df, simulated_or, figs = simulate_true_effect(
            base_variables=scenario_vars,
            n_time=args.n_time,
            strategies=intervention_strategies,
            treatment_var=configs["treatment_var"],
            cutoff_time_of_observation_display=args.cutoff_time_of_observation_display,
            sample_size=sample_size,
            seed=seed,
            plot_figures=plot_figures,
            logger_=logger.current_logger,
            verbose=args.verbose
        )
    
    logger.info(f"Ground truthシミュレーション完了 - OR: {simulated_or:.4f}")
    
    # 結果のログ開始
    logger.info("結果のログ開始")
    
    # 基本的なGround truthメトリクス
    logger.log_metric("GT_OR", float(simulated_or))
    logger.info(f"Ground truth OR: {simulated_or:.4f}")
    
    # cutoff時点のGround truth情報を抽出してログ
    gt_cutoff_metrics = extract_ground_truth_metrics_at_cutoff(
        result_df, 
        args.cutoff_time_of_observation_display
    )
    
    if gt_cutoff_metrics:
        logger.log_metrics(gt_cutoff_metrics)
        logger.info(f"Ground Truth (時点: {args.cutoff_time_of_observation_display}日)")
        for key, value in gt_cutoff_metrics.items():
            if not pd.isna(value):
                logger.info(f"  {key}: {value:.4f}")
    
    # Ground truth結果をCSVファイルとして保存してログ
    current_output_dir = logger._get_current_output_dir()
    use_temp_dir = current_output_dir is None
    temp_context = tempfile.TemporaryDirectory() if use_temp_dir else nullcontext()
    with temp_context as temp_dir:
        # 作業ディレクトリを決定
        working_dir = temp_dir if use_temp_dir else current_output_dir
        gt_result_path = os.path.join(working_dir, "ground_truth_result.csv")
        
        # CSV保存とアーティファクトログ（1箇所のみ）
        result_df.to_csv(gt_result_path, index=False)
        logger.log_artifact(gt_result_path, artifact_path="ground_truth_results")
    
    
    strategies = result_df['strategy'].unique()
    for strategy in strategies:
        res = result_df[result_df['strategy'] == strategy]
        if not res.empty:
            nested_run_name = f"ground_truth_{strategy}"
            
            with logger.start_run(run_name=nested_run_name, nested=True):
                logger.debug(f"戦略 {strategy} のメトリクス処理開始")
                
                # cutoff時点の主要メトリクスを記録
                cutoff_data = res[res['time'] == args.cutoff_time_of_observation_display]
                if not cutoff_data.empty:
                    row = cutoff_data.iloc[0]
                    strategy_metrics = {
                        f"{strategy}_incident_rate_at_{args.cutoff_time_of_observation_display}d": float(row['incident_rate']),
                        f"{strategy}_ci_lower_at_{args.cutoff_time_of_observation_display}d": float(row['ci_lower']),
                        f"{strategy}_ci_upper_at_{args.cutoff_time_of_observation_display}d": float(row['ci_upper'])
                    }
                    logger.log_metrics(strategy_metrics)
                    
    # 図を保存してログ
    if figs:
        logger.info("図のログ開始")
        
        for strat, fig_dict in figs.items():
            nested_run_name = f"ground_truth_{strat}"
            
            with logger.start_run(run_name=nested_run_name, nested=True):
                # 出力ディレクトリの決定（存在しない場合は一時ディレクトリ）
                current_output_dir = logger._get_current_output_dir()
                use_temp_dir = current_output_dir is None
                
                # 一時ディレクトリのコンテキストマネージャー（必要な場合のみ）
                temp_context = tempfile.TemporaryDirectory() if use_temp_dir else nullcontext()
                
                with temp_context as temp_dir:
                    # 作業ディレクトリを決定
                    working_dir = temp_dir if use_temp_dir else current_output_dir
                    
                    # ディレクトリパスの設定
                    images_dir = os.path.join(working_dir, "images")
                    figures_dir = os.path.join(working_dir, "figures")
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(figures_dir, exist_ok=True)
                    
                    images_saved = False
                    figures_saved = False
                    
                    # 図を保存
                    for key, fig_list in fig_dict.items():
                        for idx, fig in enumerate(fig_list):
                            if isinstance(fig, plt.Figure):
                                # PNG画像として保存
                                fig_path = os.path.join(images_dir, f"{strat}_{key}_{idx}.png")
                                fig.savefig(fig_path, bbox_inches='tight', dpi=150)
                                plt.close(fig)  # メモリ解放
                                images_saved = True
                                logger.debug(f"画像保存: {fig_path}")
                            
                            elif hasattr(fig, 'write_html'):
                                # HTML図として保存
                                fig_path = os.path.join(figures_dir, f"{strat}_{key}_{idx}.html")
                                fig.write_html(fig_path)
                                figures_saved = True
                                logger.debug(f"HTML図保存: {fig_path}")
                            
                            else:
                                logger.warning(f"不明な図オブジェクト {type(fig)} をスキップします")
                    
                    # 一時ディレクトリを使った場合のみ、実行ディレクトリへコピーする。
                    copy_local = use_temp_dir
                    
                    if images_saved:
                        logger.log_artifacts(images_dir, artifact_path="images", copy_local=copy_local)
                        logger.debug(f"画像ディレクトリを保存: {images_dir}")
                    
                    if figures_saved:
                        logger.log_artifacts(figures_dir, artifact_path="figures", copy_local=copy_local)
                        logger.debug(f"HTML図ディレクトリを保存: {figures_dir}")
                
                logger.info(f"戦略 {strat} の図をログ完了")

   
    # サマリーメトリクスをログ
    summary_metrics = {
        "n_strategies": len(strategies),
        "simulation_sample_size": int(sample_size),
        "simulation_seed": int(seed),
        "ground_truth_workers": int(ground_truth_workers),
        "n_time_points": len(result_df['time'].unique())
    }
    logger.log_metrics(summary_metrics)
    
    logger.info("結果ログ完了")
    
    return result_df, simulated_or, figs, scenario_params, scenario_vars, configs, intervention_strategies
