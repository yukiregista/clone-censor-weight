#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/paper/1_experiment.sh --experiment NAME --cutoff_time_of_intervention N [N ...] --n_workers N --coefA {0.5,1,2,4,8} --coefD {0.5,1,2,4,8} [options]

Mandatory:
  --experiment NAME
  --cutoff_time_of_intervention N [N ...]
  --n_workers N
  --coefA {0.5,1,2,4,8}
  --coefD {0.5,1,2,4,8}
Optional:
  --sample_sizes N [N ...]          Default: 100 1000 10000
  --n_runs N                        Default: 1000
  --ground_truth_sample_size N      Default: 10000000
  --bootstrap_B N                   Default: 1000
  --n_time N                        Default: 32
  --cutoff_time_of_observation N    Default: 31
  --cutoff_time_of_observation_display N  Default: 30
  --no-bootstrap
  --force

Notes:
  - coefA/coefD に応じて scripts/paper/config_overrides/ の設定を使用します。
  - --n_workers は分析runとGround Truthチャンク生成の両方に使います。
  - 出力は setting/experiment/sample/cutoff ごとに分割され、最後に timestamp を付与します。

Examples:
  scripts/paper/1_experiment.sh \
    --experiment experimentA \
    --cutoff_time_of_intervention 0 2 4 \
    --n_workers 5 \
    --coefA 1 --coefD 1
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

# --- Mandatory args ---
EXPERIMENT=""
CUTOFF_TIMES=()
N_WORKERS=""
COEF_A=""
COEF_D=""
FORCE_RUN=0
SAMPLE_SIZES=(100 1000 10000)
N_RUNS=1000
GROUND_TRUTH_SAMPLE_SIZE=10000000
BOOTSTRAP_B=1000
N_TIME=32
CUTOFF_TIME_OF_OBSERVATION=31
CUTOFF_TIME_OF_OBSERVATION_DISPLAY=30
ENABLE_BOOTSTRAP=1

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT="${2:-}"; shift 2;;
    --cutoff_time_of_intervention)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CUTOFF_TIMES+=("$1")
        shift
      done
      ;;
    --n_workers) N_WORKERS="${2:-}"; shift 2;;
    --coefA) COEF_A="${2:-}"; shift 2;;
    --coefD) COEF_D="${2:-}"; shift 2;;
    --sample_sizes)
      SAMPLE_SIZES=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SAMPLE_SIZES+=("$1")
        shift
      done
      ;;
    --n_runs) N_RUNS="${2:-}"; shift 2;;
    --ground_truth_sample_size) GROUND_TRUTH_SAMPLE_SIZE="${2:-}"; shift 2;;
    --bootstrap_B) BOOTSTRAP_B="${2:-}"; shift 2;;
    --n_time) N_TIME="${2:-}"; shift 2;;
    --cutoff_time_of_observation) CUTOFF_TIME_OF_OBSERVATION="${2:-}"; shift 2;;
    --cutoff_time_of_observation_display) CUTOFF_TIME_OF_OBSERVATION_DISPLAY="${2:-}"; shift 2;;
    --no-bootstrap) ENABLE_BOOTSTRAP=0; shift 1;;
    --force) FORCE_RUN=1; shift 1;;

    -h|--help) usage; exit 0;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

# --- Validate mandatory args ---
[[ -n "$EXPERIMENT" ]] || die "--experiment is required"
[[ ${#CUTOFF_TIMES[@]} -gt 0 ]] || die "--cutoff_time_of_intervention is required"
[[ -n "$N_WORKERS" ]] || die "--n_workers is required"
[[ -n "$COEF_A" ]] || die "--coefA is required"
[[ -n "$COEF_D" ]] || die "--coefD is required"
[[ ${#SAMPLE_SIZES[@]} -gt 0 ]] || die "--sample_sizes requires at least one value"

for numeric_value in "$N_WORKERS" "$N_RUNS" "$GROUND_TRUTH_SAMPLE_SIZE" "$BOOTSTRAP_B" "$N_TIME" "$CUTOFF_TIME_OF_OBSERVATION" "$CUTOFF_TIME_OF_OBSERVATION_DISPLAY" "${SAMPLE_SIZES[@]}"; do
  [[ "$numeric_value" =~ ^[0-9]+$ ]] || die "numeric settings must be non-negative integers"
done

# Treat as categorical strings
[[ "$COEF_A" =~ ^(0\.5|1|2|4|8)$ ]] || die "--coefA must be one of {0.5,1,2,4,8}"
[[ "$COEF_D" =~ ^(0\.5|1|2|4|8)$ ]] || die "--coefD must be one of {0.5,1,2,4,8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Run label (filesystem-safe) ---
RUN_LABEL="a${COEF_A}d${COEF_D}"

# --- Config override directory ---
CONFIG_DIR_LABEL="${RUN_LABEL}"
if [[ "${RUN_LABEL}" == "a1d1" ]]; then
  CONFIG_DIR_LABEL="setting1"
fi
CONFIG_DIR_REL="scripts/paper/config_overrides/${CONFIG_DIR_LABEL}"
CONFIG_DIR="${REPO_ROOT}/${CONFIG_DIR_REL}"
if [[ ! -d "$CONFIG_DIR" ]]; then
  die "Config override dir not found: ${CONFIG_DIR}"
fi

echo "Running ${RUN_LABEL} with config_dir ${CONFIG_DIR_REL}"
echo "Repo: ${REPO_ROOT}"
echo "Sample sizes: ${SAMPLE_SIZES[*]}"
echo "Cutoffs: ${CUTOFF_TIMES[*]}"

(
  cd "$REPO_ROOT"
  echo "Syncing uv environment..."
  uv sync --frozen --no-dev --extra research --quiet
)

for CUTOFF_TIME_OF_INTERVENTION in "${CUTOFF_TIMES[@]}"; do
for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"; do
  BASE_REL="output/experiments/${RUN_LABEL}/${EXPERIMENT}/N${SAMPLE_SIZE}_cut${CUTOFF_TIME_OF_INTERVENTION}"
  BASE_DIR="${REPO_ROOT}/${BASE_REL}"
  GT_CACHE_REL="output/experiments/${RUN_LABEL}/${EXPERIMENT}/ground_truth_cache_cut${CUTOFF_TIME_OF_INTERVENTION}"
  GT_CACHE_DIR="${REPO_ROOT}/${GT_CACHE_REL}"
  CURRENT_FILE="${BASE_DIR}/current.txt"
  ARCHIVE_DIR="${BASE_DIR}/archive"
  mkdir -p "${BASE_DIR}"
  mkdir -p "${GT_CACHE_DIR}"

  CURRENT_REL=""
  if [[ -f "${CURRENT_FILE}" ]]; then
    CURRENT_REL="$(cat "${CURRENT_FILE}")"
  fi

  if [[ -n "${CURRENT_REL}" ]]; then
    RUN_REL="${BASE_REL}/${CURRENT_REL}"
    RUN_DIR="${REPO_ROOT}/${RUN_REL}"
  else
    RUN_REL=""
    RUN_DIR=""
  fi

  if [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/simulation_complete.txt" && "${FORCE_RUN}" -eq 0 ]]; then
    echo "SKIP (complete): ${RUN_DIR}"
    continue
  fi

  if [[ -n "${RUN_DIR}" && "${FORCE_RUN}" -eq 1 ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${ARCHIVE_DIR}"
    ARCHIVE_REL="${BASE_REL}/archive/${CURRENT_REL}__forced_${timestamp}"
    ARCHIVE_PATH="${REPO_ROOT}/${ARCHIVE_REL}"
    echo "FORCE: moving existing run to ${ARCHIVE_PATH}"
    mv "${RUN_DIR}" "${ARCHIVE_PATH}"
    RUN_REL=""
    RUN_DIR=""
    echo "" > "${CURRENT_FILE}"
  fi

  if [[ -z "${RUN_REL}" ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    RUN_REL="${BASE_REL}/${timestamp}"
    RUN_DIR="${REPO_ROOT}/${RUN_REL}"
    mkdir -p "${RUN_DIR}"
    echo "${timestamp}" > "${CURRENT_FILE}"
  else
    mkdir -p "${RUN_DIR}"
    timestamp="${CURRENT_REL}"
    echo "RESUME: ${RUN_DIR}"
  fi

  # --- Metadata ---
  {
    echo "experiment=${EXPERIMENT}"
    echo "sample_size=${SAMPLE_SIZE}"
    echo "cutoff_time_of_intervention=${CUTOFF_TIME_OF_INTERVENTION}"
    echo "n_workers=${N_WORKERS}"
    echo "ground_truth_workers=${N_WORKERS}"
    echo "coefA=${COEF_A}"
    echo "coefD=${COEF_D}"
    echo "run_label=${RUN_LABEL}"
    echo "config_dir=${CONFIG_DIR_REL}"
    echo "ground_truth_cache_dir=${GT_CACHE_REL}"
    echo "outdir=${RUN_REL}"
    echo "timestamp=${timestamp}"
    echo "n_runs=${N_RUNS}"
    echo "ground_truth_sample_size=${GROUND_TRUTH_SAMPLE_SIZE}"
    echo "bootstrap=${ENABLE_BOOTSTRAP}"
    echo "bootstrap_B=${BOOTSTRAP_B}"
  } > "${RUN_DIR}/_run_meta.txt"

  echo "------------------------------------------------------------"
  echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
  echo "Output: ${RUN_DIR}"
  echo "------------------------------------------------------------"

  (
    cd "$REPO_ROOT"

    BOOTSTRAP_ARGS=()
    if [[ "${ENABLE_BOOTSTRAP}" -eq 1 ]]; then
      BOOTSTRAP_ARGS=(--bootstrap --bootstrap_B "${BOOTSTRAP_B}" --bootstrap_conf 0.95)
    fi

    uv run python scripts/paper/simulate_batch.py \
      --experiment "${EXPERIMENT}" \
      --sample_size "${SAMPLE_SIZE}" \
      --ground_truth_sample_size "${GROUND_TRUTH_SAMPLE_SIZE}" \
      --ground_truth_workers "${N_WORKERS}" \
      --n_runs "${N_RUNS}" \
      --n_time "${N_TIME}" \
      --cutoff_time_of_intervention "${CUTOFF_TIME_OF_INTERVENTION}" \
      --cutoff_time_of_observation "${CUTOFF_TIME_OF_OBSERVATION}" \
      --cutoff_time_of_observation_display "${CUTOFF_TIME_OF_OBSERVATION_DISPLAY}" \
      --n_workers "${N_WORKERS}" \
      --base_seed 1000 \
      --skip_figures \
      "${BOOTSTRAP_ARGS[@]}" \
      --output_dir "${RUN_REL}" \
      --config_dir "${CONFIG_DIR}" \
      --ground_truth_cache_dir "${GT_CACHE_REL}" \
      --save_ipw_weights

    uv run python scripts/paper/2_run_ipcw_diagnostics.py "${RUN_REL}" \
      --patterns VAR HPREV2 \
      --grace-period "${CUTOFF_TIME_OF_INTERVENTION}"
  )
done
done

echo "Done."
