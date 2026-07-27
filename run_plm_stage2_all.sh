#!/usr/bin/env bash
set -uo pipefail

# One-click stage-2 training for every completed stage-1 fold/seed.
#
# Optional environment overrides:
#   STAGE2_PYTHON=/path/to/python
#   STAGE2_CONFIG=configs/plm_stage2.yaml
#   STAGE2_MODEL_ROOT=trained_model/PLM_Unified
#   STAGE2_DEVICE=cuda
#   STAGE2_SWANLAB_MODE=online
#   STAGE2_NUM_WORKERS=2
#   STAGE2_FOLDS=1,3,5             # Default: all available folds
#   STAGE2_SEEDS=42,2026           # Default: all available seeds
#   STAGE2_LIMIT=32                 # Debug only
#   STAGE2_FORCE_RESTART=1          # Ignore an incomplete stage-2 checkpoint
#   STAGE2_STOP_ON_ERROR=1          # Stop instead of continuing after a failure
#   DRY_RUN=1                       # Print commands without training
#
# Command-line examples:
#   ./run_plm_stage2_all.sh --folds 1,3,5 --seeds 42,2026
#   ./run_plm_stage2_all.sh --folds 1 --seeds 3407 --dry-run

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${STAGE2_PYTHON:-/home/mjp/miniconda3/envs/unifyimmun/bin/python}"
CONFIG="${STAGE2_CONFIG:-configs/plm_stage2.yaml}"
MODEL_ROOT="${STAGE2_MODEL_ROOT:-trained_model/PLM_Unified}"
DEVICE="${STAGE2_DEVICE:-cuda}"
SWANLAB_MODE="${STAGE2_SWANLAB_MODE:-online}"
NUM_WORKERS="${STAGE2_NUM_WORKERS:-2}"
FOLDS_CSV="${STAGE2_FOLDS:-all}"
SEEDS_CSV="${STAGE2_SEEDS:-all}"
LIMIT="${STAGE2_LIMIT:-}"
FORCE_RESTART="${STAGE2_FORCE_RESTART:-0}"
STOP_ON_ERROR="${STAGE2_STOP_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_SCRIPT="source/train_plm_stage2.py"
EXPORT_SCRIPT="source/export_plm_stage2_task_models.py"
LOG_ROOT="${MODEL_ROOT}/stage2_logs"
NVML_COMPAT_DIR="/home/mjp/.local/lib/nvidia-595.71.05"

# A driver package update can leave the old kernel module active until reboot.
# Use the matching private NVML only for that transient state; after reboot the
# normal system nvidia-smi succeeds and this branch is skipped automatically.
if [[ "${DEVICE}" == cuda* ]] \
    && command -v nvidia-smi >/dev/null 2>&1 \
    && ! nvidia-smi -L >/dev/null 2>&1 \
    && [[ -x "${PROJECT_ROOT}/nvidia-smi-compatible.sh" ]] \
    && "${PROJECT_ROOT}/nvidia-smi-compatible.sh" -L >/dev/null 2>&1; then
    export LD_LIBRARY_PATH="${NVML_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "[NVML] Using temporary 595.71.05 compatibility library; reboot is still recommended."
fi

usage() {
    cat <<'EOF'
Usage:
  ./run_plm_stage2_all.sh [options]

Options:
  --folds CSV          Fold IDs, for example: 1,3,5 (default: all)
  --seeds CSV          Seed IDs, for example: 42,2026 (default: all)
  --device DEVICE      Training device (default: cuda)
  --swanlab-mode MODE  online, local, offline, or disabled
  --num-workers N      DataLoader workers
  --limit N            Debug only: rows per split
  --force-restart      Ignore an incomplete Stage-2 checkpoint
  --stop-on-error      Stop after the first failed run
  --dry-run            Print selected commands without training
  -h, --help           Show this help

Examples:
  ./run_plm_stage2_all.sh --folds 1 --seeds 42
  ./run_plm_stage2_all.sh --folds 2,3,4,5 --seeds 42,2026
  ./run_plm_stage2_all.sh --folds 1 --seeds 3407 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --folds requires a value" >&2
                exit 2
            }
            FOLDS_CSV="$2"
            shift 2
            ;;
        --seeds)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --seeds requires a value" >&2
                exit 2
            }
            SEEDS_CSV="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --device requires a value" >&2
                exit 2
            }
            DEVICE="$2"
            shift 2
            ;;
        --swanlab-mode)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --swanlab-mode requires a value" >&2
                exit 2
            }
            SWANLAB_MODE="$2"
            shift 2
            ;;
        --num-workers)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --num-workers requires a value" >&2
                exit 2
            }
            NUM_WORKERS="$2"
            shift 2
            ;;
        --limit)
            [[ $# -ge 2 ]] || {
                echo "[ERROR] --limit requires a value" >&2
                exit 2
            }
            LIMIT="$2"
            shift 2
            ;;
        --force-restart)
            FORCE_RESTART=1
            shift
            ;;
        --stop-on-error)
            STOP_ON_ERROR=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG}" >&2
    exit 2
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] Training entry point not found: ${TRAIN_SCRIPT}" >&2
    exit 2
fi
if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
    echo "[ERROR] Task-checkpoint export entry point not found: ${EXPORT_SCRIPT}" >&2
    exit 2
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "[ERROR] Model root not found: ${MODEL_ROOT}" >&2
    exit 2
fi
for flag_name in FORCE_RESTART STOP_ON_ERROR DRY_RUN; do
    flag_value="${!flag_name}"
    if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
        echo "[ERROR] ${flag_name} must be 0 or 1, got: ${flag_value}" >&2
        exit 2
    fi
done
if [[ ! "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] STAGE2_NUM_WORKERS must be a non-negative integer" >&2
    exit 2
fi
if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] STAGE2_LIMIT must be a positive integer" >&2
    exit 2
fi
if [[ ! "${SWANLAB_MODE}" =~ ^(online|local|offline|disabled)$ ]]; then
    echo "[ERROR] SwanLab mode must be online, local, offline, or disabled" >&2
    exit 2
fi

parse_id_filter() {
    local value="$1"
    local label="$2"
    local -n destination="$3"
    destination=()
    if [[ "${value}" == "all" ]]; then
        return
    fi
    if [[ -z "${value}" ]]; then
        echo "[ERROR] ${label} filter cannot be empty" >&2
        exit 2
    fi
    IFS=',' read -ra destination <<< "${value}"
    for item in "${destination[@]}"; do
        if [[ ! "${item}" =~ ^[1-9][0-9]*$ ]]; then
            echo "[ERROR] Invalid ${label} ID: ${item}" >&2
            exit 2
        fi
    done
}

matches_filter() {
    local value="$1"
    shift
    local candidates=("$@")
    if [[ ${#candidates[@]} -eq 0 ]]; then
        return 0
    fi
    for candidate in "${candidates[@]}"; do
        if [[ "${value}" == "${candidate}" ]]; then
            return 0
        fi
    done
    return 1
}

FOLD_FILTER=()
SEED_FILTER=()
parse_id_filter "${FOLDS_CSV}" "fold" FOLD_FILTER
parse_id_filter "${SEEDS_CSV}" "seed" SEED_FILTER

mapfile -t ALL_STAGE1_CHECKPOINTS < <(
    find "${MODEL_ROOT}" \
        -mindepth 3 \
        -maxdepth 3 \
        -type f \
        -name "last.pt" \
        -path "*/fold_*/seed_*/last.pt" \
        | sort
)

if [[ ${#ALL_STAGE1_CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[ERROR] No fold_*/seed_*/last.pt checkpoints found in ${MODEL_ROOT}" >&2
    exit 2
fi

STAGE1_CHECKPOINTS=()
for checkpoint in "${ALL_STAGE1_CHECKPOINTS[@]}"; do
    candidate_run_dir="$(dirname "${checkpoint}")"
    candidate_seed_dir="$(basename "${candidate_run_dir}")"
    candidate_fold_dir="$(basename "$(dirname "${candidate_run_dir}")")"
    if [[ ! "${candidate_fold_dir}" =~ ^fold_([0-9]+)$ ]]; then
        continue
    fi
    candidate_fold="${BASH_REMATCH[1]}"
    if [[ ! "${candidate_seed_dir}" =~ ^seed_([0-9]+)$ ]]; then
        continue
    fi
    candidate_seed="${BASH_REMATCH[1]}"
    if matches_filter "${candidate_fold}" "${FOLD_FILTER[@]}" \
        && matches_filter "${candidate_seed}" "${SEED_FILTER[@]}"; then
        STAGE1_CHECKPOINTS+=("${checkpoint}")
    fi
done

if [[ ${#STAGE1_CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[ERROR] No Stage-1 checkpoints match folds=${FOLDS_CSV} seeds=${SEEDS_CSV}" >&2
    exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${LOG_ROOT}"
fi

echo "============================================================"
echo " Unified PLM Stage 2: all completed Stage-1 runs"
echo "============================================================"
echo "Project root : ${PROJECT_ROOT}"
echo "Config       : ${CONFIG}"
echo "Model root   : ${MODEL_ROOT}"
echo "Python       : ${PYTHON_BIN}"
echo "Device       : ${DEVICE}"
echo "SwanLab mode : ${SWANLAB_MODE}"
echo "Workers      : ${NUM_WORKERS}"
echo "Fold filter  : ${FOLDS_CSV}"
echo "Seed filter  : ${SEEDS_CSV}"
echo "Runs selected: ${#STAGE1_CHECKPOINTS[@]}"
echo "Auto resume  : $([[ "${FORCE_RESTART}" == "1" ]] && echo no || echo yes)"
echo "Dry run      : ${DRY_RUN}"
echo "============================================================"

SUCCEEDED=()
SKIPPED=()
FAILED=()
PLANNED=()

checkpoint_stage() {
    local checkpoint="$1"
    "${PYTHON_BIN}" - "${checkpoint}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

if checkpoint.get("checkpoint_type") != "stage2_training":
    raise SystemExit("not_stage2_training")
print(checkpoint.get("stage", "unknown"))
PY
}

for stage1_checkpoint in "${STAGE1_CHECKPOINTS[@]}"; do
    run_dir="$(dirname "${stage1_checkpoint}")"
    seed_dir="$(basename "${run_dir}")"
    fold_dir="$(basename "$(dirname "${run_dir}")")"

    if [[ ! "${fold_dir}" =~ ^fold_([0-9]+)$ ]]; then
        echo "[ERROR] Cannot parse fold from ${stage1_checkpoint}" >&2
        FAILED+=("${stage1_checkpoint}")
        continue
    fi
    fold="${BASH_REMATCH[1]}"
    if [[ ! "${seed_dir}" =~ ^seed_([0-9]+)$ ]]; then
        echo "[ERROR] Cannot parse seed from ${stage1_checkpoint}" >&2
        FAILED+=("${stage1_checkpoint}")
        continue
    fi
    seed="${BASH_REMATCH[1]}"
    run_label="fold_${fold}/seed_${seed}"
    log_path="${LOG_ROOT}/fold_${fold}_seed_${seed}.log"

    shopt -s nullglob
    stage2_candidates=("${run_dir}"/stage2_last_*.pt)
    shopt -u nullglob
    latest_stage2=""
    if [[ ${#stage2_candidates[@]} -gt 0 ]]; then
        latest_stage2="${stage2_candidates[${#stage2_candidates[@]}-1]}"
    fi

    command=(
        "${PYTHON_BIN}"
        "-u"
        "${TRAIN_SCRIPT}"
        "--config" "${CONFIG}"
        "--fold" "${fold}"
        "--seed" "${seed}"
        "--device" "${DEVICE}"
        "--num-workers" "${NUM_WORKERS}"
        "--swanlab-mode" "${SWANLAB_MODE}"
    )

    if [[ -n "${latest_stage2}" && "${FORCE_RESTART}" != "1" ]]; then
        if ! stage="$(checkpoint_stage "${latest_stage2}")"; then
            echo "[ERROR] Cannot inspect ${latest_stage2}" >&2
            FAILED+=("${run_label}")
            if [[ "${STOP_ON_ERROR}" == "1" ]]; then
                break
            fi
            continue
        fi
        if [[ "${stage}" == "complete" ]]; then
            latest_timestamp="${latest_stage2##*stage2_last_}"
            latest_timestamp="${latest_timestamp%.pt}"
            if [[ -f "${run_dir}/stage2_best_phla_${latest_timestamp}.pt" \
                && -f "${run_dir}/stage2_best_ptcr_${latest_timestamp}.pt" ]]; then
                echo "[SKIP] ${run_label}: training and task weights already complete"
                SKIPPED+=("${run_label}")
                continue
            fi
            shopt -s nullglob
            joint_candidates=("${run_dir}"/stage2b_best_joint_*.pt)
            shopt -u nullglob
            export_source="${latest_stage2}"
            matching_joint="${run_dir}/stage2b_best_joint_${latest_timestamp}.pt"
            if [[ -f "${matching_joint}" ]]; then
                export_source="${matching_joint}"
            elif [[ ${#joint_candidates[@]} -gt 0 ]]; then
                export_source="${joint_candidates[${#joint_candidates[@]}-1]}"
            fi
            export_command=(
                "${PYTHON_BIN}"
                "${EXPORT_SCRIPT}"
                "--checkpoint" "${export_source}"
                "--output-dir" "${run_dir}"
                "--overwrite"
            )
            if [[ "${DRY_RUN}" == "1" ]]; then
                printf "[DRY RUN] export task weights for %s:" "${run_label}"
                printf " %q" "${export_command[@]}"
                printf "\n"
                PLANNED+=("${run_label}/export")
                continue
            fi
            echo "[EXPORT] ${run_label}: generating separate task weights"
            "${export_command[@]}" 2>&1 | tee -a "${LOG_ROOT}/fold_${fold}_seed_${seed}.log"
            export_return_code=${PIPESTATUS[0]}
            if [[ ${export_return_code} -eq 0 ]]; then
                echo "[OK] ${run_label}: task weights exported"
                SUCCEEDED+=("${run_label}/export")
            else
                echo "[FAILED] ${run_label}: task-weight export exit code ${export_return_code}" >&2
                FAILED+=("${run_label}/export")
                if [[ "${STOP_ON_ERROR}" == "1" ]]; then
                    break
                fi
            fi
            continue
        fi
        echo "[RESUME] ${run_label}: ${latest_stage2} (stage=${stage})"
        command+=("--resume" "${latest_stage2}")
    else
        echo "[START] ${run_label}: ${stage1_checkpoint}"
        command+=("--stage1-checkpoint" "${stage1_checkpoint}")
    fi

    if [[ -n "${LIMIT}" ]]; then
        command+=("--limit" "${LIMIT}")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf "[DRY RUN]"
        printf " %q" "${command[@]}"
        printf "\n"
        PLANNED+=("${run_label}")
        continue
    fi

    {
        echo ""
        echo "============================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${run_label}"
        echo "Realtime log: ${log_path}"
        printf "Command:"
        printf " %q" "${command[@]}"
        printf "\n"
        echo "============================================================"
    } | tee -a "${log_path}"

    "${command[@]}" 2>&1 | tee -a "${log_path}"
    return_code=${PIPESTATUS[0]}

    if [[ ${return_code} -eq 0 ]]; then
        echo "[OK] ${run_label}" | tee -a "${log_path}"
        SUCCEEDED+=("${run_label}")
    else
        echo "[FAILED] ${run_label}: exit code ${return_code}" \
            | tee -a "${log_path}" >&2
        FAILED+=("${run_label}")
        if [[ ${return_code} -eq 130 || ${return_code} -eq 143 ]]; then
            echo "[INTERRUPTED] Stop requested; exiting immediately." >&2
            exit "${return_code}"
        fi
        if [[ "${STOP_ON_ERROR}" == "1" ]]; then
            break
        fi
    fi
done

echo ""
echo "============================================================"
echo " Stage-2 batch summary"
echo "============================================================"
echo "Succeeded : ${#SUCCEEDED[@]}"
echo "Skipped   : ${#SKIPPED[@]}"
echo "Planned   : ${#PLANNED[@]}"
echo "Failed    : ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed runs:"
    printf "  - %s\n" "${FAILED[@]}"
    exit 1
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete; no training was started."
else
    echo "All discovered Stage-1 runs are complete at Stage 2."
fi
