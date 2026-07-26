#!/usr/bin/env bash
set -uo pipefail

CONFIG="configs/plm_stage1.yaml"
SEEDS=(42 2026)
DEVICE="cuda"
SWANLAB_MODE="online"
FAILED_FOLDS=()
CACHE_ROOT="embedding_cache/plm_stage1"

for fold in 2 3 4 5; do
    echo "=============================================="
    echo "  Fold ${fold} / seeds: ${SEEDS[*]}"
    echo "=============================================="

    # ------------------------------------------------------------------
    # 1. Build cache if missing
    # ------------------------------------------------------------------
    MANIFEST="${CACHE_ROOT}/fold_${fold}/peptide/manifest.json"
    if [ -f "${MANIFEST}" ]; then
        echo "[CACHE] Fold ${fold} cache found, skip precompute."
    else
        echo "[CACHE] Fold ${fold} cache NOT found, building..."
        ret=0
        python source/precompute_plm_embeddings.py \
            --config "${CONFIG}" \
            --fold "${fold}" \
            --device "${DEVICE}" || ret=$?
        if [ "${ret}" -ne 0 ]; then
            echo "[ERROR] Cache build for fold ${fold} FAILED (exit code: ${ret})"
            FAILED_FOLDS+=("${fold}")
            echo ""
            continue
        fi
        echo "[CACHE] Fold ${fold} cache built successfully."
    fi

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    ret=0
    python source/run_plm_stage1_seeds.py \
        --config "${CONFIG}" \
        --fold "${fold}" \
        --seeds "${SEEDS[@]}" \
        --device "${DEVICE}" \
        --swanlab-mode "${SWANLAB_MODE}" || ret=$?
    if [ "${ret}" -eq 0 ]; then
        echo "[OK] Fold ${fold} finished successfully."
    else
        echo "[ERROR] Fold ${fold} training FAILED (exit code: ${ret})"
        FAILED_FOLDS+=("${fold}")
    fi
    echo ""
done

# ------------------------------------------------------------------
# 3. Summary
# ------------------------------------------------------------------
if [ ${#FAILED_FOLDS[@]} -eq 0 ]; then
    echo "All folds complete successfully."
else
    echo "=============================================="
    echo "  [ERROR] The following folds FAILED: ${FAILED_FOLDS[*]}"
    echo "=============================================="
    exit 1
fi
