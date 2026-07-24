# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

UnifyImmun predicts antigen binding specificity to **both HLA and TCR** with a single unified cross-attention model. The active line of work (branch `PLM`) is a **new additive PLM stage-1 pipeline** that sits alongside the legacy scripts: it trains adapter layers over **frozen** residue-level embeddings from ESM-C (peptide/HLA) and TCR-BERT (TCR), with module-targeted FGM adversarial training and a two-phase progressive schedule (1A warm-up → 1B alternating dual-task). The legacy `run_all_phases.py` / `HLA_test.py` / `TCR_test.py` scripts referenced in README.md are **not present** in this branch — do not invent them; the PLM pipeline is the one that exists.

`PLM_STAGE1.md` is the authoritative runbook for the PLM pipeline (cache construction, debug training, three-seed experiments, resume, evaluation, SwanLab). Read it before changing training flow.

## Environment

- Python 3.10, CUDA PyTorch. Server-local model checkpoints are referenced by absolute path in the YAML:
  - ESM-C: `/home/mjp/model/ESMC-300M` (hidden 960)
  - TCR-BERT: `/home/mjp/model/TCR-Bert` (hidden 768)
- `requirements-plm.txt` is the **PLM-pipeline** dependency set (not `requirements.txt`). It pins `transformers @ git+https://github.com/Biohub/transformers.git@3a8956fb...` — this fork is required for ESM-C and must match the checkpoint's exported version. Install: `pip install -r requirements-plm.txt` (after a CUDA torch build is already present).

## Common commands

All entry-point scripts live in `source/` and resolve their own default config (`configs/plm_stage1.yaml`), so they run from anywhere. Config paths in the YAML are relative to `paths.project_root` (`..`), i.e. the repo root.

```bash
# 1. Verify local checkpoints + tokenizer residue alignment (run before any training)
python source/smoke_test_plm_encoders.py --config configs/plm_stage1.yaml --device cuda

# 2. Build the residue-level HDF5 cache (fold 1 train+val by default; ~21 GB float16)
python source/precompute_plm_embeddings.py --config configs/plm_stage1.yaml --fold 1 --device cuda
#    Rebuild explicitly with --overwrite; rebuild subset with --entities peptide hla

# 3. Debug training (small row limit, in-process cache reads, local SwanLab)
python source/train_plm_stage1.py --config configs/plm_stage1.yaml --fold 1 --seed 9999 \
  --limit 4096 --num-workers 0 --swanlab-mode local
#    Online (no-cache) debug: add --no-cache --limit 256 (uses training.online_batch_size)

# 4. Full three-seed experiment (subprocesses, one per seed)
python source/run_plm_stage1_seeds.py --config configs/plm_stage1.yaml --fold 1 \
  --seeds 42 3407 2026 --device cuda --swanlab-mode online

# 5. Resume from a checkpoint
python source/train_plm_stage1.py --config configs/plm_stage1.yaml --fold 1 --seed 42 \
  --resume trained_model/PLM_Unified/fold_1/seed_42/last.pt

# 6. Evaluate a checkpoint on configured HLA/TCR splits
python source/evaluate_plm_stage1.py --config configs/plm_stage1.yaml --fold 1 --seed 42
```

### Conventions enforced by the runbook

- `--limit` is **smoke-test only**; never use it for real experiments.
- Debug seed is **9999** — it must not collide with the experiment seeds **42 / 3407 / 2026**.
- `--num-workers 0` keeps HDF5 cache reads in the main process to ease data-issue debugging.
- Evaluating new splits (independent/external/covid/triple): first add the split to `cache.hla_splits` / `cache.tcr_splits`, rebuild the cache, then pass `--hla-split` / `--tcr-split`. All test sets in fixed-width cache at once can exceed 30 GB — build per disk budget.
- Mixed precision is `bf16`; autocast is only enabled on CUDA.

## Tests

Tests are synthetic (no real checkpoints / no GPU needed) and import via the `source.` package path, so run them from the **repo root**:

```bash
python -m unittest tests.test_plm_unified          # or: python -m pytest tests/
```

## Architecture (PLM pipeline)

`source/plm_unified/` is the pipeline package; `source/*.py` are thin CLI entry points that wire YAML → config → the modules below.

- **`config.py`** — Typed dataclasses (`PathsConfig`, `ModelConfig`, `CacheConfig`, `TrainingConfig`, `SwanLabConfig`, `ExperimentConfig`) with `load_config` (YAML→dataclass) and `apply_overrides` (CLI flags). `ExperimentConfig` also derives `cache_dir()` / `run_dir()` (per-fold, per-seed) and validates that model checkpoints exist.
- **`encoders.py`** — Frozen residue-level encoders (`ESMCResidueEncoder`, `TCRBertResidueEncoder`) plus `OnlineEncoderBundle` (used when `--no-cache`). `checkpoint_fingerprint` hashes checkpoint file structure (sizes + small text/safetensors samples) — **not** the multi-GB weights — so cache validity can be tracked cheaply.
- **`cache.py`** — Sharded, fixed-width HDF5 residue cache with a **SQLite sequence index**. `EmbeddingCacheWriter` writes shards; `EmbeddingCache` reads them back by sequence key via `CacheLocator(shard, row, length)`. Has a `MANIFEST_VERSION`; rebuilding requires `--overwrite`.
- **`data.py`** — CSV discovery (`split_path` maps task/split/fold → `data/data_HLA` or `data/data_TCR` files), sequence normalization, and `make_cached_dataloader` / `make_online_dataloader`. `Task = Literal["phla", "ptcr"]`.
- **`attention.py`** — `SequenceAdapter` (Linear→GELU→Dropout→Linear→LayerNorm, masked fill), `CrossAttentionBlock` (4-head `nn.MultiheadAttention`), `MaskedAttentionPooling`, `BindingClassifier`.
- **`model.py`** — `UnifiedBindingModel`: **one shared Peptide Adapter** + two receptor Adapters (HLA, TCR) + two cross-attention branches + two classifier heads. Single `forward(task, peptide, p_mask, receptor, r_mask)` returns `BindingOutput(logits, cross_attention, pooling_weights)`.
- **`fgm.py`** — `ModuleFGM`: old-style per-parameter Fast Gradient Method that attacks only **explicitly selected modules** (in 1B, the current task's Peptide Adapter + receptor Adapter), with BN state handling and backup/restore.
- **`trainer.py`** — `Stage1Trainer` implementing the two-phase schedule: 1A `warmup` (per-task epochs, no FGM) → writes `warmup_last.pt` → 1B `stage1b` (pHLA/pTCR alternate per epoch with module-targeted FGM). Tracks `best_joint_score` = `(pHLA AUROC + pHLA AUPR + pTCR AUROC + pTCR AUPR)/4`, early-stops after `max_rounds` or `early_stopping_patience` consecutive rounds with < `early_stopping_min_delta` improvement. Checkpoints: `warmup_last.pt`, `last.pt`, `best_joint.pt`. Also exports `set_global_seed`, `build_optimizer`/`build_scheduler`, `resolve_device`, `load_training_checkpoint`, `peek_swanlab_run_id`.
- **`metrics.py`** — `BinaryMetrics` (loss, AUROC, AUPR, ACC, MCC, F1, precision/recall/sensitivity/specificity) + `joint_score` / `prefixed_metrics` helpers (sklearn-backed).
- **`evaluation.py`** — `evaluate_model`: standalone inference-mode loop with autocast, shared by the eval CLI.
- **`tracking.py`** — `SwanLabTracker`: thin adapter that stays import-free when `mode="disabled"` (so tests/CPU runs don't require swanlab). Default project `unifyimmun`, group `plm-stage1-fold{fold}`, experiment `PLM-Stage1-fold{f}-seed{s}`.

### Data layout

- `data/data_HLA/*.csv` — pHLA splits: `{train,val}_fold_{1..5}.csv` plus `{dataset,independent,external,HPV,neoantigen}_set.csv`. Columns include `peptide` and `HLA`.
- `data/data_TCR/*.csv` — pTCR splits: same fold pattern plus `{covid,independent,triple}_set.csv`. Columns include `tcr` (CDR3), `peptide`, `HLA`.
- `data/data_dict.npy` — legacy lookup table.
- `embedding_cache/plm_stage1/fold_{fold}/{peptide,hla,tcr}/` — built caches (gitignored).
- `trained_model/PLM_Unified/fold_{fold}/seed_{seed}/` — checkpoints + `seed_summary.json` (gitignored).
- `swanlog/` — SwanLab run logs (gitignored).

## Conventions

- Code is typed (`from __future__ import annotations`, dataclasses, `Literal` types); match this style in new code.
- Paths in the YAML are relative to `paths.project_root` and resolved through `Path` — never hardcode absolute paths in modules (only the two model checkpoints are absolute, as server-local defaults in the config dataclasses).
- Cache dtype, shard size, and precompute batch size live under `cache:` in the YAML, not in code.
