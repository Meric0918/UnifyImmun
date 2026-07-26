# Unified PLM Stage 1

该管线实现《Embedding论文问题与相关研究》第 6–12 章确认后的版本：

- Peptide/HLA：`/home/mjp/model/ESMC-300M`；
- TCR：`/home/mjp/model/TCR-Bert`；
- HLA 第一版直接编码现有 33/34 位 pseudo-sequence；
- 三个独立 Adapter，共享一个 Peptide Adapter；
- 4-head cross-attention、masked attention pooling、双分类头；
- 1A 无 FGM 预热，1B pHLA/pTCR 逐 epoch 交替；
- 1B 只对当前任务的 Peptide Adapter 与 receptor Adapter 使用 FGM；
- SwanLab 按 fold 分组，三个随机种子分别记录。

## 1. 环境

服务器已有 CUDA PyTorch 环境时，只安装缺失依赖：

```bash
cd /home/mjp/Project/PLM/UnifyImmun
pip install -r requirements-plm.txt
```

`transformers==4.57.6` 与 ESM-C checkpoint 的导出版本一致。正式运行前先验证
两个本地 checkpoint、tokenizer 和逐残基对齐：

```bash
python source/smoke_test_plm_encoders.py \
  --config configs/plm_stage1.yaml \
  --device cuda
```

预期关键输出：

```text
ESM-C hidden size：960
TCR-BERT hidden size：768
ESM-C lengths：[9, 34]
TCR-BERT lengths：[13, 15]
```

任一长度不一致都应停止，不要跳过 tokenizer 对齐检查。

## 2. 构建残基级缓存

默认只收集 Fold 1 的 train 和 validation（固定宽度 float16 预计约 21 GB），
过滤不在 8–15 范围内的 peptide，再生成分片 HDF5 和 SQLite 索引：

```bash
python source/precompute_plm_embeddings.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --device cuda
```

缓存目录：

```text
embedding_cache/plm_stage1/fold_1/
  filter_stats_<YYYYMMDD_HHMMSS_ffffff>.json
  peptide/
  hla/
  tcr/
```

如需重建，必须显式传入 `--overwrite`。可使用 `--entities peptide hla` 或
`--entities tcr` 单独重建实体缓存。

## 3. 调试训练

调试时限制每个 split 的行数并关闭远端记录：

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 9999 \
  --limit 4096 \
  --num-workers 0 \
  --swanlab-mode local
```

`--limit` 只用于 smoke test，正式实验禁止使用。调试时使用
`--num-workers 0` 可将 HDF5 缓存读取留在主进程，便于定位数据问题。
调试 seed 使用 9999，避免覆盖正式实验的 seed 42/3407/2026。

无需缓存的在线编码调试必须同时指定 `--no-cache` 和 `--limit`，并自动使用
`training.online_batch_size`：

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42 \
  --no-cache \
  --limit 256 \
  --swanlab-mode local
```

## 4. 正式三随机种子实验

```bash
python source/run_plm_stage1_seeds.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seeds 42 3407 2026 \
  --device cuda \
  --swanlab-mode online
```

每个 seed 的产物：

```text
trained_model/PLM_Unified/fold_1/seed_<seed>/
  warmup_last_<YYYYMMDD_HHMMSS_ffffff>.pt
  last_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_phla_<YYYYMMDD_HHMMSS_ffffff>.pt
  best_ptcr_<YYYYMMDD_HHMMSS_ffffff>.pt
```

同一次训练的所有文件共享同一个时间戳，重复运行相同 fold/seed 不会覆盖旧结果。
`last_<timestamp>.pt` 保留优化器、scheduler、两个任务分支及各任务早停状态，
用于恢复训练。`best_phla_<timestamp>.pt` 与 `best_ptcr_<timestamp>.pt`
是精简的独立推理模型，分别只包含共享
Peptide Adapter 和对应任务的 receptor Adapter、cross-attention、pooling
及分类头。

三次训练完成后生成：

```text
trained_model/PLM_Unified/fold_1/seed_summary_<YYYYMMDD_HHMMSS_ffffff>.json
```

两个任务分别使用以下分数选择各自的最佳 checkpoint：

```text
task_score = (task AUROC + task AUPR) / 2
```

每个任务分别维护最佳分数和 patience。某任务连续 3 个有效训练 round
提升不超过 `1e-4` 时，仅停止该任务；另一任务可继续训练。两个任务均停止
或达到最大 10 rounds 后结束。训练过程不再生成 `best_joint.pt`。

## 5. 断点续训与评估

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42 \
  --resume trained_model/PLM_Unified/fold_1/seed_42/last_20260726_120000_000001.pt
```

验证集评估：

```bash
python source/evaluate_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42
```

未传 `--checkpoint` 时，评估脚本自动选择最新一组时间戳一致的 pHLA/pTCR
权重；生成的评估 JSON 也会自动添加新的时间戳后缀。

单任务 checkpoint 必须配合对应的 `--tasks` 使用：

```bash
python source/evaluate_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --checkpoint trained_model/PLM_Unified/fold_1/seed_42/best_phla_20260726_120000_000001.pt \
  --tasks phla \
  --fold 1 \
  --device cuda
```

旧版本已有的 `best_joint.pt` 无需重新训练，可直接拆分：

```bash
python source/export_plm_task_models.py \
  --checkpoint trained_model/PLM_Unified/fold_1/seed_42/best_joint.pt
```

如需评估 independent/external/covid/triple，先把相应 split 加入 YAML 的
`cache.hla_splits` 或 `cache.tcr_splits` 并重建缓存，再向评估命令传入
`--hla-split`/`--tcr-split`。全部测试集同时使用固定宽度缓存可能超过 30 GB，
应按磁盘预算分批构建。

## 6. SwanLab 记录

默认 project 为 `unifyimmun`，group 为 `plm-stage1-fold1`。每个 run 记录：

- 配置、模型路径和 cache checkpoint 指纹；
- 数据保留/过滤数量；
- clean/adversarial loss；
- FGM epsilon、攻击参数数量和 Adapter 梯度范数；
- 每任务样本数、optimizer step 数及比例；
- AUROC、AUPR、accuracy、MCC、F1；
- joint score、best joint score、early-stopping 状态；
- 各参数组学习率和 checkpoint 恢复所需 run id。
