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
  filter_stats.json
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
  warmup_last.pt
  last.pt
  best_joint.pt
```

三次训练完成后生成：

```text
trained_model/PLM_Unified/fold_1/seed_summary.json
```

联合 checkpoint 指标为：

```text
(pHLA AUROC + pHLA AUPR + pTCR AUROC + pTCR AUPR) / 4
```

最大 10 rounds，连续 3 rounds 提升不超过 `1e-4` 时早停。

## 5. 断点续训与评估

```bash
python source/train_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42 \
  --resume trained_model/PLM_Unified/fold_1/seed_42/last.pt
```

验证集评估：

```bash
python source/evaluate_plm_stage1.py \
  --config configs/plm_stage1.yaml \
  --fold 1 \
  --seed 42
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
