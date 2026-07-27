# Unified PLM Stage 2

该管线实现《Embedding论文问题与相关研究》第 15 章的逐步解冻训练，并沿用阶段
1 的统一双任务下游模型。

## 1. 已确认的训练策略

阶段 2 必须从同一 fold/seed 已完成训练的
`last_<YYYYMMDD_HHMMSS_ffffff>.pt` 初始化。不能使用或合并
`best_phla`/`best_ptcr`，因为两个文件可能包含不同训练时刻的共享 Peptide
Adapter。

阶段 2A：

- Peptide ESM-C 解冻最后 2 个 Transformer blocks；
- HLA ESM-C、TCR-BERT 完全冻结；
- 固定训练 2 个 pHLA → pTCR alternating rounds；
- 两轮完成后回载 `joint_score` 最佳的阶段 2A checkpoint，再进入 2B。

阶段 2B：

- Peptide ESM-C 解冻最后 2 层；
- HLA ESM-C 解冻最后 1 层；
- TCR-BERT 解冻最后 2 层；
- 至少训练 5 轮，最多 20 轮；
- 连续 5 轮 `joint_score` 未提升后早停。

其中：

```text
joint_score = (
    pHLA AUROC + pHLA AUPR + pTCR AUROC + pTCR AUPR
) / 4
```

阶段 2A/2B 均关闭 FGM。默认训练 micro batch 为 8，梯度累积 16 次，有效 batch
为 128；验证不保留激活和梯度，单独使用 batch size 64。pHLA 和 pTCR
分别累积，不跨任务阶段混合。

## 2. 分层学习率与动态调度

默认学习率：

| 参数组 | 学习率 |
|---|---:|
| Peptide ESM-C 最后一层 | `1e-5` |
| Peptide ESM-C 倒数第二层 | `5e-6` |
| HLA ESM-C 最后一层（仅 2B） | `1e-5` |
| TCR-BERT 最后一层（仅 2B） | `1e-5` |
| TCR-BERT 倒数第二层（仅 2B） | `5e-6` |
| 三个 Adapter | `5e-5` |
| Cross-attention | `1e-4` |
| Pooling | `1e-4` |
| Classifier | `1e-4` |

上述值是各参数组的基准/峰值学习率。2A 和 2B 各自创建 optimizer 与
scheduler：前 5% optimizer steps 线性 warm-up，随后 cosine decay。进入
2B 时因解冻范围改变，会重建 optimizer/scheduler。

默认同时使用：

- AdamW，weight decay `0.01`；
- gradient clipping `1.0`；
- bf16；
- Transformer Engine、xFormers 和 FlashAttention CUDA 加速核；
- ESM-C 顶层手动 activation checkpointing；
- TCR-BERT 原生 gradient checkpointing。

阶段 2A 的 HLA ESM-C 完全冻结，且 pHLA 数据会高度重复使用少量 HLA
pseudo-sequence。因此训练器会在 GPU 上按序列惰性缓存 HLA residue embedding：
每条不同 HLA 只经过一次冻结 ESM-C；进入阶段 2B、HLA 顶层解冻前自动清空缓存。
Peptide 和阶段 2B 中已解冻的 HLA/TCR 不会缓存，梯度和训练目标保持不变。

## 3. 实时训练日志

默认每个 optimizer step 输出一次实时进度（即每累计 16 个 micro batches），并同步
写入 SwanLab 和本地日志。日志包含 batch/optimizer-step 进度、当前及平均 loss、
梯度范数、各参数组动态学习率、吞吐量、耗时、ETA 和 GPU
allocated/reserved/peak 显存。验证阶段也会定期输出进度以及最终 AUROC、AUPR、
ACC、MCC 和 F1。

一键脚本使用无缓冲 Python 输出，并通过 `tee` 同时显示在终端和保存到：

```text
trained_model/PLM_Unified/stage2_logs/fold_<fold>_seed_<seed>.log
```

如需减少输出频率，可将配置中的 `stage2.log_every_optimizer_steps` 调大；该字段只
影响日志频率，修改后仍可恢复已有 checkpoint。`stage2.eval_batch_size` 也属于
运行时参数，可在恢复 checkpoint 时按显存大小调整，不改变训练状态。

## 4. 启动训练

### 一键训练全部 fold/seed

根目录脚本会自动发现
`trained_model/PLM_Unified/fold_*/seed_*/last.pt`，在单个 GPU 上依次训练。
已完成的阶段 2 会跳过，未完成的 `stage2_last_<timestamp>.pt` 会自动恢复：

```bash
./run_plm_stage2_all.sh
```

指定 fold 和 seed（使用逗号分隔）：

```bash
./run_plm_stage2_all.sh \
  --folds 2,3,4,5 \
  --seeds 42,2026
```

只训练一个组合：

```bash
./run_plm_stage2_all.sh --folds 1 --seeds 3407
```

也可以通过环境变量设置：

```bash
STAGE2_FOLDS=1,3,5 \
STAGE2_SEEDS=42,2026 \
./run_plm_stage2_all.sh
```

脚本仅训练所选 fold/seed 在磁盘上实际存在的组合。例如 fold 2 没有 seed
3407 的 `last.pt` 时，不会凭空创建该组合；若筛选结果为空会直接报错。

首次使用前可以只预览将要执行的任务：

```bash
DRY_RUN=1 ./run_plm_stage2_all.sh
```

常用覆盖参数：

```bash
STAGE2_DEVICE=cuda \
STAGE2_SWANLAB_MODE=online \
STAGE2_NUM_WORKERS=2 \
./run_plm_stage2_all.sh
```

少量数据逐任务调试：

```bash
STAGE2_LIMIT=32 \
STAGE2_SWANLAB_MODE=local \
STAGE2_NUM_WORKERS=0 \
./run_plm_stage2_all.sh
```

每个 fold/seed 的控制台输出会同时追加到
`trained_model/PLM_Unified/stage2_logs/`。脚本默认某个任务失败后继续其余任务；
设置 `STAGE2_STOP_ON_ERROR=1` 可在首次失败时停止。设置
`STAGE2_FORCE_RESTART=1` 可忽略未完成的阶段 2 checkpoint 并创建一次新训练。

### 单独训练一个 fold/seed

先确认阶段 1 checkpoint 的 `stage` 已为 `complete`，且 fold、seed 和模型配置
与阶段 2一致：

```bash
python source/train_plm_stage2.py \
  --config configs/plm_stage2.yaml \
  --stage1-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/last_<timestamp>.pt \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

少量数据调试：

```bash
python source/train_plm_stage2.py \
  --config configs/plm_stage2.yaml \
  --stage1-checkpoint \
    trained_model/PLM_Unified/fold_1/seed_42/last_<timestamp>.pt \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --limit 32 \
  --num-workers 0 \
  --swanlab-mode local
```

阶段 2 始终在线编码原始序列，不能使用阶段 1 的 embedding cache，因为已解冻
PLM 的输出必须参与反向传播。

## 5. Checkpoint 与恢复

每次运行生成：

```text
stage2a_best_joint_<timestamp>.pt
stage2b_best_joint_<timestamp>.pt
stage2_last_<timestamp>.pt
stage2_best_phla_<timestamp>.pt
stage2_best_ptcr_<timestamp>.pt
```

其中最后两个是类似阶段 1 `best_phla`/`best_ptcr` 的任务专属权重：

- `stage2_best_phla`：Peptide ESM-C、HLA ESM-C、共享 Peptide Adapter，以及
  pHLA 的 HLA Adapter、Cross-attention、Pooling 和 Classifier；
- `stage2_best_ptcr`：Peptide ESM-C、TCR-BERT、共享 Peptide Adapter，以及
  pTCR 的 TCR Adapter、Cross-attention、Pooling 和 Classifier。

任务专属文件按各自任务的 `(AUROC + AUPR) / 2` 独立选择，不会把 pHLA 和 pTCR
分类头混在同一个部署权重中。joint 文件仍然保留，用于恢复统一交替训练。

`stage2a_best_joint`、`stage2b_best_joint` 和 `stage2_last` 这三个训练 checkpoint
包含：

- Peptide ESM-C 的全部参数，包括冻结底层；
- HLA ESM-C 的全部参数，包括冻结底层；
- TCR-BERT backbone 的全部参数，包括冻结底层；
- 完整统一下游模型；
- optimizer、scheduler、scaler、DataLoader RNG 和全局 RNG 状态。

两个 `stage2_best_*` 任务 checkpoint 不保存 optimizer，而只保存该任务部署所需的
两个 PLM、共享 Adapter 和对应分类分支。所有 PLM 参数仍然包含冻结底层。训练
checkpoint 文件较大，原子写入时还会短暂生成同等大小的 `.tmp` 文件，运行目录必须
预留至少约两个 checkpoint 文件大小的额外空间。

断点恢复：

```bash
python source/train_plm_stage2.py \
  --config configs/plm_stage2.yaml \
  --resume \
    trained_model/PLM_Unified/fold_1/seed_42/stage2_last_<timestamp>.pt \
  --fold 1 \
  --seed 42 \
  --device cuda \
  --swanlab-mode online
```

如果某次旧阶段 2 训练已经有 `stage2_last`，但还没有两个任务专属文件，可以单独
导出，无需重新训练：

```bash
python source/export_plm_stage2_task_models.py \
  --checkpoint trained_model/PLM_Unified/fold_1/seed_42/stage2_last_<timestamp>.pt \
  --overwrite
```

一键脚本会自动检测这种情况并执行导出。

恢复时仍会从配置路径构建三个基础 PLM 和 tokenizer，再用 checkpoint 中的完整
权重覆盖；基础 checkpoint 指纹、模型配置、fold、seed 和阶段 2 配置必须一致。

## 6. CUDA 加速环境与 NVML

`unifyimmun` 环境当前与 `torch 2.6.0+cu124` 配套安装：

```text
transformer-engine / transformer-engine-cu12 / transformer-engine-torch  1.13.0
xformers                                                            0.0.29.post2
flash-attn                                                          2.7.4.post1
```

环境变量 `CUDA_HOME` 已持久化为
`/home/mjp/miniconda3/envs/unifyimmun/cuda`，训练入口也会自动发现该 CUDA 12.4
overlay。启动日志中 ESM-C 不应再出现三个加速库缺失的 fallback 提示。

当前机器曾出现内核 NVIDIA module 为 `595.71.05`、用户态 NVML 为 `595.84`
的更新后未重启状态。项目根目录的临时命令可在不打断训练时查看 GPU：

```bash
./nvidia-smi-compatible.sh
```

一键脚本会在系统 `nvidia-smi` 失败、兼容命令成功时自动为新训练进程设置对应
NVML。永久修复是在已保存 checkpoint 后重启系统，让磁盘上的 `595.84` kernel
module 生效；重启后脚本会自动回到系统 NVML，不再使用旧兼容库。
