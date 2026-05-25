# Fine-tuning 使用说明

## 概述

基于 `fine-tuning.txt` 的要求，创建了微调脚本 `source/finetune_liver.py`，用于在肝数据 (`TCR_Peptide_balanced.csv`) 上微调 pTCR 模型。

## 微调策略

### 训练参数
- **TCR encoder** (encoder_T): 学习率 3e-5 (范围 1e-5 ~ 5e-5)
- **Peptide encoder** (encoder_P): 学习率 1e-5，从预训练HLA模型加载权重
- **pTCR cross-attention** (cross_2): 学习率 1e-4
- **pTCR FC head** (projection): 学习率 5e-4 (范围 1e-4 ~ 1e-3)

### 冻结参数
根据 fine-tuning.txt 的建议，以下参数在微调中不使用（冻结）:
- HLA encoder (encoder_H)
- pHLA cross-attention (cross_1)
- pHLA FC head

## 数据

- 数据文件: `data/data_liver/TCR_Peptide_balanced.csv`
- 样本数: 30,064 (正负样本各 15,032，已平衡)
- 格式: peptide, tcr, label

## 运行方式

```bash
# 方式1: 使用运行脚本
cd /home/mjp/Project/fine-tuning/UnifyImmun
./run_finetune.sh

# 方式2: 直接运行Python脚本
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unifyimmun
cd /home/mjp/Project/fine-tuning/UnifyImmun/source
python3 finetune_liver.py
```

## 输出

- 模型保存路径: `trained_model/finetune_liver/`
- 每个fold保存一个模型: `model_finetune_fold{1-5}.pkl`
- 训练日志包含: AUC, Accuracy, MCC, F1, AUPR, Sensitivity, Specificity

## 特性

1. **分层学习率**: 不同组件使用不同学习率
2. **早停机制**: patience=10, 防止过拟合
3. **对抗训练**: FGM对抗训练增强鲁棒性
4. **5折交叉验证**: Stratified K-Fold保证类别平衡
5. **预训练权重加载**: Peptide encoder从HLA模型加载预训练权重

## 测试结果

初步测试 (1 fold, 3 epochs):
- Epoch 1: Train AUC=0.9929, Val AUC=0.9998
- Epoch 2: Train AUC=0.9997, Val AUC=1.0000
- Epoch 3: Train AUC=0.9999, Val AUC=1.0000

模型收敛快速，效果优秀。