# TCR `_set` 数据集评估报告

## 汇总结果

| Dataset | N | AUROC | AUPR | Accuracy | MCC | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| covid_set.csv | 1,089,310 | 0.633083 | 0.626987 | 0.570336 | 0.177573 | 0.394681 | 0.676502 | 0.278614 | 0.865292 |
| independent_set.csv | 28,707 | 0.939995 | 0.896854 | 0.874769 | 0.725444 | 0.821968 | 0.815627 | 0.828409 | 0.899620 |
| triple_set.csv | 97,043 | 0.882824 | 0.877842 | 0.860175 | 0.704837 | 0.801950 | 0.895466 | 0.726119 | 0.945836 |

## 评估配置

- 模型：`/home/mjp/Project/fine-tuning/UnifyImmun/trained_model/TCR_2/model_TCR.pkl`
- 模型 SHA-256：`5b312f0968a18a9c69400ad5ec48cd0bb4041c14f83ce48d592eb9a9d532d82e`
- 数据目录：`/home/mjp/Project/fine-tuning/UnifyImmun/data/data_TCR`
- 推理设备：`cuda:0`
- Batch size：8192
- 分类阈值：0.5（正类概率严格大于阈值时预测为 1）
- AUPR：`precision_recall_curve` 后对 Recall–Precision 曲线作梯形积分
- 完整性：评估所有非空样本，不丢弃最后一个不完整 batch

## 文件

- [covid_set.csv](covid_set.md)
- [independent_set.csv](independent_set.md)
- [triple_set.csv](triple_set.md)
