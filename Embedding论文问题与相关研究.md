# 分支 · 分支 · 论文问题与相关研究

**User:** Anonymous  
**Created:** 7/4/2026 18:02:50  
**Updated:** 7/4/2026 18:05:12  
**Exported:** 7/4/2026 18:14:00  
**Link:** [https://chatgpt.com/c/6a48da31-a770-83ec-9a1a-28a642005851](https://chatgpt.com/c/6a48da31-a770-83ec-9a1a-28a642005851)  

## Prompt:
2026/7/4 18:02:46

先按照路线 B：论文增强版，最推荐，给我详细的具体训练步骤

## Response:
2026/7/4 18:02:49

> **搜索官方文档和技术来源**
>
> [SeprotHub/Model-RASH-ESMC-300M · Hugging Face](https://huggingface.co/SeprotHub/Model-RASH-ESMC-300M)
> [Self-supervised learning of T cell receptor sequences exposes core properties for T cell membership - PubMed](https://pubmed.ncbi.nlm.nih.gov/38669334/)
> [TCR-BERT: learning the grammar of T-cell receptors for flexible antigen-binding analyses](https://proceedings.mlr.press/v240/wu24b.html)
> [Structure-based prediction of T cell receptor recognition of unseen epitopes using TCRen | Nature Computational Science](https://www.nature.com/articles/s43588-024-00653-0)
> [Contrastive learning of T cell receptor representations - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405471224003697)
> [GitHub - Biohub/esm · GitHub](https://github.com/evolutionaryscale/esm)
> [TCR-GPT: Integrating Autoregressive Model and Reinforcement Learning for T-Cell Receptor Repertoires Generation](https://arxiv.org/abs/2408.01156)
> [Somatic recombination for T cell receptor - is this the correct process? Is the leader sequence still supposed to be there at the mRNA level?](https://www.reddit.com/r/Immunology/comments/1adihxo)
>
> **制定路线B的训练步骤**
>
> 用户提到路线B，其中包括ESM-C/ESM2与TCR-BERT的结合，并提到了冻结和逐层解冻等技术。针对HCC项目，计划可以分为几个阶段：数据准备、架构调整、冻结编码器预热、两阶段交替训练、逐步解冻以及HCC特定的有监督微调等。需要确认具体数据集，包括原始的UnifyImmun基准数据以及HCC数据集。
>
> **确认免疫伪序列长度**
>
> 免疫伪序列长度可能是固定的，不必断定为34，除非论文明确说明HLA伪序列是34个氨基酸。需要查阅文件确认确切的序列长度。
>
> **引用和细节确认**
>
> 我们有足够的资源，似乎不需要更多工具。但由于使用了网络数据，所有网络支持的声明都需要引用。可以引用ESMC官方仓库和TCR-BERT论文。大部分计划是我们的推荐意见，可以作为推断引用。任何关于ESMC和TCR-BERT能力的声明都应该引用。需要明确区分论文事实与工程推荐，还要确认原始维度和架构。
>
> **模型权重共享与内存管理**
>
> 关于ESM-C 300M模型，是应该分享HLA和肽物质的共享权重，还是初始化两个独立的版本？如果共享，部分解冻时会产生冲突。两个300M模型会占用大量显存，可能
>
> **考虑使用独立编码器和输入规范**
>
> 对于ESM-C 300M，可能需要使用独立的编码器，两个实例从相同的检查点初始化，但参数独立。可以默认使用高质量的设置，内存优化版在阶段1共享冻结的ESM-C，解冻后或使用不同的适配器。TCR-BERT输入可能是全TCR序列或仅CDR3，HLA输入建议使用34氨基酸的伪序列，有助于兼容性。
>
> **调整训练阶段和超参数**
>
> 对抗训练沿用旧版 UnifyImmun 的 FGM 思路，但不再通过参数名字符串匹配攻击目标。pHLA 阶段只攻击 Peptide Adapter 和 HLA Adapter，pTCR 阶段只攻击 Peptide Adapter 和 TCR Adapter；epsilon 固定从 1.0 起步。训练计划分为多个阶段，阶段0为基线，阶段1A为无 FGM 的冻结 PLM 预热，阶段1B在冻结 PLM 的交替训练中启用 FGM。
>
> **阶段2逐渐解冻和训练计划**
>
> 在阶段2中，我们将部分解冻最后两个ESM-C的肽块，HLA ESM-C的最后1-2层，TCR-BERT的最后两层。为了避免不平衡，每轮训练为一个pHLA-1个全周期与pTCR-1个全周期交替。使用不同学习率和微批次，结合梯度积累进行高效训练，采用早停策略。
>
> 已思考 1m 22s

# 路线 B：PLM 增强版 UnifyImmun 详细训练方案

本方案采用：

- **Peptide encoder：ESM-C 300M**
- **HLA encoder：ESM-C 300M**
- **TCR encoder：TCR-BERT**
- 保留 UnifyImmun 的：
  - peptide-HLA cross-attention；
  - peptide-TCR cross-attention；
  - 两阶段交替训练；
  - FGM 参数对抗训练；
  - 两个独立分类头。

服务器本地 checkpoint：

```text
ESM-C 300M：/home/mjp/model/ESMC-300M
TCR-BERT：  /home/mjp/model/TCR-Bert
```

原始 UnifyImmun 将 HLA pseudo-sequence、peptide 和 TCR CDR3 分别编码为固定长度的 64 维矩阵，其中 HLA 长度为 34、peptide 最大长度为 15、TCR 最大长度为 34；随后分别经过独立的单层 self-attention encoder，再进入两个 cross-attention 分支。`Unifyimmun.pdf`

改造后，不再使用原模型的：

```text
字符 Embedding
+ 正余弦位置编码
+ 单层 Self-Attention Encoder
```

而是替换为：

```text
预训练语言模型
+ Adapter/Projection
```

ESM-C 用于蛋白序列的表征学习；TCR-BERT 则是针对 TCR 序列训练的语言模型，适合作为 TCR 分支的初始化编码器。([GitHub](https://github.com/evolutionaryscale/esm?utm_source=chatgpt.com))

---

# 一、最终模型结构

```text
                        Peptide
                           │
                    ESM-C 300M
                           │
                   Peptide Adapter
                           │
              peptide token representations
                    ┌──────┴──────┐
                    │             │
                    │             │
HLA完整序列          │             │        TCR CDR3β
    │               │             │             │
ESM-C 300M          │             │         TCR-BERT
    │               │             │             │
抽取34个HLA          │             │         TCR Adapter
pseudo位点           │             │             │
    │               │             │             │
HLA Adapter          │             │             │
    │               │             │             │
    └── pHLA Cross-Attention      └── pTCR Cross-Attention
              │                               │
       pHLA Pooling                     pTCR Pooling
              │                               │
       pHLA Classifier                  pTCR Classifier
              │                               │
       HLA结合概率                      TCR结合概率
```

建议第一版只处理：

- HLA Class I；
- peptide 长度 8-15；
- TCR CDR3β；
- 二分类标签：binding / non-binding。

αβ 配对 TCR 可以作为后续增强，不建议第一版同时实现。

---

# 二、阶段 0：先固定数据和评估协议

这一阶段非常重要。如果数据划分不严格，换成大模型后很容易得到虚高结果。

## 2.1 pHLA 数据表

建议统一为：

```text
sample_id
peptide
hla_allele
hla_full_sequence
hla_pseudo_sequence
label
source
split
```

示例：

```csv
sample_id,peptide,hla_allele,hla_full_sequence,hla_pseudo_sequence,label
1,LLFGYPVYV,HLA-A*02:01,GSHSMRY...,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY,1
```

## 2.2 pTCR 数据表

```text
sample_id
peptide
cdr3
chain_type
label
source
patient_id
split
```

示例：

```csv
sample_id,peptide,cdr3,chain_type,label
1,LLFGYPVYV,CASSLGQAYEQYF,TRB,1
```

## 2.3 数据清洗

执行以下操作：

1. 全部序列转换为大写。
2. 去除空格、`*`、终止符和非法字符。
3. peptide 保留长度 8-15。
4. TCR CDR3 建议保留长度 5-34。
5. 去除完全重复记录。
6. 如果同一 pair 同时具有正、负标签，单独放入冲突集合，不直接用于训练。
7. HLA allele 统一格式：

```text
HLA-A02:01
HLA-A*02:01
A0201
```

全部转换为：

```text
HLA-A*02:01
```

对于 `X、B、Z、J、U、O` 等非标准氨基酸，第一版建议删除对应样本，而不是随意映射。

---

# 三、数据划分

不能只做随机 pair 划分。

建议至少构建以下三套划分。

## 3.1 Random-pair split

用于和原始 UnifyImmun 对照：

```text
train : validation : test = 80% : 10% : 10%
```

同一个 peptide 或 TCR 可以出现在不同集合，但 pair 不重复。

## 3.2 Peptide-unseen split

按照 peptide 分组：

```text
训练集 peptide ∩ 测试集 peptide = ∅
```

这是最重要的泛化测试，因为原论文也指出 unseen peptide 是更困难的场景。

## 3.3 TCR-unseen split

按照 CDR3 分组：

```text
训练集 CDR3 ∩ 测试集 CDR3 = ∅
```

更严格的第四套测试可以是：

```text
peptide 和 CDR3 在测试集中都未见过
```

## 3.4 防止负样本泄漏

正确流程是：

```text
先划分正样本
→ 再在每个 split 内分别构建负样本
```

不要先生成负样本再划分，否则同一个 peptide-TCR 组合或者高度相关组合可能跨集合出现。

---

# 四、负样本构建

第一版尽量与原始 UnifyImmun 保持一致。

原论文的 pHLA 和 pTCR 数据都使用了随机错配和 unbound sequence pool 等方式构建负样本，并指出负样本构造偏差是模型泛化的重要限制。`A unified cross-attention model for predicting antigen binding specificity to both HLA and TCR molecules.pdf`

## 4.1 pHLA 负样本

建议 1:1 或 1:2 正负比例：

- 50% 随机错配；
- 50% unbound peptide pool。

随机错配时：

```text
已知 HLA-A*02:01 ↔ peptide_A 为正样本
随机选择 peptide_B
构成 HLA-A*02:01 ↔ peptide_B
```

必须排除已经存在的已知正 pair。

## 4.2 pTCR 负样本

建议先使用 1:1：

- 50% peptide 和 CDR3 随机错配；
- 50% 来自未结合序列池。

在模型稳定后，再加入 hard negatives：

```text
与正样本 peptide 高度相似，但实验上不结合的 peptide
```

第一版不建议立即使用模型生成的 hard negative，避免训练流程过于复杂。

---

# 五、HLA 序列处理

## 5.1 推荐方式：完整 HLA 序列编码后抽取 34 个位点

原始 UnifyImmun 使用固定长度 34 的 HLA pseudo-sequence。`Unifyimmun.pdf`

但是直接把 34 个不连续位置拼接成伪序列输入 ESM-C，会破坏原始蛋白上下文。更推荐：

```text
完整 HLA heavy-chain 序列
→ ESM-C
→ 获得每个残基的 hidden state
→ 按 pseudo-sequence position index 抽取34个位置
→ HLA Adapter
```

得到：

```text
HLA feature shape = [batch, 34, d_model]
```

这种做法兼顾了：

- ESM-C 的完整序列上下文；
- UnifyImmun 使用的 HLA binding groove 位点；
- 与原始模型基本一致的 cross-attention 输入长度。

## 5.2 缺少完整 HLA 序列时

可以暂时直接编码：

```text
34-aa HLA pseudo-sequence
→ ESM-C
```

但应将其作为降级方案，并在实验中与“完整序列后抽取”进行对照。

---

# 六、三个编码器的具体实现

## 6.1 Peptide encoder

输入原始 peptide，不预先补零：

```text
LLFGYPVYV
```

经过 ESM-C：

```python
peptide_hidden = peptide_esmc(
    input_ids=peptide_ids,
    attention_mask=peptide_mask
).last_hidden_state
```

删除模型特殊 token：

```text
BOS、EOS、PAD
```

最终只保留真实氨基酸位置：

```text
[batch, peptide_length, d_esmc]
```

第一版只保留长度 8-15 的 peptide。现有 pHLA 数据中少量长度为 7 或
16-23 的样本在缓存构建和训练加载时使用同一规则过滤，并记录保留/丢弃数量。

然后进入 Adapter。

---

## 6.2 HLA encoder

当前数据文件的 `HLA` 列只包含 33/34 位 pseudo-sequence，没有等位基因名、
完整 HLA 序列或完整序列位点映射。因此第一版直接将 pseudo-sequence 输入
ESM-C：

```python
hla_pseudo_hidden = hla_esmc(
    input_ids=hla_ids,
    attention_mask=hla_mask
).last_hidden_state
```

输出：

```text
[batch, 34, d_esmc]
```

然后进入 HLA Adapter。

代码保留“完整 HLA 序列编码后 gather 34 位点”的扩展边界；获得完整序列映射
之前，不执行伪造的 gather。

---

## 6.3 TCR encoder

第一版输入 CDR3β：

```text
CASSLGQAYEQYF
```

通过 TCR-BERT：

```python
tcr_hidden = tcr_bert(
    input_ids=tcr_ids,
    attention_mask=tcr_mask
).last_hidden_state
```

TCR-BERT tokenizer 必须接收空格分隔的残基：

```text
C A S S L G Q A Y E Q Y F
```

其特殊 token 为 `*`（CLS）、`|`（SEP）、`$`（PAD）、`?`（UNK）和
`.`（MASK）。缓存前必须校验剔除特殊 token 后的 token 数与 CDR3 残基数一致；
禁止把未分隔的 CDR3 静默编码成 `[UNK]`。

同样删除特殊 token，只保留残基级表示：

```text
[batch, cdr3_length, d_tcrbert]
```

不要只使用 `[CLS]` 向量，因为后面的 cross-attention 需要残基级表示。

---

# 七、Adapter 设计

不同预训练模型的隐藏维度不同，因此必须投影到同一个维度。

建议统一：

```text
d_model = 128
```

Adapter 结构：

```python
class SequenceAdapter(nn.Module):
    def __init__(self, input_dim, model_dim=128):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, hidden):
        return self.adapter(hidden)
```

三个分支分别拥有自己的 Adapter：

```text
PeptideAdapter
HLAAdapter
TCRAdapter
```

不要共享 Adapter，因为三种序列具有不同的数据分布。

输出形状：

```text
peptide_feature: [B, Lp, 128]
hla_feature:     [B, 34, 128]
tcr_feature:     [B, Lt, 128]
```

---

# 八、Cross-attention 模块

原始 UnifyImmun 中，peptide 作为 Query，HLA 或 TCR 作为 Key 和 Value。论文使用一个单层、单头 cross-attention。`Unifyimmun.pdf`

增强版本建议使用：

```text
1层 Cross-Attention
4个 attention heads
d_model = 128
dropout = 0.1
```

## 8.1 pHLA 分支

```python
phla_feature, phla_attention = phla_cross_attention(
    query=peptide_feature,
    key=hla_feature,
    value=hla_feature,
    key_padding_mask=hla_padding_mask
)
```

输出：

```text
[B, Lp, 128]
```

## 8.2 pTCR 分支

```python
ptcr_feature, ptcr_attention = ptcr_cross_attention(
    query=peptide_feature,
    key=tcr_feature,
    value=tcr_feature,
    key_padding_mask=tcr_padding_mask
)
```

输出：

```text
[B, Lp, 128]
```

## 8.3 Cross-attention block

推荐完整结构：

```text
Cross-Attention
→ Residual connection
→ LayerNorm
→ Feed Forward 128→512→128
→ Residual connection
→ LayerNorm
```

---

# 九、Pooling 与分类头

不建议简单 flatten，因为使用 PLM 后序列长度可能变化，而且 flatten 会让模型强依赖 padding 长度。

建议使用 masked attention pooling：

```python
score = pooling_linear(sequence_feature)
score = score.masked_fill(~mask, -1e9)
weight = torch.softmax(score, dim=1)
pooled = torch.sum(sequence_feature * weight, dim=1)
```

输出：

```text
[B, 128]
```

分类头：

```text
Linear 128→64
GELU
Dropout 0.2
Linear 64→2
```

继续使用原论文的 two-class cross-entropy：

```python
classification_loss = F.cross_entropy(logits, labels)
```

这样便于和原始 UnifyImmun 对比。原论文也是使用 cross-entropy，并通过独立的 pHLA、pTCR 输出头进行分类。`A unified cross-attention model for predicting antigen binding specificity to both HLA and TCR molecules.pdf`

---

# 十、阶段 1：冻结三个 PLM，训练新加入模块

这一阶段的目标不是立即微调大模型，而是先让：

- Adapter；
- Cross-attention；
- Pooling；
- Classifier；

学会如何使用已有的预训练表示。

## 10.1 冻结参数

```python
for param in peptide_esmc.parameters():
    param.requires_grad = False

for param in hla_esmc.parameters():
    param.requires_grad = False

for param in tcr_bert.parameters():
    param.requires_grad = False
```

保持：

```python
peptide_esmc.eval()
hla_esmc.eval()
tcr_bert.eval()
```

这是为了关闭 PLM 内部的 dropout，确保缓存表示稳定。

## 10.2 默认预缓存 embedding

由于 PLM 完全冻结，阶段 1 默认对所有唯一序列预计算：

```text
unique peptide → ESM-C hidden states
unique HLA → ESM-C hidden states
unique CDR3 → TCR-BERT hidden states
```

第一版保存为：

```text
分片 HDF5 hidden/mask/length
+
SQLite sequence → shard/row 索引
```

缓存内容：

```text
sequence
hidden_state
attention_mask
sequence_length
model_version
```

推荐使用 float16 或 bfloat16 存储。

不要只缓存 pooled embedding，要缓存 residue-level hidden states，因为后续需要 cross-attention。

缓存 manifest 必须记录 checkpoint 路径、checkpoint 指纹、hidden size、最大长度、
数据类型、序列数量和 shard 信息。阶段 1 默认只缓存 train/validation（固定宽度
float16 预计约 21 GB）；独立测试集按评估需要分批扩展，避免全部测试集同时缓存
超过 30 GB。三个随机种子共享同一份缓存；在线 PLM forward 只作为 smoke test
和调试后备，不作为阶段 1 正式训练路径。

在线调试必须显式使用 `--no-cache --limit N`，并使用独立的
`online_batch_size=32`，防止误将百万级正式训练切换为在线 PLM forward。

---

# 十一、阶段 1A：无 FGM 的分类头预热

刚初始化的 Adapter 和 cross-attention 输出是随机的，此时直接加入 FGM 容易不稳定。

先进行短暂预热：

```text
pHLA：2 epochs
pTCR：2 epochs
FGM：关闭
PLM：全部冻结
```

优化器：

```python
AdamW(
    lr=3e-4,
    weight_decay=1e-2
)
```

建议分别设置参数组：

| 模块 | 学习率 |
|---|---:|
| Adapter | 3×10⁻⁴ |
| Cross-attention | 2×10⁻⁴ |
| Pooling | 3×10⁻⁴ |
| Classifier | 3×10⁻⁴ |

其他参数：

```text
batch size：1024，使用缓存时
gradient clipping：1.0
warm-up：总 step 的 5%
scheduler：cosine decay
mixed precision：bf16 或 fp16
```

如果未缓存 embedding：

```text
batch size：32-128
```

---

# 十二、阶段 1B：冻结 PLM 的两阶段交替训练

原始 UnifyImmun 的核心训练方式是：

1. pHLA 阶段冻结 TCR 分支；
2. pTCR 阶段冻结 HLA 分支；
3. peptide 分支在两个阶段持续更新；
4. 两个阶段交替直到收敛。`Unifyimmun.pdf`

在 PLM 冻结阶段，“持续更新 peptide encoder”实际对应：

```text
持续更新 Peptide Adapter
```

因为 ESM-C 本身暂时冻结。

## 12.1 一个训练 Round

```text
Round n
├── Phase A：训练 pHLA 1 epoch
└── Phase B：训练 pTCR 1 epoch
```

建议执行：

```text
5-10 rounds
```

第一版严格按“一任务一个完整 epoch”执行，不人为平衡 optimizer step。Fold 1
中 pHLA 样本约为 pTCR 的 4 倍，因此 SwanLab 必须记录每个任务的样本数、
optimizer step 数、step 比例和共享 Peptide Adapter 梯度范数。等 step 训练仅
作为后续消融实验。

## 12.2 Phase A：pHLA 训练

参与 forward：

```text
Peptide ESM-C
Peptide Adapter
HLA ESM-C
HLA Adapter
pHLA Cross-Attention
pHLA Pooling
pHLA Classifier
```

更新参数：

```text
Peptide Adapter
HLA Adapter
pHLA Cross-Attention
pHLA Pooling
pHLA Classifier
```

冻结：

```text
全部 PLM
TCR Adapter
pTCR Cross-Attention
pTCR Classifier
```

## 12.3 Phase B：pTCR 训练

参与 forward：

```text
Peptide ESM-C
Peptide Adapter
TCR-BERT
TCR Adapter
pTCR Cross-Attention
pTCR Pooling
pTCR Classifier
```

更新：

```text
Peptide Adapter
TCR Adapter
pTCR Cross-Attention
pTCR Pooling
pTCR Classifier
```

冻结：

```text
全部 PLM
HLA Adapter
pHLA Cross-Attention
pHLA Classifier
```

这样，Peptide Adapter 会同时吸收：

```text
HLA呈递相关信息
+
TCR识别相关信息
```

## 12.4 阶段 1B 的 FGM

阶段 1B 沿用旧版 UnifyImmun 的 FGM 双 backward 训练逻辑：

```text
clean forward
→ clean CE backward
→ 根据 clean gradient 生成 FGM 参数扰动
→ adversarial forward
→ adversarial CE backward
→ 恢复原始参数
→ gradient clipping
→ optimizer step
```

攻击范围必须按模块对象显式指定，不使用旧实现中容易误命中的：

```python
"projection" in parameter_name
```

pHLA 阶段只攻击：

```text
Peptide Adapter
HLA Adapter
```

pTCR 阶段只攻击：

```text
Peptide Adapter
TCR Adapter
```

不攻击：

```text
冻结的 ESM-C / TCR-BERT
Cross-attention
Pooling
Classifier
```

默认参数：

```text
epsilon：1.0
adversarial loss：two-class cross-entropy
总梯度：clean CE 梯度 + adversarial CE 梯度
```

为兼容旧模型，FGM 工具仍应在对抗 forward 期间冻结 BatchNorm running
statistics；新模型使用 LayerNorm，不依赖该兼容逻辑。

---

# 十三、FGM 参数对抗训练

本方案不采用 VAT，而是沿用旧版代码中的 FGM 双 backward 训练方式。FGM
直接扰动当前训练分支的 Adapter 参数，不修改离散氨基酸序列，也不扰动冻结
PLM 的参数。

每个 batch 的训练顺序为：

```text
clean CE backward
→ FGM attack(epsilon=1.0)
→ adversarial CE backward
→ FGM restore
→ optimizer step
```

## 13.1 pHLA 阶段

只攻击以下模块的可训练参数：

```text
Peptide Adapter
HLA Adapter
```

## 13.2 pTCR 阶段

只攻击以下模块的可训练参数：

```text
Peptide Adapter
TCR Adapter
```

## 13.3 损失函数

```text
L_clean = CE(clean_logits, labels)
L_adversarial = CE(adversarial_logits, labels)
L_phase = L_clean + L_adversarial
```

FGM 不额外引入 KL divergence 或 `lambda_vat`。默认：

```text
epsilon：1.0
FGM：阶段 1A 关闭，阶段 1B 开启
```

SwanLab 需要分别记录 clean loss、adversarial loss、epsilon、攻击参数数量和
攻击模块梯度范数，便于判断 FGM 是否实际生效。

---

# 十四、阶段 1 的停止标准

阶段 1B 不再使用 pHLA 与 pTCR 的联合分数进行早停。两个分类头分别选择
各自的最佳模型，并独立维护 early-stopping 状态。

每个任务完成一个有效训练 round 后，在该任务的验证集上评估 AUROC 和 AUPR，
并定义任务分数：

$$
S_{\text{pHLA}}
=
\frac{
AUROC_{\text{pHLA}}
+
AUPR_{\text{pHLA}}
}{2}
$$

$$
S_{\text{pTCR}}
=
\frac{
AUROC_{\text{pTCR}}
+
AUPR_{\text{pTCR}}
}{2}
$$

即代码中的：

```text
task_score = (task AUROC + task AUPR) / 2
```

两个任务分别维护 `best_task_score`、`no_improve_rounds` 和 `stopped`：

```text
max rounds：10
early-stopping patience：3 个该任务的有效训练 rounds
min delta：1e-4
```

只有当当前任务分数严格大于
`best_task_score + 1e-4` 时，才视为提升并将该任务的 patience 清零；否则
`no_improve_rounds` 加 1。某任务连续 3 个有效训练 rounds 未达到上述提升条件时，
仅停止该任务后续的训练与验证，另一个尚未停止的任务继续训练。两个任务都停止，
或全局达到 10 rounds 时，阶段 1B 结束。

同一次训练的 checkpoint 文件共享同一个时间戳：

```text
warmup_last_<timestamp>.pt
last_<timestamp>.pt
best_phla_<timestamp>.pt
best_ptcr_<timestamp>.pt
```

训练过程不再生成 `best_joint.pt`。`best_phla_<timestamp>.pt` 和
`best_ptcr_<timestamp>.pt` 仅在对应任务自身的验证分数提升时更新，且分别只保存
共享 Peptide Adapter 与对应任务的 receptor Adapter、cross-attention、pooling
和分类头，用于独立推理。

用于断点续训的 `last_<timestamp>.pt` 保存完整双任务模型、optimizer、scheduler、
AMP scaler、当前 round、随机数状态、完整配置、缓存 manifest/模型指纹、
SwanLab run id，以及两个任务各自的最佳验证指标和 early-stopping 状态。

---

# 十五、阶段 2：逐步解冻 PLM 顶层

完成冻结训练后，再微调 PLM。

不要一次性同时解冻所有编码器。

推荐分成两个子阶段。

---

## 15.1 阶段 2A：先解冻 Peptide ESM-C 顶层

原因是 peptide encoder 是两个任务共享的中心。

解冻：

```text
Peptide ESM-C 最后 2 个 Transformer blocks
Peptide Adapter
三个下游分支模块
```

继续冻结：

```text
HLA ESM-C
TCR-BERT
```

训练：

```text
2 个完整 alternating rounds
```

学习率：

| 参数 | 学习率 |
|---|---:|
| Peptide ESM-C 最后一层 | 1×10⁻⁵ |
| Peptide ESM-C 倒数第二层 | 5×10⁻⁶ |
| Adapter | 5×10⁻⁵ |
| Cross-attention | 1×10⁻⁴ |
| Pooling/Classifier | 1×10⁻⁴ |

使用：

```text
weight decay：0.01
gradient clipping：1.0
warm-up：5%
bf16
gradient checkpointing
```

---

## 15.2 阶段 2B：解冻三个编码器的顶层

随后解冻：

```text
Peptide ESM-C：最后 2 层
HLA ESM-C：最后 1-2 层
TCR-BERT：最后 2 层
```

其余底层继续冻结。

训练：

```text
3-6 个 alternating rounds
```

推荐参数：

| 模块 | 学习率 |
|---|---:|
| PLM 顶层 | 5×10⁻⁶～1×10⁻⁵ |
| Adapter | 3×10⁻⁵～5×10⁻⁵ |
| Cross-attention | 5×10⁻⁵～1×10⁻⁴ |
| Classifier | 1×10⁻⁴ |

---

# 十六、显存不足时的具体处理

两个 ESM-C 300M 加一个 TCR-BERT，如果同时保留梯度，单张 24 GB GPU 可能比较紧张。

由于训练是分阶段的，可以只把当前阶段需要的模型放在 GPU。

## pHLA 阶段

```text
GPU：
Peptide ESM-C
HLA ESM-C
pHLA branch

CPU：
TCR-BERT
pTCR branch
```

## pTCR 阶段

```text
GPU：
Peptide ESM-C
TCR-BERT
pTCR branch

CPU：
HLA ESM-C
pHLA branch
```

建议：

```text
micro batch size：8 或 16
gradient accumulation：8 或 16
effective batch size：128
bf16：开启
gradient checkpointing：开启
```

实际显存占用取决于具体 checkpoint、优化器和序列长度，需要从 micro batch 8 开始测试。

---

# 十七、完整训练循环伪代码

```python
best_joint_score = -float("inf")
patience = 0

for round_id in range(max_rounds):

    # =====================================
    # Phase A: pHLA
    # =====================================
    set_trainable_modules(task="phla", stage=current_stage)

    for batch in phla_train_loader:
        optimizer.zero_grad()

        peptide_feat = encode_peptide(batch["peptide"])
        hla_feat = encode_hla(batch["hla"])

        clean_logits = phla_forward(
            peptide_feat=peptide_feat,
            receptor_feat=hla_feat,
        )

        ce_loss = cross_entropy(
            clean_logits,
            batch["label"],
        )

        ce_loss.backward()

        fgm.attack(
            modules=[model.peptide_adapter, model.hla_adapter],
            epsilon=1.0,
        )
        adversarial_logits = phla_forward(
            peptide_feat=peptide_feat,
            receptor_feat=hla_feat,
        )
        adversarial_loss = cross_entropy(
            adversarial_logits,
            batch["label"],
        )
        adversarial_loss.backward()
        fgm.restore()

        clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        scheduler.step()

    phla_metrics = evaluate(phla_val_loader, task="phla")

    # =====================================
    # Phase B: pTCR
    # =====================================
    set_trainable_modules(task="ptcr", stage=current_stage)

    for batch in ptcr_train_loader:
        optimizer.zero_grad()

        peptide_feat = encode_peptide(batch["peptide"])
        tcr_feat = encode_tcr(batch["cdr3"])

        clean_logits = ptcr_forward(
            peptide_feat=peptide_feat,
            receptor_feat=tcr_feat,
        )

        ce_loss = cross_entropy(
            clean_logits,
            batch["label"],
        )

        ce_loss.backward()

        fgm.attack(
            modules=[model.peptide_adapter, model.tcr_adapter],
            epsilon=1.0,
        )
        adversarial_logits = ptcr_forward(
            peptide_feat=peptide_feat,
            receptor_feat=tcr_feat,
        )
        adversarial_loss = cross_entropy(
            adversarial_logits,
            batch["label"],
        )
        adversarial_loss.backward()
        fgm.restore()

        clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        scheduler.step()

    ptcr_metrics = evaluate(ptcr_val_loader, task="ptcr")

    joint_score = (
        phla_metrics["auroc"]
        + phla_metrics["aupr"]
        + ptcr_metrics["auroc"]
        + ptcr_metrics["aupr"]
    ) / 4

    if joint_score > best_joint_score:
        best_joint_score = joint_score
        save_checkpoint()
        patience = 0
    else:
        patience += 1

    if patience >= 3:
        break
```

---

# 十八、推荐超参数总表

## 冻结 PLM 阶段

| 参数 | 推荐值 |
|---|---:|
| d_model | 128 |
| cross-attention 层数 | 1 |
| attention heads | 4 |
| adapter dropout | 0.1 |
| classifier dropout | 0.2 |
| batch size | 1024，缓存 PLM 特征时 |
| optimizer | AdamW |
| adapter LR | 3×10⁻⁴ |
| cross-attention LR | 2×10⁻⁴ |
| weight decay | 0.01 |
| warm-up | 5% |
| gradient clipping | 1.0 |
| FGM | 阶段 1A 关闭，阶段 1B 开启 |
| FGM epsilon | 1.0 |
| FGM 攻击范围 | 当前任务的 Peptide Adapter + Receptor Adapter |
| alternating rounds | 5-10 |
| early-stop patience | 3 rounds |

## 部分解冻阶段

| 参数 | 推荐值 |
|---|---:|
| micro batch | 8-16 |
| effective batch | 128 |
| PLM LR | 5×10⁻⁶～1×10⁻⁵ |
| adapter LR | 3×10⁻⁵～5×10⁻⁵ |
| head LR | 1×10⁻⁴ |
| Peptide ESM-C 解冻 | 最后 2 层 |
| HLA ESM-C 解冻 | 最后 1-2 层 |
| TCR-BERT 解冻 | 最后 2 层 |
| alternating rounds | 3-6 |
| gradient checkpointing | 开启 |
| mixed precision | bf16 优先 |

---

# 十九、训练完成后的评估

每个任务至少报告：

```text
AUROC
AUPR
Accuracy
Precision
Recall
F1
MCC
Specificity
```

另外报告：

```text
Top-100 PPV
Top-1,000 PPV
Top-5,000 PPV
```

评估集合：

```text
Random-pair test
Peptide-unseen test
TCR-unseen test
Both-unseen test
```

阈值必须在 validation set 上确定，例如：

```text
选择 validation MCC 最大对应的阈值
```

不能在测试集上选择阈值。

建议再进行温度缩放：

```text
validation logits
→ temperature scaling
→ calibrated probability
```

因为你后续要生成 peptide × CDR3 概率矩阵，概率校准非常重要。

---

# 二十、必须完成的消融实验

至少训练以下版本：

| 版本 | 作用 |
|---|---|
| 原始 UnifyImmun | baseline |
| PLM + Adapter，全部冻结，无 FGM | 验证 PLM 表征本身 |
| PLM + Adapter，全部冻结，有 FGM | 验证 FGM |
| 仅解冻 Peptide ESM-C 顶层 | 验证共享 peptide encoder |
| 三个 encoder 部分解冻 | 完整路线 B |
| ESM-C peptide + 原始 HLA/TCR encoder | 分析 peptide PLM 贡献 |
| 原始 peptide encoder + ESM-C/TCR-BERT | 分析 receptor PLM 贡献 |

每个实验至少运行 3 个随机种子：

```text
42
3407
2026
```

最终报告：

```text
mean ± standard deviation
```

---

# 二十一、针对你的肝癌数据集继续微调

完成通用 pHLA/pTCR 训练后，再进入 HCC 微调。

## HCC 微调建议

从路线 B 的最佳 checkpoint 开始：

```text
冻结 PLM 底层
保持顶层部分解冻
训练 Adapter + Cross-attention + Classifier
```

学习率：

| 模块 | HCC 微调学习率 |
|---|---:|
| PLM 顶层 | 2×10⁻⁶～5×10⁻⁶ |
| Adapter | 2×10⁻⁵ |
| Cross-attention | 3×10⁻⁵ |
| Classifier | 5×10⁻⁵ |

为了防止小规模 HCC 数据导致灾难性遗忘，每个 batch 可以混入通用数据：

```text
80% HCC 数据
20% 原始通用 pTCR/pHLA 数据
```

如果 HCC 数据以患者为单位，必须按患者划分：

```text
训练患者 ∩ 测试患者 = ∅
```

如果目标是筛选未见 peptide，还应同时采用 peptide-group split。

最终得到的模型可用于：

```text
HCC peptide 集合 × HCC CDR3β 集合
→ 结合概率矩阵
→ peptide 广度、强度、多样性综合评分
→ 肝癌候选肽段排名
```

---

# 二十二、建议实际执行顺序

建议按下面顺序逐步实施，不要一次性全部改造：

```text
V0：复现原始 UnifyImmun
↓
V1：接入三个 PLM，但全部冻结，不加 FGM
↓
V2：冻结 PLM，加入两阶段训练
↓
V3：冻结 PLM，加入定向 Adapter FGM
↓
V4：解冻 Peptide ESM-C 最后两层
↓
V5：解冻三个 PLM 顶层，形成完整路线 B
↓
V6：在 HCC 数据集上微调
↓
V7：生成 peptide × CDR3 概率矩阵并排名
```

其中 **V5 才是完整的路线 B 模型**，V1-V4 是为了定位每一个模块的真实贡献，避免最终模型性能变化时无法判断问题来源。

---

**Sources:**

- [GitHub - Biohub/esm · GitHub](https://github.com/evolutionaryscale/esm?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
