# Unified PLM Stage-1 Model Architecture Analysis

> Generated 2026-07-24 from branch `PLM` (commit `6860801`).
> Every claim is annotated with `file:line_number` references.

---

## 1. Overall Architecture Diagram (ASCII)

```
                       Peptide Sequences                     Receptor Sequences
                      "SIINFEKL" x N                         "HLA-A0201..." / "CASS...F" x N
                            │                                          │
                            ▼                                          ▼
              ┌─────────────────────────┐          ┌──────────────────────────────────────┐
              │   Frozen ESM-C Encoder   │          │  Frozen ESM-C (HLA) / TCR-BERT (TCR) │
              │   hidden_size = 960      │          │  HLA hidden_size = 960               │
              │   max_length = 15        │          │  TCR hidden_size = 768               │
              │   (encoders.py:127-196)  │          │  HLA max_length = 34                 │
              │   dtype: float16         │          │  TCR max_length = 34                 │
              └────────────┬────────────┘          │  (encoders.py:127-277)                │
                           │                       │  dtype: float16                       │
                           ▼                       └──────────────┬───────────────────────┘
              ┌─────────────────────────┐                          │
              │  Input tensor           │                          ▼
              │  [B, 15, 960] float16   │          ┌──────────────────────────────────────┐
              └────────────┬────────────┘          │  Input tensor                        │
                           │                       │  HLA: [B, 34, 960] float16           │
                           ▼                       │  TCR: [B, 34, 768] float16           │
              ┌─────────────────────────┐          └──────────────┬───────────────────────┘
              │  Peptide Adapter        │                          │
              │  Linear(960,256)        │                          ▼
              │  -> GELU               │          ┌──────────────────────────────────────┐
              │  -> Dropout(0.1)       │          │  Receptor Adapter                    │
              │  -> Linear(256,128)     │          │  HLA: HLAAdapter (960->256->128)     │
              │  -> LayerNorm(128)     │          │  TCR: TCRAdapter (768->256->128)     │
              │  (attention.py:9-28)   │          │  Same architecture as peptide         │
              └────────────┬────────────┘          │  (attention.py:9-28)                 │
                           │                       └──────────────┬───────────────────────┘
                           ▼                                      │
              ┌─────────────────────────┐                          │
              │  Adapter Output          │                          │
              │  [B, 15, 128] float32    │                          │
              └────────────┬────────────┘                          │
                           │                                       ▼
                           │                       ┌──────────────────────────────────────┐
                           │                       │  Adapter Output                      │
                           │                       │  [B, L_rec, 128] float32             │
                           │                       └──────────────┬───────────────────────┘
                           │                                      │
                           └──────────────┬───────────────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  Cross-Attention Block         │
                          │  query = peptide [B,15,128]    │
                          │  key/value = receptor [B,Lr,128]│
                          │  nn.MultiheadAttention(128,4)  │
                          │  batch_first=True              │
                          │  (attention.py:31-86)          │
                          │                                │
                          │  peptide attends to receptor   │
                          │  receptor_mask blocks padding  │
                          │  positions in key/value        │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │  [B, 15, 128]                  │
                          │  (residual + ff residual)      │
                          │  (mask fill peptide padding)   │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │  Masked Attention Pooling       │
                          │  Linear(128,1) -> softmax      │
                          │  weighted sum over residues     │
                          │  (attention.py:89-108)         │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │  [B, 128] (pooled vector)      │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │  Binding Classifier             │
                          │  Linear(128,64) -> GELU        │
                          │  -> Dropout(0.2)               │
                          │  -> Linear(64,2)               │
                          │  (attention.py:111-122)        │
                          └──────────────┬────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │  [B, 2] logits                  │
                          │  (class 0: non-binding,        │
                          │   class 1: binding)             │
                          │  Loss: CrossEntropyLoss         │
                          └───────────────────────────────┘

        ┌─── Shared Components ─────────────────────────────────────┐
        │  One Peptide Adapter is shared across both tasks           │
        │  pHLA branch: HLAAdapter + pHLA_CrossAttention +          │
        │               pHLA_Pooling + pHLA_Classifier              │
        │  pTCR branch: TCRAdapter + pTCR_CrossAttention +          │
        │               pTCR_Pooling + pTCR_Classifier              │
        │  (model.py:45-60)                                         │
        └──────────────────────────────────────────────────────────┘
```

### Dual-Task Architecture Summary

The model has **one shared Peptide Adapter** and **two independent receptor pipelines**:

| Component | Shared? | pHLA branch | pTCR branch |
|---|---|---|---|
| PeptideAdapter | **Shared** | Used | Used |
| ReceptorAdapter | Separate | `hla_adapter` (960 in) | `tcr_adapter` (768 in) |
| CrossAttentionBlock | Separate | `phla_cross_attention` | `ptcr_cross_attention` |
| MaskedAttentionPooling | Separate | `phla_pooling` | `ptcr_pooling` |
| BindingClassifier | Separate | `phla_classifier` | `ptcr_classifier` |

`model.py:62-102` (`forward`) and `model.py:104-121` (`task_modules`) and `model.py:124-135` (`fgm_modules`)

---

## 2. Detailed Component Analysis

### 2.1 Frozen Encoders (encoders.py)

#### 2.1.1 `FrozenResidueEncoder` (base class, `encoders.py:107-125`)

Abstract base. Methods:
- `encode(sequences) -> ResidueBatch`: raises `NotImplementedError`
- `train(mode=True)`: overridden to force `eval` mode and `False` return, preventing any training of encoder weights
- Attributes: `hidden_size: int`, `checkpoint_fingerprint: str`, `checkpoint_path: Path`, `max_length: int`

`ResidueBatch` dataclass (`encoders.py:16-19`):
- `hidden: Tensor` -- [B, max_length, hidden_dim], float16
- `mask: Tensor` -- [B, max_length], bool (True = real residue)
- `lengths: Tensor` -- [B], int64 (actual residue count)

#### 2.1.2 `ESMCResidueEncoder` (`encoders.py:127-196`)

```text
Input:  sequences: Sequence[str]  (e.g. ["SIINFEKL", ...])
        dtype: "float16" (on CUDA), "float32" (on CPU)
        max_length: 15 (peptide) or 34 (HLA)

Internal:
  _normalise_sequence: strip whitespace, remove hyphens, uppercase
  tokenizer: AutoTokenizer (from checkpoint, all_special_ids tracked)
  model: AutoModelForMaskedLM (frozen, requires_grad=False, eval mode)
  model.config.d_model (= hidden_size): 960

  Tokenization: add_special_tokens=True, padding=True, truncation=False
  -> input_ids: [B, L_tok], attention_mask: [B, L_tok]
  -> model(input_ids, attention_mask)
  -> last_hidden_state: [B, L_tok, 960]

  _pad_residue_hidden:
    Strip BOS/EOS/PAD/CLS/SEP tokens from each sequence
    Validate: kept residue count == expected len(sequence)
    Pad to max_length with zeros

Output: ResidueBatch
  hidden: [B, max_length, 960]  float16
  mask:   [B, max_length]       bool
  lengths:[B]                   int64
```

`encoders.py:164-196` (encode), `encoders.py:61-104` (_pad_residue_hidden)

#### 2.1.3 `TCRBertResidueEncoder` (`encoders.py:199-277`)

```text
Input:  sequences: Sequence[str]  (e.g. ["CASSYSTNSYEQYF", ...])
        dtype: "float16" (on CUDA)
        max_length: 34

Internal:
  _normalise_sequence: strip whitespace, remove hyphens, uppercase
  " ".join(sequence):  space-separated characters for BERT tokenizer
  tokenizer: AutoTokenizer (use_fast=False, from checkpoint)
  model: base_model only (classifier head stripped)
         AutoModelForSequenceClassification loaded, then .base_model extracted
         frozen, requires_grad=False, eval mode
  model.config.hidden_size: 768

  Tokenization: add_special_tokens=True, padding=True, truncation=False
  -> input_ids: [B, L_tok], attention_mask, token_type_ids
  -> model(input_ids, attention_mask, token_type_ids)
  -> last_hidden_state: [B, L_tok, 768]

  _pad_residue_hidden:
    Strip special tokens
    Validate alignment
    Pad to max_length

Output: ResidueBatch
  hidden: [B, 34, 768]  float16
  mask:   [B, 34]       bool
  lengths:[B]           int64
```

`encoders.py:252-277` (encode)

#### 2.1.4 `_pad_residue_hidden` (shared alignment function, `encoders.py:61-104`)

```text
Input:
  hidden:          [B, L_tok, D]  raw token-level hidden states
  input_ids:       [B, L_tok]     token IDs from tokenizer
  attention_mask:  [B, L_tok]     1=active token
  special_token_ids: set[int]     all_special_ids from tokenizer
  expected_lengths: list[int]     len(original_sequence) per sample
  max_length: int                 configured max

Process per row:
  1. keep = attention_mask[row] is True
  2. keep &= input_ids[row] NOT in special_token_ids
  3. row_hidden = hidden[row][keep]   -> [actual_length, D]
  4. Validate: actual_length == expected_lengths[row]
  5. Validate: actual_length <= max_length
  6. Copy to padded result [row, :actual_length, :]
  7. Set mask[row, :actual_length] = True

Output: ResidueBatch
  hidden: [B, max_length, D]  (padded with zeros)
  mask:   [B, max_length]     (True for real residues only)
  lengths:[B]                 (actual residue counts)
```

`encoders.py:61-104`

#### 2.1.5 `OnlineEncoderBundle` (`encoders.py:280-336`)

Wraps both frozen encoders for the `--no-cache` debug path. Only used with `--limit`.

```text
Constructor args:
  esmc_path, tcr_bert_path: Path objects
  peptide_max_length: 15, hla_max_length: 34, tcr_max_length: 34
  device: torch.device
  dtype: "float16"

encode(task, peptide_sequences, receptor_sequences):
  1. ESM-C: encode peptides (max_length=peptide_max_length)
  2. task == "phla": ESM-C encodes HLA (max_length=hla_max_length)
  3. task == "ptcr": TCR-BERT encodes TCR
  Returns dict: peptide_hidden, peptide_mask, receptor_hidden, receptor_mask
```

`encoders.py:280-336` (class), `encoders.py:309-329` (encode)

### 2.2 SequenceAdapter (`attention.py:9-28`)

Projects frozen encoder hidden states from the PLM's native dimension down to the unified model dimension.

```text
Constructor args: input_dim, model_dim=128, hidden_dim=256, dropout=0.1

Structure:
  nn.Sequential(
    nn.Linear(input_dim, hidden_dim),      # input_dim -> 256
    nn.GELU(),
    nn.Dropout(dropout),                    # 0.1
    nn.Linear(hidden_dim, model_dim),       # 256 -> 128
    nn.LayerNorm(model_dim),                # 128
  )

Input:  hidden: [B, L, input_dim]  (960 for peptide/HLA, 768 for TCR)
        mask:   [B, L]             (bool, True = real residue)

Process:
  1. network(hidden) -> [B, L, model_dim=128]
  2. masked_fill(~mask.unsqueeze(-1), 0.0)  -- zero out padding positions

Output: [B, L, 128] float32
```

`attention.py:9-28`

The three adapter instances in the model:

| Adapter | input_dim | Source | `model.py:line` |
|---|---|---|---|
| `peptide_adapter` | 960 | ESM-C | `model.py:46` |
| `hla_adapter` | 960 | ESM-C | `model.py:48` |
| `tcr_adapter` | 768 | TCR-BERT | `model.py:49` |

All share `model_dim=128, hidden_dim=256, dropout=0.1` from `adapter_kwargs` at `model.py:34-38`.

### 2.3 CrossAttentionBlock (`attention.py:31-86`)

Cross-attention where **peptide attends to receptor** (query=peptide, key/value=receptor). This is a standard Transformer-style block with pre-LayerNorm residual connections.

```text
Constructor args: model_dim=128, num_heads=4, feedforward_dim=512, dropout=0.1

Structure:
  self.attention = nn.MultiheadAttention(
    embed_dim=128, num_heads=4, dropout=0.1, batch_first=True
  )                          # head_dim = 128/4 = 32
  self.attention_dropout = nn.Dropout(0.1)
  self.attention_norm = nn.LayerNorm(128)
  self.feedforward = nn.Sequential(
    nn.Linear(128, 512),      # expand
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 128),      # project back
    nn.Dropout(0.1),
  )
  self.feedforward_norm = nn.LayerNorm(128)

Input:  peptide:        [B, Lp, 128]   (adapter output)
        receptor:       [B, Lr, 128]   (adapter output)
        peptide_mask:   [B, Lp]        (bool)
        receptor_mask:  [B, Lr]        (bool)
        return_attention: bool (default False)

Process:
  1. Validate masks have at least one True per sample (peptide and receptor)
  2. Cross-attention:
     query = peptide             [B, Lp, 128]
     key = receptor              [B, Lr, 128]
     value = receptor            [B, Lr, 128]
     key_padding_mask = ~receptor_mask   (True = ignore this position)
     average_attn_weights = False        (return per-head weights)
     need_weights = return_attention
     -> attended: [B, Lp, 128]
     -> weights: [B, num_heads, Lp, Lr] or None
  3. Residual + Norm:
     hidden = LayerNorm(peptide + Dropout(attended))   -> [B, Lp, 128]
  4. Feedforward + Residual + Norm:
     hidden = LayerNorm(hidden + Feedforward(hidden))  -> [B, Lp, 128]
  5. Mask fill peptide padding positions:
     hidden = hidden.masked_fill(~peptide_mask.unsqueeze(-1), 0.0)

Output: hidden: [B, Lp, 128]
        weights: [B, 4, Lp, Lr] or None
```

`attention.py:31-86`

### 2.4 MaskedAttentionPooling (`attention.py:89-108`)

Learned attention pooling over peptide residues after cross-attention.

```text
Constructor args: model_dim=128

Structure:
  self.score = nn.Linear(128, 1)    # scalar score per residue

Input:  sequence: [B, Lp, 128]   (cross-attention output)
        mask:     [B, Lp]        (bool)
        return_weights: bool

Process:
  1. scores = Linear(128,1)(sequence).squeeze(-1)  -> [B, Lp]
  2. scores masked_fill(~mask, -inf)                -- ignore padded positions
  3. weights = softmax(scores, dim=1)               -> [B, Lp]
  4. pooled = sum(sequence * weights.unsqueeze(-1), dim=1)  -> [B, 128]

Output: pooled:  [B, 128]
        weights: [B, Lp] or None
```

`attention.py:89-108`

### 2.5 BindingClassifier (`attention.py:111-122`)

Two-layer MLP head for binary classification.

```text
Constructor args: model_dim=128, dropout=0.2

Structure:
  nn.Sequential(
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),
  )

Input:  pooled: [B, 128]   (pooled peptide representation)

Output: [B, 2]             (raw logits: [logit_non_binding, logit_binding])
```

`attention.py:111-122`

Loss is `nn.CrossEntropyLoss()` applied to these 2-class logits (`trainer.py:185`).

### 2.6 UnifiedBindingModel (`model.py:29-148`)

The top-level model that wires everything together.

```text
Constructor: __init__(self, config: ModelConfig)
  model_dim = config.model_dim = 128
  num_attention_heads = 4

  Shared:
    self.peptide_adapter = SequenceAdapter(960, model_dim=128, hidden_dim=256, dropout=0.1)

  pHLA branch:
    self.hla_adapter = SequenceAdapter(960, ...)
    self.phla_cross_attention = CrossAttentionBlock(128, 4, 512, 0.1)
    self.phla_pooling = MaskedAttentionPooling(128)
    self.phla_classifier = BindingClassifier(128, 0.2)

  pTCR branch:
    self.tcr_adapter = SequenceAdapter(768, ...)
    self.ptcr_cross_attention = CrossAttentionBlock(128, 4, 512, 0.1)
    self.ptcr_pooling = MaskedAttentionPooling(128)
    self.ptcr_classifier = BindingClassifier(128, 0.2)
```

`model.py:32-60`

#### forward pass (`model.py:62-102`)

```text
forward(task, peptide_hidden, peptide_mask, receptor_hidden, receptor_mask, *, return_attention=False):
  1. peptide = peptide_adapter(peptide_hidden, peptide_mask)
     -> [B, Lp, 128]

  2. If task == "phla":
       receptor = hla_adapter(receptor_hidden, receptor_mask)  -> [B, Lr, 128]
       use phla_cross_attention, phla_pooling, phla_classifier
     elif task == "ptcr":
       receptor = tcr_adapter(receptor_hidden, receptor_mask)  -> [B, Lr, 128]
       use ptcr_cross_attention, ptcr_pooling, ptcr_classifier

  3. sequence, attention_weights = cross_attention(peptide, receptor, p_mask, r_mask)
     -> sequence: [B, Lp, 128]

  4. pooled, pooling_weights = pooling(sequence, peptide_mask)
     -> pooled: [B, 128]

  5. logits = classifier(pooled)  -> [B, 2]

  Returns: BindingOutput(
    logits=[B, 2],
    cross_attention=[B, 4, Lp, Lr] or None,
    pooling_weights=[B, Lp] or None
  )
```

`model.py:62-102`

#### task_modules (`model.py:104-121`)

Returns all modules active for a given task (shared peptide_adapter + task-specific modules). Used by `set_trainable_task`.

#### fgm_modules (`model.py:124-135`)

Returns only the modules that get FGM attack: **shared peptide_adapter + the task's receptor adapter**. Cross-attention, pooling, and classifier are explicitly excluded from FGM perturbation.

| Task | FGM-attacked modules |
|---|---|
| `"phla"` | `peptide_adapter`, `hla_adapter` |
| `"ptcr"` | `peptide_adapter`, `tcr_adapter` |

`model.py:124-135`

#### set_trainable_task (`model.py:137-143`)

Freezes all parameters, then enables only the modules returned by `task_modules(task)`. The cross-attention, pooling, and classifier for the **inactive** task remain frozen during training on the active task.

`model.py:137-143`

---

## 3. Training Flow

### 3.1 Optimizer (`trainer.py:95-130`)

Uses `AdamW` with **four parameter groups**, each with its own learning rate:

| Group Name | Modules | LR | Weight Decay |
|---|---|---|---|
| `adapters` | peptide_adapter + hla_adapter + tcr_adapter | `3e-4` | `1e-2` |
| `cross_attention` | phla_cross_attention + ptcr_cross_attention | `2e-4` | `1e-2` |
| `pooling` | phla_pooling + ptcr_pooling | `3e-4` | `1e-2` |
| `classifiers` | phla_classifier + ptcr_classifier | `3e-4` | `1e-2` |

`trainer.py:95-130` and `configs/plm_stage1.yaml:45-48` / `config.py:65-68`

Note: Even though `set_trainable_task` only enables gradients for the active task's modules at each step, the optimizer is built over **all** modules. The inactive task's parameters simply have zero gradients and do not update.

### 3.2 Scheduler (`trainer.py:133-150`)

Linear warmup followed by cosine decay:

```text
total_steps = steps_per_pair * (warmup_epochs_per_task + max_rounds)
warmup_steps = total_steps * 0.05    (= scheduler_warmup_ratio)

Schedule:
  step < warmup_steps:  lr = step / warmup_steps (linear, starts from 0)
  step >= warmup_steps: lr = 0.5 * (1 + cos(pi * progress))
                        where progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

`trainer.py:133-150` and `config.py:70` / `configs/plm_stage1.yaml:50`

### 3.3 Mixed Precision (`trainer.py:213-218`)

```text
Autocast policy:
  device != cuda or precision == "none":  no autocast (float32)
  precision == "fp16":  torch.autocast("cuda", dtype=float16)
  precision == "bf16":  torch.autocast("cuda", dtype=bfloat16)  [DEFAULT]

GradScaler:
  Only enabled for fp16 autocast (not bf16)
  self.scaler = torch.amp.GradScaler("cuda", enabled=(device==cuda and precision=="fp16"))
```

`trainer.py:196-218` and `config.py:74` / `configs/plm_stage1.yaml:54`

### 3.4 Phase 1A: Warmup (`trainer.py:429-464`)

```text
Purpose:  Per-task warmup WITHOUT FGM adversarial training
Duration: warmup_epochs_per_task = 2 epochs per task (4 epochs total)
Order:    pHLA first, then pTCR

For each task in ("phla", "ptcr"):
  For epoch in 1..warmup_epochs_per_task:
    set_trainable_task(task)        # freeze inactive task modules
    train_epoch(task, use_fgm=False)
    Save: run_dir / "last.pt"

After warmup completes:
  stage = "stage1b"
  Save: run_dir / "warmup_last.pt"
  Save: run_dir / "last.pt"
```

`trainer.py:429-464`

During warmup:
- `set_trainable_task(task)` enables only: `peptide_adapter` + `{task}_adapter` + `{task}_cross_attention` + `{task}_pooling` + `{task}_classifier`
- `use_fgm=False`: only clean loss is computed, no adversarial perturbation

### 3.5 Phase 1B: Alternating with FGM (`trainer.py:467-548`)

```text
Purpose:  Per-task alternating training with FGM adversarial attacks
Duration: max 10 rounds; each task has its own patience=3

Each round:
  1. If pHLA is active, run its training epoch:
     set_trainable_task("phla")
     train_epoch("phla", use_fgm=True)
     -> FGM attacks: peptide_adapter + hla_adapter only

  2. If pTCR is active, run its training epoch:
     set_trainable_task("ptcr")
     train_epoch("ptcr", use_fgm=True)
     -> FGM attacks: peptide_adapter + tcr_adapter only

  3. Validate each active task:
     task_score = (task.auroc + task.aupr) / 2

  4. Track each task independently:
     If task_score > best_task_score[task] + 1e-4:
       Save: run_dir / "best_{task}.pt"
       best_task_score[task] = task_score
       no_improve_rounds[task] = 0
     Else:
       no_improve_rounds[task] += 1

  5. Per-task early stopping:
     If no_improve_rounds[task] >= 3:
       Stop training that task only
     Stop the loop when both tasks have stopped or max_rounds is reached

After completion:
  stage = "complete"
  Save: run_dir / "last.pt"
```

`trainer.py:467-558` and `config.py:61-64` / `configs/plm_stage1.yaml:41-44`

### 3.6 FGM Attack Details (`trainer.py:269-397` and `fgm.py`)

The FGM attack is applied **per-batch** inside `train_epoch`:

```text
For each batch:
  1. Forward clean:
     clean_loss = CrossEntropyLoss(model(task, batch).logits, labels)
     scaler.scale(clean_loss).backward()

  2. FGM attack (if use_fgm):
     fgm.attack(fgm_modules(task), epsilon=1.0)
     -> For each parameter in target modules:
        parameter += epsilon * parameter.grad / ||parameter.grad||
     -> BN: track_running_stats temporarily set to False

  3. Forward adversarial:
     adversarial_loss = CrossEntropyLoss(model(task, batch).logits, labels)
     scaler.scale(adversarial_loss).backward()

  4. FGM restore:  # in finally block
     Restore all attacked parameters to their original values
     Restore BN track_running_stats

  5. Gradient clipping, optimizer step, scheduler step
```

`trainer.py:293-330` and `fgm.py:27-73`

The gradient from the adversarial forward is **accumulated** with the clean gradient (both backpropagated). This is a standard FGM formulation where the total parameter update is based on the sum of clean + adversarial gradients.

### 3.7 Loss Function

`nn.CrossEntropyLoss()` (no label smoothing, no class weights).

`trainer.py:185`

### 3.8 Per-Task Early-Stopping Score

```python
pHLA_score = (pHLA_AUROC + pHLA_AUPR) / 2.0
pTCR_score = (pTCR_AUROC + pTCR_AUPR) / 2.0
```

The two scores, best metrics, patience counters, and stopped flags are independent.

### 3.9 Metrics Computed (`metrics.py:29-75`)

All computed by sklearn via `compute_binary_metrics`:

| Metric | Computation |
|---|---|
| `loss` | Average CrossEntropyLoss over samples |
| `auroc` | `sklearn.metrics.roc_auc_score` |
| `aupr` | AUC of `sklearn.metrics.precision_recall_curve` |
| `accuracy` | `sklearn.metrics.accuracy_score` (threshold=0.5) |
| `mcc` | `sklearn.metrics.matthews_corrcoef` |
| `f1` | `sklearn.metrics.f1_score` (zero_division=0) |
| `precision` | `sklearn.metrics.precision_score` |
| `recall` | `sklearn.metrics.recall_score` |
| `sensitivity` | tp / (tp + fn) |
| `specificity` | tn / (tn + fp) |

`metrics.py:29-75`

### 3.10 Evaluation Mode (`evaluation.py:17-73`)

Standalone evaluation uses `torch.inference_mode()`, same autocast policy, same binary metrics computation. No FGM. Runs `model.eval()` and applies `CrossEntropyLoss` + softmax probabilities.

`evaluation.py:17-73`

---

## 4. Data Flow

### 4.1 From Sequences to Residue Embeddings

**Cached path** (production):
```text
Raw sequences in CSV
  -> precompute_plm_embeddings.py calls build_embedding_cache() for each entity
  -> Frozen encoder.encode() batched, detach().cpu().float16().numpy()
  -> EmbeddingCacheWriter writes HDF5 shards + SQLite index
  -> CachedPairDataset reads CSV, resolves sequence -> CacheLocator via SQLite
  -> CachedPairCollator reads HDF5 shards, returns tensors
  -> Stage1Trainer._move_batch moves to GPU
```

**Online path** (debug only with `--no-cache --limit N`):
```text
Raw sequences in CSV
  -> OnlinePairDataset stores (peptide_seq, receptor_seq, label) tuples
  -> _collate_online_pairs returns raw sequence lists
  -> Stage1Trainer._move_batch calls online_encoders.encode() on-the-fly
  -> Frozen encoders run on GPU, return ResidueBatch
```

`data.py:115-195` (CachedPairDataset), `data.py:197-226` (CachedPairCollator), `data.py:229-299` (OnlinePairDataset), `trainer.py:220-258` (_move_batch)

### 4.2 Cache System (`cache.py`)

**Writing** (`EmbeddingCacheWriter`, `cache.py:39-245`):
- Writes sharded HDF5 files (`part-00000.h5`, `part-00001.h5`, ...) in `embedding_cache/plm_stage1/fold_N/{entity}/shards/`
- Each shard contains datasets: `sequences` (string), `hidden` (float16, [N, max_length, hidden_dim]), `mask` (uint8, [N, max_length]), `lengths` (int16, [N])
- SQLite `index.sqlite` with table `sequence_index(sequence TEXT PK, shard INT, row_idx INT, length INT)`
- `manifest.json` records fingerprint, hidden_dim, max_length, shard metadata
- Shard size: 16384 sequences per shard; chunk size: min(64, shard_size) for HDF5 compression
- Total typical size: ~21 GB (float16) for fold 1 train+val

`cache.py:39-245`

**Reading** (`EmbeddingCache`, `cache.py:248-381`):
- `resolve(sequence) -> CacheLocator`: SQLite lookup by sequence string
- `worker_reader()`: lightweight copy for DataLoader workers (no index loaded, index lookups happen in main process)
- `get_many(locators) -> (hidden, mask)`: groups reads by shard, deduplicates rows within same batch, returns numpy arrays in caller order
- LRU shard handle cache (max 64 open shards)

`cache.py:248-381`

The `CacheLocator` dataclass (`cache.py:21-25`):
```python
@dataclass(frozen=True)
class CacheLocator:
    shard: int    # which HDF5 file
    row: int      # which row within that shard
    length: int   # actual residue count (for validation, not needed at read time)
```

### 4.3 CSV Data Layout

**pHLA CSVs** (`data/data_HLA/`):
- `train_fold_{1..5}.csv`, `val_fold_{1..5}.csv`
- Columns: `peptide`, `HLA`, `label` (0 or 1)
- Additional test sets: `dataset_set.csv`, `independent_set.csv`, `external_set.csv`, `HPV_set.csv`, `neoantigen_set.csv`

**pTCR CSVs** (`data/data_TCR/`):
- `train_fold_{1..5}.csv`, `val_fold_{1..5}.csv`
- Columns: `peptide`, `tcr`, `label` (0 or 1)
- Additional test sets: `covid_set.csv`, `independent_set.csv`, `triple_set.csv`

`data.py:27-43` (split_path), `data.py:46-51` (task_columns)

### 4.4 Batching and Padding

**Cached data path:**
- Sequences are stored at fixed `max_length` in HDF5 (15 for peptide, 34 for HLA/TCR)
- All rows in a batch have the same padded shape [B, max_length, D]
- `peptide_mask` and `receptor_mask` mark real vs padding positions
- No dynamic padding is needed because the cache is fixed-width

**Sequence length filtering:**
- Peptides shorter than 8 or longer than 15 are filtered out during cache construction and dataset loading
- No length filtering on HLA (34 aa pseudo-sequence) or TCR (CDR3, max 34)

`data.py:20-24` (normalise_sequence), `data.py:71-112` (collect_cache_sequences filtering), `config.py:33-36` (min/max lengths)

### 4.5 Batch Shape Summary

At the model forward, for a batch of size B:

| Tensor | pHLA shape | pTCR shape | dtype |
|---|---|---|---|
| `peptide_hidden` | [B, 15, 960] | [B, 15, 960] | float16 (cache) / bf16 (autocast) |
| `peptide_mask` | [B, 15] | [B, 15] | bool |
| `receptor_hidden` | [B, 34, 960] | [B, 34, 768] | float16 / bf16 |
| `receptor_mask` | [B, 34] | [B, 34] | bool |
| `labels` | [B] | [B] | int64 (0 or 1) |

After adapter:
| Tensor | Shape | dtype |
|---|---|---|
| `peptide` | [B, 15, 128] | float32 (inside autocast: bf16) |
| `receptor` | [B, Lr, 128] | float32 (inside autocast: bf16) |

After cross-attention:
| Tensor | Shape |
|---|---|
| `sequence` | [B, 15, 128] |
| `attention_weights` | [B, 4, 15, Lr] (if return_attention=True) |

After pooling:
| Tensor | Shape |
|---|---|
| `pooled` | [B, 128] |
| `pooling_weights` | [B, 15] (if return_weights=True) |

After classifier:
| Tensor | Shape |
|---|---|
| `logits` | [B, 2] |

---

## 5. Key Hyperparameters

Extracted from `configs/plm_stage1.yaml` and `config.py` defaults.

### 5.1 Model Architecture

| Parameter | Value | Source |
|---|---|---|
| `peptide_input_dim` | 960 | `configs/plm_stage1.yaml:11` |
| `hla_input_dim` | 960 | `configs/plm_stage1.yaml:12` |
| `tcr_input_dim` | 768 | `configs/plm_stage1.yaml:13` |
| `adapter_hidden_dim` | 256 | `configs/plm_stage1.yaml:14` |
| `model_dim` | 128 | `configs/plm_stage1.yaml:15` |
| `num_attention_heads` | 4 | `configs/plm_stage1.yaml:16` |
| `feedforward_dim` | 512 | `configs/plm_stage1.yaml:17` |
| `adapter_dropout` | 0.1 | `configs/plm_stage1.yaml:18` |
| `attention_dropout` | 0.1 | `configs/plm_stage1.yaml:19` |
| `classifier_dropout` | 0.2 | `configs/plm_stage1.yaml:20` |
| `peptide_min_length` | 8 | `configs/plm_stage1.yaml:21` |
| `peptide_max_length` | 15 | `configs/plm_stage1.yaml:22` |
| `hla_max_length` | 34 | `configs/plm_stage1.yaml:23` |
| `tcr_max_length` | 34 | `configs/plm_stage1.yaml:24` |

### 5.2 Training

| Parameter | Value | Source |
|---|---|---|
| `batch_size` (cached) | 1024 | `configs/plm_stage1.yaml:35` |
| `online_batch_size` | 32 | `configs/plm_stage1.yaml:36` |
| `num_workers` | 2 | `configs/plm_stage1.yaml:37` |
| `warmup_epochs_per_task` | 2 | `configs/plm_stage1.yaml:38` |
| `max_rounds` | 10 | `configs/plm_stage1.yaml:39` |
| `early_stopping_patience` | 3 | `configs/plm_stage1.yaml:40` |
| `early_stopping_min_delta` | 1e-4 | `configs/plm_stage1.yaml:41` |
| `adapter_lr` | 3e-4 | `configs/plm_stage1.yaml:42` |
| `cross_attention_lr` | 2e-4 | `configs/plm_stage1.yaml:43` |
| `pooling_lr` | 3e-4 | `configs/plm_stage1.yaml:44` |
| `classifier_lr` | 3e-4 | `configs/plm_stage1.yaml:45` |
| `weight_decay` | 1e-2 | `configs/plm_stage1.yaml:46` |
| `scheduler_warmup_ratio` | 0.05 | `configs/plm_stage1.yaml:47` |
| `gradient_clip_norm` | 1.0 | `configs/plm_stage1.yaml:48` |
| `fgm_epsilon` | 1.0 | `configs/plm_stage1.yaml:49` |
| `threshold` | 0.5 | `configs/plm_stage1.yaml:50` |
| `mixed_precision` | bf16 | `configs/plm_stage1.yaml:51` |
| `log_every_steps` | 50 | `configs/plm_stage1.yaml:52` |
| `pin_memory` | true | `configs/plm_stage1.yaml:53` |
| `persistent_workers` | false | `configs/plm_stage1.yaml:54` |
| `device` | auto | `configs/plm_stage1.yaml:55` |

### 5.3 Cache

| Parameter | Value | Source |
|---|---|---|
| `dtype` | float16 | `configs/plm_stage1.yaml:26` |
| `shard_size` | 16384 | `configs/plm_stage1.yaml:27` |
| `precompute_batch_size` | 32 | `configs/plm_stage1.yaml:28` |
| `compression` | null | `configs/plm_stage1.yaml:30` |

### 5.4 Optimizer Parameter Groups

| Group | Modules | LR | Weight Decay |
|---|---|---|---|
| `adapters` | peptide + hla + tcr adapters | 3e-4 | 1e-2 |
| `cross_attention` | phla + ptcr cross-attention blocks | 2e-4 | 1e-2 |
| `pooling` | phla + ptcr pooling | 3e-4 | 1e-2 |
| `classifiers` | phla + ptcr classifier heads | 3e-4 | 1e-2 |

`trainer.py:100-126`

### 5.5 Model Checkpoints

| Checkpoint | When Saved | Contents |
|---|---|---|
| `last.pt` | After every epoch and round | model, optimizer, scheduler, scaler, stage, RNG state, dataloader state |
| `warmup_last.pt` | After warmup completes (end of phase 1A) | Full checkpoint state |
| `best_phla.pt` | When the pHLA score improves by > 1e-4 | Shared peptide path plus the pHLA adapter, attention, pooling, and classifier |
| `best_ptcr.pt` | When the pTCR score improves by > 1e-4 | Shared peptide path plus the pTCR adapter, attention, pooling, and classifier |

The two task checkpoints use `TaskSpecificBindingModel`. They omit optimizer state and
the unused receptor branch, so they are independent inference artifacts rather than
training-resume checkpoints. New training does not create `best_joint.pt`; legacy
`best_joint.pt` files can still be split with
`source/export_plm_task_models.py`.

---

## 6. Component Connectivity Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                        UnifiedBindingModel                         │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │ PeptideAdapter│   │  HLAAdapter  │   │  TCRAdapter  │           │
│  │ (960->128)   │   │ (960->128)   │   │ (768->128)   │           │
│  │ [Shared]     │   │ [pHLA only]  │   │ [pTCR only]  │           │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         │                  │                  │                    │
│         │    ┌─────────────┼──────────────────┼───────┐           │
│         │    │             │                  │       │           │
│         ▼    │             ▼                  ▼       │           │
│  ┌──────────┴┴────────┐  ┌──────────────────────────┐│           │
│  │ pHLA CrossAttention│  │ pTCR CrossAttention      ││           │
│  │ (4-head, 128-dim)  │  │ (4-head, 128-dim)        ││           │
│  └──────────┬─────────┘  └──────────┬───────────────┘│           │
│             │                       │                 │           │
│             ▼                       ▼                 │           │
│  ┌──────────────────┐  ┌──────────────────┐          │           │
│  │ pHLA Pooling     │  │ pTCR Pooling     │          │           │
│  │ (attn over peps) │  │ (attn over peps) │          │           │
│  └────────┬─────────┘  └────────┬─────────┘          │           │
│           │                     │                     │           │
│           ▼                     ▼                     │           │
│  ┌──────────────────┐  ┌──────────────────┐          │           │
│  │ pHLA Classifier  │  │ pTCR Classifier  │          │           │
│  │ (128->64->2)     │  │ (128->64->2)     │          │           │
│  └──────────────────┘  └──────────────────┘          │           │
│                                                       │           │
│  FGM Attack Targets (fgm_modules):                    │           │
│    pHLA: peptide_adapter + hla_adapter                │           │
│    pTCR: peptide_adapter + tcr_adapter                │           │
│                                                       │           │
│  Trainable (task_modules):                            │           │
│    pHLA: peptide + hla adapters + pHLA attn/pool/cls  │           │
│    pTCR: peptide + tcr adapters + pTCR attn/pool/cls  │           │
└──────────────────────────────────────────────────────┘
```

---

## 7. Model Size Estimate

Based on the dimensions above, counting only the **trainable adapter-level parameters** (frozen encoders excluded):

| Component | Parameters (approx) |
|---|---|
| `peptide_adapter` (960->256->128) | 960*256 + 256 + 256*128 + 128 + 2*128 (LN) = 245,760 + 256 + 32,768 + 128 + 256 = ~279K |
| `hla_adapter` (960->256->128) | same = ~279K |
| `tcr_adapter` (768->256->128) | 768*256 + 256 + 256*128 + 128 + 256 = 196,608 + 256 + 32,768 + 128 + 256 = ~230K |
| `phla_cross_attention` | MHA(128,4): 128*128*4 (in_proj) + 128*128 (out_proj) + FF: 128*512 + 512*128 + 2 LN = ~65K + 16K + 131K + 512 = ~213K |
| `ptcr_cross_attention` | same = ~213K |
| `phla_pooling` | 128*1 + 1 = ~129 |
| `ptcr_pooling` | same = ~129 |
| `phla_classifier` | 128*64 + 64 + 64*2 + 2 = 8,192 + 64 + 128 + 2 = ~8.4K |
| `ptcr_classifier` | same = ~8.4K |
| **Total trainable** | **~1.51M parameters** |

This is a very lightweight adapter architecture -- the frozen encoders (ESM-C 300M + TCR-BERT) are orders of magnitude larger at approximately 600M+ parameters.

---

## 8. Performance Characteristics

| Aspect | Detail |
|---|---|
| **Bottleneck dimension** | 128 (`model_dim`) throughout the adapter/attention pathway |
| **Attention heads** | 4 heads of dimension 32 each |
| **Cross-attention pattern** | Peptide (query) attends to receptor (key/value) -- asymmetric |
| **Receptor length** | HLA: 34 pseudo-sequence residues; TCR CDR3: up to 34 amino acids |
| **Peptide length** | Filtered to 8-15 amino acids (min=8, max=15) |
| **Batch size** | 1024 (cached mode), 32 (online mode) |
| **Memory (cache)** | ~21 GB float16 for fold 1 train+val (all three entities) |
| **Training speed** | GPU-optimized with bf16 autocast, gradient scaling for fp16 only |
| **No dynamic padding** | Fixed-width cache eliminates per-batch padding overhead |
| **FGM cost** | 2x forward pass per batch (clean + adversarial), ~2x training time during phase 1B |
