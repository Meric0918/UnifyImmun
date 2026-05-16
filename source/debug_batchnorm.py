"""
诊断脚本：检查BatchNorm running stats是否被FGM污染
"""

import torch
import torch.nn as nn
import os

from models.esm2_embedding import (
    Mymodel_HLA_ESM2,
    data_load_HLA_ESM2,
    batch_size,
    device,
)

# FGM类
class FGM_ESM2:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

print("=" * 60)
print("BatchNorm Running Stats诊断")
print("=" * 60)

# 初始化模型
model = Mymodel_HLA_ESM2(freeze_esm2=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# 加载数据
train_loader = data_load_HLA_ESM2(type_="train", fold=1, batch_size=batch_size)
pep_inputs, hla_inputs, labels, pep_masks, hla_masks = next(iter(train_loader))

pep_inputs = pep_inputs.to(device)
hla_inputs = hla_inputs.to(device)
labels = labels.to(device)
pep_masks = pep_masks.to(device)
hla_masks = hla_masks.to(device)

# 训练模式
model.train()
model.encoder_H.src_emb.esm2.eval()
model.encoder_P.src_emb.esm2.eval()

fgm = FGM_ESM2(model)

# 获取BatchNorm层 (projection.2)
bn_layer = model.projection[2]  # BatchNorm1d(256)

print("\n[初始BatchNorm状态]")
print(f"  running_mean: mean={bn_layer.running_mean.mean().item():.4f}, max={bn_layer.running_mean.abs().max().item():.4f}")
print(f"  running_var: mean={bn_layer.running_var.mean().item():.4f}, max={bn_layer.running_var.abs().max().item():.4f}")
print(f"  num_batches_tracked: {bn_layer.num_batches_tracked.item()}")

# 记录初始running stats
init_running_mean = bn_layer.running_mean.clone()
init_running_var = bn_layer.running_var.clone()
init_num_batches = bn_layer.num_batches_tracked.item()

print("\n[正常Forward + Backward]")
outputs1, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss1 = criterion(outputs1, labels)
print(f"  Loss: {loss1.item():.4f}")
loss1.backward()

print("\n[BatchNorm状态 - 第1次forward后]")
print(f"  running_mean变化: {(bn_layer.running_mean - init_running_mean).abs().max().item():.6f}")
print(f"  running_var变化: {(bn_layer.running_var - init_running_var).abs().max().item():.6f}")
print(f"  num_batches_tracked: {bn_layer.num_batches_tracked.item()}")

# 记录第1次forward后的running stats
after1_running_mean = bn_layer.running_mean.clone()
after1_running_var = bn_layer.running_var.clone()

print("\n[FGM Attack + Forward]")
fgm.attack(epsilon=1.0)
outputs2, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss2 = criterion(outputs2, labels)
print(f"  Loss (attack后): {loss2.item():.4f}")

print("\n[BatchNorm状态 - FGM forward后]")
print(f"  running_mean变化: {(bn_layer.running_mean - after1_running_mean).abs().max().item():.6f}")
print(f"  running_var变化: {(bn_layer.running_var - after1_running_var).abs().max().item():.6f}")
print(f"  num_batches_tracked: {bn_layer.num_batches_tracked.item()}")

# FGM forward后running stats变化量
fgm_running_mean_change = (bn_layer.running_mean - after1_running_mean).abs().max().item()
fgm_running_var_change = (bn_layer.running_var - after1_running_var).abs().max().item()

print("\n[FGM Restore + Backward + Optimizer Step]")
loss2.backward()
fgm.restore()
optimizer.step()
optimizer.zero_grad()

print("\n[BatchNorm状态 - restore和step后]")
print(f"  running_mean: mean={bn_layer.running_mean.mean().item():.4f}")
print(f"  running_var: mean={bn_layer.running_var.mean().item():.4f}")
print(f"  num_batches_tracked: {bn_layer.num_batches_tracked.item()}")

# 再次正常forward
print("\n[最终Forward]")
outputs3, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss3 = criterion(outputs3, labels)
print(f"  Loss: {loss3.item():.4f}")

print("\n" + "=" * 60)
print("诊断结果")
print("=" * 60)
print(f"  初始Loss: {loss1.item():.4f}")
print(f"  FGM attack后Loss: {loss2.item():.4f}")
print(f"  最终Loss: {loss3.item():.4f}")
print(f"\n  BatchNorm running_mean被FGM污染: {fgm_running_mean_change:.6f}")
print(f"  BatchNorm running_var被FGM污染: {fgm_running_var_change:.6f}")

if fgm_running_mean_change > 0.01 or fgm_running_var_change > 0.01:
    print("\n  ⚠️⚠️⚠️ BatchNorm running stats被FGM严重污染!")
    print("  这是导致训练不稳定的根本原因!")
    print("  解决方案:")
    print("  1. 在FGM restore时也恢复running_mean和running_var")
    print("  2. 或者将BatchNorm换成LayerNorm")
    print("  3. 或者去掉FGM对抗训练")
else:
    print("\n  BatchNorm running stats变化较小")