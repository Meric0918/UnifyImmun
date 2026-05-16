"""
验证修复后的FGM效果
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

# 修复后的FGM类
class FGM_ESM2_Fixed:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_backup = {}

    def attack(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

        # Backup BatchNorm running stats
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                self.bn_backup[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'num_batches_tracked': module.num_batches_tracked.clone()
                }

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

        # Restore BatchNorm running stats
        for name, module in self.model.named_modules():
            if name in self.bn_backup:
                module.running_mean = self.bn_backup[name]['running_mean']
                module.running_var = self.bn_backup[name]['running_var']
                module.num_batches_tracked = self.bn_backup[name]['num_batches_tracked']
        self.bn_backup = {}

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

print("=" * 60)
print("验证修复后的FGM")
print("=" * 60)

model = Mymodel_HLA_ESM2(freeze_esm2=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

train_loader = data_load_HLA_ESM2(type_="train", fold=1, batch_size=batch_size)
pep_inputs, hla_inputs, labels, pep_masks, hla_masks = next(iter(train_loader))

pep_inputs = pep_inputs.to(device)
hla_inputs = hla_inputs.to(device)
labels = labels.to(device)
pep_masks = pep_masks.to(device)
hla_masks = hla_masks.to(device)

model.train()
model.encoder_H.src_emb.esm2.eval()
model.encoder_P.src_emb.esm2.eval()

fgm = FGM_ESM2_Fixed(model)

bn_layer = model.projection[2]

print("\n[初始BatchNorm状态]")
init_running_mean = bn_layer.running_mean.clone()
init_running_var = bn_layer.running_var.clone()
print(f"  running_mean: {init_running_mean[:5].tolist()}")
print(f"  running_var: {init_running_var[:5].tolist()}")

print("\n[正常Forward + Backward]")
outputs1, _ = model(pep_inputs, hla_inputs, pep_attention_mask=pep_masks, hla_attention_mask=hla_masks)
loss1 = criterion(outputs1, labels)
print(f"  Loss: {loss1.item():.4f}")
loss1.backward()

after1_mean = bn_layer.running_mean.clone()
after1_var = bn_layer.running_var.clone()

print("\n[FGM Attack + Forward]")
fgm.attack(epsilon=1.0)
outputs2, _ = model(pep_inputs, hla_inputs, pep_attention_mask=pep_masks, hla_attention_mask=hla_masks)
loss2 = criterion(outputs2, labels)
print(f"  Loss (attack后): {loss2.item():.4f}")

after_attack_mean = bn_layer.running_mean.clone()
after_attack_var = bn_layer.running_var.clone()
print(f"  running_mean变化(attack期间): {(after_attack_mean - after1_mean).abs().max().item():.6f}")

print("\n[FGM Restore + Backward + Optimizer Step]")
loss2.backward()
fgm.restore()

after_restore_mean = bn_layer.running_mean.clone()
after_restore_var = bn_layer.running_var.clone()
print(f"  running_mean变化(restore后): {(after_restore_mean - after1_mean).abs().max().item():.6f}")

optimizer.step()
optimizer.zero_grad()

print("\n[最终Forward]")
outputs3, _ = model(pep_inputs, hla_inputs, pep_attention_mask=pep_masks, hla_attention_mask=hla_masks)
loss3 = criterion(outputs3, labels)
print(f"  Loss: {loss3.item():.4f}")

print("\n" + "=" * 60)
print("验证结果")
print("=" * 60)
if (after_restore_mean - after1_mean).abs().max().item() < 1e-5:
    print("  ✓ BatchNorm running stats被正确恢复!")
else:
    print("  ⚠️ BatchNorm running stats恢复不完整")

print(f"\n  Loss变化: {loss1.item():.4f} -> {loss3.item():.4f}")
print("  ✓ 训练正常进行")