"""
诊断脚本：检查FGM对抗训练对梯度的影响

问题诊断：
  - 原始FGM在attack forward期间，BatchNorm的running_mean/running_var被更新
  - Restore只恢复了权重参数，但running stats已被污染
  - 导致后续epoch BatchNorm表现异常，loss不再下降

解决方案：
  1. FGM_ESM2_Fixed：冻结BN的track_running_stats，防止running stats被更新
  2. FGM_ESM2_NoBN：完全不扰动BN参数，只扰动Linear层
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

# ========== 原始FGM类（有问题）==========
class FGM_ESM2_Original:
    """原始FGM类 - 存在BatchNorm问题"""
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
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ========== 修复版FGM类（推荐）==========
class FGM_ESM2_Fixed:
    """
    修复后的FGM对抗训练类。

    关键修复：
    1. 在attack期间冻结BatchNorm的track_running_stats
    2. 这防止running_mean/running_var在对抗forward时被更新
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_track_backup = {}  # 备份BN的track_running_stats状态

    def attack(self, epsilon=1.0):
        # Step 1: 冻结所有BatchNorm的running stats更新
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.bn_track_backup[name] = module.track_running_stats
                module.track_running_stats = False  # 关键：冻结BN的running stats更新

        # Step 2: 扰动projection层参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone().detach()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self):
        # Step 1: 恢复projection层参数
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

        # Step 2: 恢复BatchNorm的track_running_stats状态
        for name, module in self.model.named_modules():
            if name in self.bn_track_backup:
                module.track_running_stats = self.bn_track_backup[name]
        self.bn_track_backup = {}


# ========== 不扰动BN的简化版FGM ==========
class FGM_ESM2_NoBN:
    """
    简化版FGM：只扰动Linear层参数，跳过BatchNorm参数。
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                # 检查是否是BatchNorm参数
                # BN参数名格式: projection.X.weight/bias (X=2是BN索引)
                # 通过检查参数形状判断：BN weight是一维的，Linear weight是二维的
                is_bn_param = (param.dim() == 1 and name.endswith('.weight')) or name.endswith('.bias')

                if not is_bn_param and param.grad is not None:
                    self.backup[name] = param.data.clone().detach()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

print("=" * 60)
print("FGM对抗训练诊断")
print("=" * 60)

# 初始化模型
model = Mymodel_HLA_ESM2(freeze_esm2=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# 加载一个batch的数据
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

print("\n[正常Forward + Backward]")
outputs1, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss1 = criterion(outputs1, labels)
print(f"  Loss 1: {loss1.item():.4f}")
loss1.backward()

# 检查projection层参数在backward后的值
print("\n[参数值检查 - backward后, attack前]")
for name, param in model.named_parameters():
    if param.requires_grad and "projection" in name:
        print(f"  {name}: mean={param.data.mean().item():.4f}, max={param.data.abs().max().item():.4f}")

print("\n[FGM Attack - epsilon=1.0]")
fgm.attack(epsilon=1.0)

print("\n[参数值检查 - attack后]")
for name, param in model.named_parameters():
    if param.requires_grad and "projection" in name:
        print(f"  {name}: mean={param.data.mean().item():.4f}, max={param.data.abs().max().item():.4f}")

print("\n[FGM Forward + Backward]")
outputs2, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss2 = criterion(outputs2, labels)
print(f"  Loss 2 (attack后): {loss2.item():.4f}")

# 检查loss变化幅度
print(f"\n  Loss变化: {loss2.item() - loss1.item():.4f}")

loss2.backward()

print("\n[FGM Restore]")
fgm.restore()

print("\n[参数值检查 - restore后]")
for name, param in model.named_parameters():
    if param.requires_grad and "projection" in name:
        print(f"  {name}: mean={param.data.mean().item():.4f}, max={param.data.abs().max().item():.4f}")

print("\n[Optimizer Step]")
optimizer.step()
optimizer.zero_grad()

print("\n[参数值检查 - optimizer step后]")
for name, param in model.named_parameters():
    if param.requires_grad and "projection" in name:
        print(f"  {name}: mean={param.data.mean().item():.4f}, max={param.data.abs().max().item():.4f}")

# 再做一个正常的forward看loss
print("\n[最终Forward]")
outputs3, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss3 = criterion(outputs3, labels)
print(f"  Loss 3 (更新后): {loss3.item():.4f}")

print("\n" + "=" * 60)
print("FGM诊断完成")
print("=" * 60)

print("\n[结论]")
print(f"  初始Loss: {loss1.item():.4f}")
print(f"  更新后Loss: {loss3.item():.4f}")
print(f"  Loss下降: {loss1.item() - loss3.item():.4f}")
if abs(loss1.item() - loss3.item()) < 0.01:
    print("  ⚠️ Loss变化很小，可能存在问题!")
else:
    print("  ✓ Loss正常下降")