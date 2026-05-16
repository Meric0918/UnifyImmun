"""
诊断脚本：检查模型训练过程中的梯度状态
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

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

print("=" * 60)
print("梯度诊断检查")
print("=" * 60)

# 初始化模型
model = Mymodel_HLA_ESM2(freeze_esm2=True).to(device)
criterion = nn.CrossEntropyLoss()

# 检查哪些参数是可训练的
print("\n[可训练参数列表]")
trainable_params = []
total_trainable = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)
        total_trainable += param.numel()
        print(f"  {name}: shape={param.shape}, numel={param.numel()}")

print(f"\n总可训练参数数量: {total_trainable}")

# 加载一个batch的数据
print("\n[加载测试数据]")
train_loader = data_load_HLA_ESM2(type_="train", fold=1, batch_size=batch_size)
pep_inputs, hla_inputs, labels, pep_masks, hla_masks = next(iter(train_loader))

pep_inputs = pep_inputs.to(device)
hla_inputs = hla_inputs.to(device)
labels = labels.to(device)
pep_masks = pep_masks.to(device)
hla_masks = hla_masks.to(device)

print(f"  batch_size: {labels.shape[0]}")
print(f"  labels distribution: 0={sum(labels.cpu().numpy()==0)}, 1={sum(labels.cpu().numpy()==1)}")

# Forward pass
print("\n[Forward Pass]")
model.train()
model.encoder_H.src_emb.esm2.eval()
model.encoder_P.src_emb.esm2.eval()

outputs, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss = criterion(outputs, labels)
print(f"  Initial loss: {loss.item():.4f}")

# Backward pass
print("\n[Backward Pass - 检查梯度]")
loss.backward()

# 检查每个可训练参数的梯度
print("\n[梯度统计]")
has_zero_grad = []
has_nan_grad = []
has_inf_grad = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            print(f"  {name}: grad=None (警告!)")
            has_zero_grad.append(name)
        elif torch.isnan(param.grad).any():
            print(f"  {name}: grad包含NaN (严重问题!)")
            has_nan_grad.append(name)
        elif torch.isinf(param.grad).any():
            print(f"  {name}: grad包含Inf (严重问题!)")
            has_inf_grad.append(name)
        elif param.grad.abs().max().item() == 0:
            print(f"  {name}: grad全为0")
            has_zero_grad.append(name)
        else:
            grad_max = param.grad.abs().max().item()
            grad_mean = param.grad.abs().mean().item()
            print(f"  {name}: grad_max={grad_max:.6f}, grad_mean={grad_mean:.6f}")

# 问题汇总
print("\n[问题汇总]")
if has_nan_grad:
    print(f"  发现NaN梯度的参数: {has_nan_grad}")
if has_inf_grad:
    print(f"  发现Inf梯度的参数: {has_inf_grad}")
if has_zero_grad:
    print(f"  发现零梯度的参数: {has_zero_grad}")
else:
    print("  所有可训练参数都有有效梯度")

# 检查ESM2Embedding的projection层是否在可训练列表中
print("\n[关键层检查]")
print("  encoder_H.src_emb.projection 是否可训练:",
      "encoder_H.src_emb.projection.weight" in trainable_params)
print("  encoder_P.src_emb.projection 是否可训练:",
      "encoder_P.src_emb.projection.weight" in trainable_params)
print("  projection.0 (最终分类层) 是否可训练:",
      "projection.0.weight" in trainable_params)

# 模拟两次forward（模拟FGM的情况）
print("\n[模拟第二个batch]")
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

# 第一个batch更新
optimizer.step()
optimizer.zero_grad()

# 再次forward
outputs2, _ = model(
    pep_inputs, hla_inputs,
    pep_attention_mask=pep_masks,
    hla_attention_mask=hla_masks
)
loss2 = criterion(outputs2, labels)
print(f"  Second forward loss: {loss2.item():.4f}")
loss2.backward()

# 检查第二次梯度
print("\n[第二次梯度统计]")
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_max = param.grad.abs().max().item()
        if grad_max == 0:
            print(f"  {name}: grad全为0 (问题!)")
        else:
            print(f"  {name}: grad_max={grad_max:.6f}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)