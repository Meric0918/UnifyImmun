"""
修复方案1: FGM类也备份和恢复BatchNorm的running stats
"""
class FGM_ESM2_Fixed:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_backup = {}  # 新增：备份BatchNorm running stats

    def attack(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

        # 新增：备份所有BatchNorm层的running stats
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

        # 新增：恢复BatchNorm running stats
        for name, module in self.model.named_modules():
            if name in self.bn_backup:
                module.running_mean = self.bn_backup[name]['running_mean']
                module.running_var = self.bn_backup[name]['running_var']
                module.num_batches_tracked = self.bn_backup[name]['num_batches_tracked']
        self.bn_backup = {}