"""Module-targeted Fast Gradient Method adversarial training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


@dataclass
class FGMStats:
    attacked_parameter_count: int
    mean_gradient_norm: float
    max_gradient_norm: float


class ModuleFGM:
    """Apply old-style per-parameter FGM to explicitly selected modules."""

    def __init__(self, root_model: nn.Module):
        self.root_model = root_model
        self._backup: dict[str, tuple[nn.Parameter, torch.Tensor]] = {}
        self._batch_norm_state: dict[str, bool] = {}

    @torch.no_grad()
    def attack(
        self,
        modules: Mapping[str, nn.Module],
        *,
        epsilon: float = 1.0,
    ) -> FGMStats:
        if self._backup:
            raise RuntimeError("FGM attack called twice without restore")
        if epsilon <= 0:
            raise ValueError("FGM epsilon must be positive")

        for name, module in self.root_model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self._batch_norm_state[name] = module.track_running_stats
                module.track_running_stats = False

        gradient_norms: list[float] = []
        seen_parameters: set[int] = set()
        for module_name, module in modules.items():
            for parameter_name, parameter in module.named_parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                norm = torch.linalg.vector_norm(parameter.grad)
                if not torch.isfinite(norm) or norm.item() == 0:
                    continue
                qualified_name = f"{module_name}.{parameter_name}"
                self._backup[qualified_name] = (
                    parameter,
                    parameter.detach().clone(),
                )
                parameter.add_(epsilon * parameter.grad / norm)
                gradient_norms.append(float(norm.detach().cpu()))

        if not self._backup:
            self._restore_batch_norm()
            raise RuntimeError(
                "FGM did not find any trainable target parameter with a finite gradient"
            )
        return FGMStats(
            attacked_parameter_count=len(self._backup),
            mean_gradient_norm=sum(gradient_norms) / len(gradient_norms),
            max_gradient_norm=max(gradient_norms),
        )

    @torch.no_grad()
    def restore(self) -> None:
        for parameter, backup in self._backup.values():
            parameter.copy_(backup)
        self._backup.clear()
        self._restore_batch_norm()

    def _restore_batch_norm(self) -> None:
        modules = dict(self.root_model.named_modules())
        for name, track_running_stats in self._batch_norm_state.items():
            module = modules.get(name)
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = track_running_stats
        self._batch_norm_state.clear()
