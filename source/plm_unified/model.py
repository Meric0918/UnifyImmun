"""One unified model with a shared peptide Adapter and two task heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .attention import (
    BindingClassifier,
    CrossAttentionBlock,
    MaskedAttentionPooling,
    SequenceAdapter,
)
from .config import ModelConfig

Task = Literal["phla", "ptcr"]

TASK_MODULE_NAMES: dict[Task, tuple[str, ...]] = {
    "phla": (
        "peptide_adapter",
        "hla_adapter",
        "phla_cross_attention",
        "phla_pooling",
        "phla_classifier",
    ),
    "ptcr": (
        "peptide_adapter",
        "tcr_adapter",
        "ptcr_cross_attention",
        "ptcr_pooling",
        "ptcr_classifier",
    ),
}


@dataclass
class BindingOutput:
    logits: torch.Tensor
    cross_attention: torch.Tensor | None = None
    pooling_weights: torch.Tensor | None = None


class UnifiedBindingModel(nn.Module):
    """Adapter-level joint model trained from cached frozen-PLM features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        adapter_kwargs = {
            "model_dim": config.model_dim,
            "hidden_dim": config.adapter_hidden_dim,
            "dropout": config.adapter_dropout,
        }
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        self.peptide_adapter = SequenceAdapter(
            config.peptide_input_dim, **adapter_kwargs
        )
        self.hla_adapter = SequenceAdapter(config.hla_input_dim, **adapter_kwargs)
        self.tcr_adapter = SequenceAdapter(config.tcr_input_dim, **adapter_kwargs)

        self.phla_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.ptcr_cross_attention = CrossAttentionBlock(**attention_kwargs)
        self.phla_pooling = MaskedAttentionPooling(config.model_dim)
        self.ptcr_pooling = MaskedAttentionPooling(config.model_dim)
        self.phla_classifier = BindingClassifier(
            config.model_dim, config.classifier_dropout
        )
        self.ptcr_classifier = BindingClassifier(
            config.model_dim, config.classifier_dropout
        )

    def forward(
        self,
        task: Task,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_hidden: torch.Tensor,
        receptor_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> BindingOutput:
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        if task == "phla":
            receptor = self.hla_adapter(receptor_hidden, receptor_mask)
            cross_attention = self.phla_cross_attention
            pooling = self.phla_pooling
            classifier = self.phla_classifier
        elif task == "ptcr":
            receptor = self.tcr_adapter(receptor_hidden, receptor_mask)
            cross_attention = self.ptcr_cross_attention
            pooling = self.ptcr_pooling
            classifier = self.ptcr_classifier
        else:
            raise ValueError(f"Unknown task: {task}")

        sequence, attention_weights = cross_attention(
            peptide,
            receptor,
            peptide_mask,
            receptor_mask,
            return_attention=return_attention,
        )
        pooled, pooling_weights = pooling(
            sequence,
            peptide_mask,
            return_weights=return_attention,
        )
        return BindingOutput(
            logits=classifier(pooled),
            cross_attention=attention_weights,
            pooling_weights=pooling_weights,
        )

    def task_modules(self, task: Task) -> dict[str, nn.Module]:
        try:
            names = TASK_MODULE_NAMES[task]
        except KeyError as exc:
            raise ValueError(f"Unknown task: {task}") from exc
        return {name: getattr(self, name) for name in names}

    def fgm_modules(self, task: Task) -> dict[str, nn.Module]:
        if task == "phla":
            return {
                "peptide_adapter": self.peptide_adapter,
                "hla_adapter": self.hla_adapter,
            }
        if task == "ptcr":
            return {
                "peptide_adapter": self.peptide_adapter,
                "tcr_adapter": self.tcr_adapter,
            }
        raise ValueError(f"Unknown task: {task}")

    def set_trainable_task(self, task: Task) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        active_modules = self.task_modules(task)
        for module in active_modules.values():
            for parameter in module.parameters():
                parameter.requires_grad = True

    def trainable_parameter_names(self) -> list[str]:
        return [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]


class TaskSpecificBindingModel(nn.Module):
    """One deployable pHLA or pTCR branch exported from the unified model."""

    def __init__(self, config: ModelConfig, task: Task):
        super().__init__()
        if task not in TASK_MODULE_NAMES:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        adapter_kwargs = {
            "model_dim": config.model_dim,
            "hidden_dim": config.adapter_hidden_dim,
            "dropout": config.adapter_dropout,
        }
        attention_kwargs = {
            "model_dim": config.model_dim,
            "num_heads": config.num_attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.attention_dropout,
        }
        self.peptide_adapter = SequenceAdapter(
            config.peptide_input_dim, **adapter_kwargs
        )
        if task == "phla":
            self.hla_adapter = SequenceAdapter(
                config.hla_input_dim, **adapter_kwargs
            )
            self.phla_cross_attention = CrossAttentionBlock(**attention_kwargs)
            self.phla_pooling = MaskedAttentionPooling(config.model_dim)
            self.phla_classifier = BindingClassifier(
                config.model_dim, config.classifier_dropout
            )
        else:
            self.tcr_adapter = SequenceAdapter(
                config.tcr_input_dim, **adapter_kwargs
            )
            self.ptcr_cross_attention = CrossAttentionBlock(**attention_kwargs)
            self.ptcr_pooling = MaskedAttentionPooling(config.model_dim)
            self.ptcr_classifier = BindingClassifier(
                config.model_dim, config.classifier_dropout
            )

    def forward(
        self,
        task: Task,
        peptide_hidden: torch.Tensor,
        peptide_mask: torch.Tensor,
        receptor_hidden: torch.Tensor,
        receptor_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> BindingOutput:
        if task != self.task:
            raise ValueError(
                f"This checkpoint is for task '{self.task}', not '{task}'"
            )
        peptide = self.peptide_adapter(peptide_hidden, peptide_mask)
        if task == "phla":
            receptor = self.hla_adapter(receptor_hidden, receptor_mask)
            cross_attention = self.phla_cross_attention
            pooling = self.phla_pooling
            classifier = self.phla_classifier
        else:
            receptor = self.tcr_adapter(receptor_hidden, receptor_mask)
            cross_attention = self.ptcr_cross_attention
            pooling = self.ptcr_pooling
            classifier = self.ptcr_classifier

        sequence, attention_weights = cross_attention(
            peptide,
            receptor,
            peptide_mask,
            receptor_mask,
            return_attention=return_attention,
        )
        pooled, pooling_weights = pooling(
            sequence,
            peptide_mask,
            return_weights=return_attention,
        )
        return BindingOutput(
            logits=classifier(pooled),
            cross_attention=attention_weights,
            pooling_weights=pooling_weights,
        )


def extract_task_state_dict(
    state_dict: dict[str, torch.Tensor],
    task: Task,
) -> dict[str, torch.Tensor]:
    """Select the shared peptide path and one task branch from unified weights."""

    try:
        prefixes = tuple(f"{name}." for name in TASK_MODULE_NAMES[task])
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task}") from exc
    selected = {
        name: value.detach().cpu()
        for name, value in state_dict.items()
        if name.startswith(prefixes)
    }
    if not selected:
        raise ValueError(f"No parameters found for task '{task}'")
    return selected
