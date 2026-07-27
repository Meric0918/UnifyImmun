"""Online PLM encoders and progressive top-layer unfreezing for stage 2."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoders import (
    ResidueBatch,
    _load_trusted_state_dict,
    _normalise_sequence,
    _pad_residue_hidden,
    _torch_dtype,
    _validate_lengths,
    checkpoint_fingerprint,
)
from .model import BindingOutput, Task, UnifiedBindingModel

FineTuningStage = Literal["stage2a", "stage2b"]


class FineTunableResidueEncoder(nn.Module):
    """Interface shared by ESM-C and TCR-BERT fine-tuning encoders."""

    hidden_size: int

    def __init__(
        self,
        checkpoint_path: str | Path,
        max_length: int,
        *,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_fingerprint = checkpoint_fingerprint(self.checkpoint_path)
        self.max_length = max_length
        self.gradient_checkpointing = gradient_checkpointing
        self._active_training = False

    def transformer_blocks(self) -> nn.ModuleList:
        raise NotImplementedError

    def set_active_training(self, active: bool) -> None:
        """Use training mode only while this encoder participates in a task."""

        self._active_training = active
        self.model.train(active)

    def _gradient_context(self):
        if self._active_training:
            return nullcontext()
        return torch.no_grad()

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "encoder_class": type(self).__name__,
            "transformer_layers": len(self.transformer_blocks()),
            "gradient_checkpointing": self.gradient_checkpointing,
        }


class ESMCFineTuningEncoder(FineTunableResidueEncoder):
    """A separate ESM-C instance whose final Transformer blocks can be trained."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        max_length: int,
        device: torch.device,
        *,
        dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
    ):
        super().__init__(
            checkpoint_path,
            max_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        try:
            from safetensors.torch import load_file as load_safetensors
            from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers>=4.57.6 is required for stage-2 ESM-C fine-tuning"
            ) from exc

        self.device = device
        load_dtype = _torch_dtype(dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        model_config = AutoConfig.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        self.model = AutoModelForMaskedLM.from_config(model_config)
        weights_path = self.checkpoint_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"ESM-C weights not found: {weights_path}")
        state_dict = load_safetensors(weights_path, device="cpu")
        # TE serializes empty per-module tuning metadata as ``_extra_state``.
        # Transformers' meta-device loader treats the corresponding BytesIO
        # values as Tensors and crashes before loading.  These values contain
        # no learned parameters, so load the actual tensors directly and let
        # TE recreate its empty runtime metadata.
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.endswith("._extra_state")
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.endswith("._extra_state")]
        if missing or unexpected:
            raise RuntimeError(
                "ESM-C state_dict does not match its config: "
                f"missing={missing}, unexpected={unexpected}"
            )
        del state_dict
        self.model.to(device=device, dtype=load_dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        backbone = self._backbone()
        hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_size")
        self.hidden_size = int(hidden_size)
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        if not hasattr(backbone, "transformer") or not hasattr(
            backbone.transformer, "blocks"
        ):
            raise RuntimeError(
                "Unsupported ESM-C structure: expected backbone.transformer.blocks"
            )
        if gradient_checkpointing:
            _enable_esmc_activation_checkpointing(backbone.transformer)

    def _backbone(self) -> nn.Module:
        backbone = getattr(self.model, "esmc", None)
        if backbone is None:
            backbone = getattr(self.model, "base_model", None)
        if backbone is None:
            raise RuntimeError("Could not locate the ESM-C backbone")
        return backbone

    def transformer_blocks(self) -> nn.ModuleList:
        return self._backbone().transformer.blocks

    def forward(self, sequences: Sequence[str]) -> ResidueBatch:
        cleaned = [_normalise_sequence(sequence) for sequence in sequences]
        _validate_lengths(cleaned, self.max_length, "ESM-C")
        tokens = self.tokenizer(
            cleaned,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        with self._gradient_context():
            outputs = self._backbone()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                compute_sae=False,
            )
            hidden = outputs.last_hidden_state
            return _pad_residue_hidden(
                hidden=hidden,
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_token_ids=self.special_token_ids,
                expected_lengths=[len(sequence) for sequence in cleaned],
                max_length=self.max_length,
            )


class TCRBertFineTuningEncoder(FineTunableResidueEncoder):
    """TCR-BERT backbone with its legacy classifier removed."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        max_length: int,
        device: torch.device,
        *,
        dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
    ):
        super().__init__(
            checkpoint_path,
            max_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        try:
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for stage-2 TCR-BERT fine-tuning"
            ) from exc

        self.device = device
        load_dtype = _torch_dtype(dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
            use_fast=False,
        )
        model_config = AutoConfig.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        classifier_model = AutoModelForSequenceClassification.from_config(model_config)
        weights_path = self.checkpoint_path / "pytorch_model.bin"
        if not weights_path.is_file():
            raise FileNotFoundError(f"TCR-BERT weights not found: {weights_path}")
        state_dict = _load_trusted_state_dict(weights_path)
        missing, unexpected = classifier_model.load_state_dict(state_dict, strict=False)
        unexpected = [
            key for key in unexpected if key != "bert.embeddings.position_ids"
        ]
        if missing or unexpected:
            raise RuntimeError(
                "TCR-BERT state_dict does not match its config: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model = classifier_model.base_model.to(
            device=device,
            dtype=load_dtype,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        if not hasattr(self.model, "encoder") or not hasattr(
            self.model.encoder, "layer"
        ):
            raise RuntimeError(
                "Unsupported TCR-BERT structure: expected backbone.encoder.layer"
            )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def transformer_blocks(self) -> nn.ModuleList:
        return self.model.encoder.layer

    def forward(self, sequences: Sequence[str]) -> ResidueBatch:
        cleaned = [_normalise_sequence(sequence) for sequence in sequences]
        _validate_lengths(cleaned, self.max_length, "TCR-BERT")
        tokens = self.tokenizer(
            [" ".join(sequence) for sequence in cleaned],
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in tokens.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with self._gradient_context():
            outputs = self.model(**model_inputs, return_dict=True)
            return _pad_residue_hidden(
                hidden=outputs.last_hidden_state,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                special_token_ids=self.special_token_ids,
                expected_lengths=[len(sequence) for sequence in cleaned],
                max_length=self.max_length,
            )


class ProgressiveFineTuningModel(nn.Module):
    """Three independent PLMs followed by the unified stage-1 binding model."""

    def __init__(
        self,
        binding_model: UnifiedBindingModel,
        peptide_encoder: FineTunableResidueEncoder,
        hla_encoder: FineTunableResidueEncoder,
        tcr_encoder: FineTunableResidueEncoder,
        *,
        peptide_unfrozen_layers: int = 2,
        hla_unfrozen_layers: int = 1,
        tcr_unfrozen_layers: int = 2,
    ):
        super().__init__()
        self.binding_model = binding_model
        self.peptide_encoder = peptide_encoder
        self.hla_encoder = hla_encoder
        self.tcr_encoder = tcr_encoder
        self.peptide_unfrozen_layers = peptide_unfrozen_layers
        self.hla_unfrozen_layers = hla_unfrozen_layers
        self.tcr_unfrozen_layers = tcr_unfrozen_layers
        for encoder, count, name in (
            (peptide_encoder, peptide_unfrozen_layers, "Peptide ESM-C"),
            (hla_encoder, hla_unfrozen_layers, "HLA ESM-C"),
            (tcr_encoder, tcr_unfrozen_layers, "TCR-BERT"),
        ):
            if count > len(encoder.transformer_blocks()):
                raise ValueError(
                    f"{name} has {len(encoder.transformer_blocks())} blocks, "
                    f"cannot unfreeze {count}"
                )
        # Stage 2A keeps HLA ESM-C frozen while the pHLA dataset reuses a very
        # small set of HLA pseudo-sequences.  Keep those residue embeddings on
        # the training device so each distinct HLA is encoded only once.  This
        # is a plain Python cache (not a buffer), so it is deliberately absent
        # from checkpoints and is rebuilt lazily after resume.
        self._stage2a_hla_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._stage2a_hla_cache_requests = 0
        self._stage2a_hla_cache_misses = 0
        self._active_stage: FineTuningStage | None = None
        self.requires_grad_(False)
        self.eval()

    def set_trainable_task(self, task: Task, stage: FineTuningStage) -> None:
        """Freeze everything, then activate only this task and stage."""

        if stage not in {"stage2a", "stage2b"}:
            raise ValueError(f"Unknown fine-tuning stage: {stage}")
        self._active_stage = stage
        if stage == "stage2b" and self._stage2a_hla_cache:
            self.clear_stage2a_hla_cache()
        self.requires_grad_(False)
        self.eval()

        active_modules = self.binding_model.task_modules(task)
        for module in active_modules.values():
            module.requires_grad_(True)
            module.train(True)

        self._activate_encoder(
            self.peptide_encoder,
            self.peptide_unfrozen_layers,
        )
        if stage == "stage2b" and task == "phla":
            self._activate_encoder(
                self.hla_encoder,
                self.hla_unfrozen_layers,
            )
        elif stage == "stage2b" and task == "ptcr":
            self._activate_encoder(
                self.tcr_encoder,
                self.tcr_unfrozen_layers,
            )
        elif task not in {"phla", "ptcr"}:
            raise ValueError(f"Unknown task: {task}")

    def set_evaluation(self) -> None:
        self.eval()
        for encoder in (
            self.peptide_encoder,
            self.hla_encoder,
            self.tcr_encoder,
        ):
            encoder.set_active_training(False)

    @staticmethod
    def _activate_encoder(
        encoder: FineTunableResidueEncoder,
        layer_count: int,
    ) -> None:
        for block in encoder.transformer_blocks()[-layer_count:]:
            block.requires_grad_(True)
        encoder.set_active_training(True)

    def forward(
        self,
        task: Task,
        peptide_sequences: Sequence[str],
        receptor_sequences: Sequence[str],
    ) -> BindingOutput:
        peptide = self.peptide_encoder(peptide_sequences)
        if task == "phla":
            if self._active_stage == "stage2a":
                receptor = self._encode_cached_stage2a_hla(receptor_sequences)
            else:
                receptor = self.hla_encoder(receptor_sequences)
        elif task == "ptcr":
            receptor = self.tcr_encoder(receptor_sequences)
        else:
            raise ValueError(f"Unknown task: {task}")
        return self.binding_model(
            task=task,
            peptide_hidden=peptide.hidden,
            peptide_mask=peptide.mask,
            receptor_hidden=receptor.hidden,
            receptor_mask=receptor.mask,
        )

    def _encode_cached_stage2a_hla(
        self,
        sequences: Sequence[str],
    ) -> ResidueBatch:
        """Encode each frozen Stage-2A HLA pseudo-sequence only once."""

        cleaned = [_normalise_sequence(sequence) for sequence in sequences]
        missing = list(
            dict.fromkeys(
                sequence
                for sequence in cleaned
                if sequence not in self._stage2a_hla_cache
            )
        )
        self._stage2a_hla_cache_requests += len(cleaned)
        self._stage2a_hla_cache_misses += len(missing)
        if missing:
            encoded = self.hla_encoder(missing)
            for index, sequence in enumerate(missing):
                self._stage2a_hla_cache[sequence] = (
                    encoded.hidden[index].detach(),
                    encoded.mask[index].detach(),
                    encoded.lengths[index].detach(),
                )

        rows = [self._stage2a_hla_cache[sequence] for sequence in cleaned]
        return ResidueBatch(
            hidden=torch.stack([row[0] for row in rows]),
            mask=torch.stack([row[1] for row in rows]),
            lengths=torch.stack([row[2] for row in rows]),
        )

    def clear_stage2a_hla_cache(self) -> None:
        """Release frozen HLA embeddings before HLA becomes trainable."""

        self._stage2a_hla_cache.clear()

    def stage2a_hla_cache_stats(self) -> dict[str, int]:
        requests = self._stage2a_hla_cache_requests
        misses = self._stage2a_hla_cache_misses
        return {
            "entries": len(self._stage2a_hla_cache),
            "requests": requests,
            "hits": requests - misses,
            "misses": misses,
        }

    def encoder_metadata(self) -> dict[str, Any]:
        return {
            "peptide": self.peptide_encoder.metadata(),
            "hla": self.hla_encoder.metadata(),
            "tcr": self.tcr_encoder.metadata(),
        }


def _enable_esmc_activation_checkpointing(transformer_stack: nn.Module) -> None:
    """Checkpoint trainable ESM-C blocks without changing state-dict keys."""

    if getattr(transformer_stack, "_stage2_activation_checkpointing", False):
        return
    original_forward = transformer_stack.forward

    def checkpointed_forward(
        stack,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        layers_to_collect: list[int] | None = None,
        output_attentions: bool = False,
    ):
        if output_attentions or layers_to_collect:
            return original_forward(
                x,
                sequence_id=sequence_id,
                layers_to_collect=layers_to_collect,
                output_attentions=output_attentions,
            )

        for block in stack.blocks:
            trainable = any(parameter.requires_grad for parameter in block.parameters())
            if torch.is_grad_enabled() and block.training and trainable:
                def run_block(hidden, current_block=block):
                    return current_block(
                        hidden,
                        sequence_id,
                        output_attentions=False,
                    )[0]

                x = checkpoint(run_block, x, use_reentrant=False)
            else:
                x, _ = block(
                    x,
                    sequence_id,
                    output_attentions=False,
                )
        norm_x = stack.norm(x)
        return norm_x, x, (), None

    transformer_stack.forward = MethodType(
        checkpointed_forward,
        transformer_stack,
    )
    transformer_stack._stage2_activation_checkpointing = True
