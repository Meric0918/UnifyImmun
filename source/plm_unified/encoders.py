"""Frozen residue-level encoders used to build the stage-1 cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class ResidueBatch:
    hidden: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor


def checkpoint_fingerprint(path: str | Path) -> str:
    """Create a stable local-checkpoint fingerprint without hashing multi-GB weights."""

    checkpoint = Path(path).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    digest = hashlib.sha256()
    for item in sorted(checkpoint.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(checkpoint).as_posix()
        stat = item.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if item.suffix in {".json", ".txt", ".md"} and stat.st_size <= 2_000_000:
            digest.update(item.read_bytes())
        elif item.suffix in {".bin", ".safetensors"}:
            sample_size = min(1_048_576, stat.st_size)
            with item.open("rb") as handle:
                digest.update(handle.read(sample_size))
                if stat.st_size > sample_size:
                    handle.seek(-sample_size, 2)
                    digest.update(handle.read(sample_size))
    return digest.hexdigest()


def _torch_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported encoder dtype: {name}")


def _pad_residue_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: set[int],
    expected_lengths: Sequence[int],
    max_length: int,
) -> ResidueBatch:
    """Remove BOS/EOS/PAD/CLS/SEP tokens and pad only real residues."""

    if hidden.ndim != 3:
        raise ValueError(f"Expected [B, L, D] hidden states, got {tuple(hidden.shape)}")
    batch_size, _, hidden_dim = hidden.shape
    result = hidden.new_zeros((batch_size, max_length, hidden_dim))
    residue_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool, device=hidden.device
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=hidden.device)

    for row in range(batch_size):
        active = attention_mask[row].bool()
        ids = input_ids[row]
        keep = active.clone()
        for token_id in special_token_ids:
            keep &= ids.ne(token_id)
        row_hidden = hidden[row][keep]
        actual_length = row_hidden.shape[0]
        expected_length = expected_lengths[row]
        if actual_length != expected_length:
            kept_ids = ids[keep].detach().cpu().tolist()
            raise ValueError(
                "Tokenizer-to-residue alignment failed: "
                f"expected {expected_length} residues, found {actual_length}; "
                f"kept token ids={kept_ids}"
            )
        if actual_length > max_length:
            raise ValueError(
                f"Sequence length {actual_length} exceeds configured max {max_length}"
            )
        result[row, :actual_length] = row_hidden
        residue_mask[row, :actual_length] = True
        lengths[row] = actual_length

    return ResidueBatch(hidden=result, mask=residue_mask, lengths=lengths)


class FrozenResidueEncoder(nn.Module):
    """Base interface for frozen PLMs that return residue-level hidden states."""

    hidden_size: int
    checkpoint_fingerprint: str

    def __init__(self, checkpoint_path: str | Path, max_length: int):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.max_length = max_length
        self.checkpoint_fingerprint = checkpoint_fingerprint(self.checkpoint_path)

    def encode(self, sequences: Sequence[str]) -> ResidueBatch:
        raise NotImplementedError

    def train(self, mode: bool = True) -> "FrozenResidueEncoder":
        super().train(False)
        return self


class ESMCResidueEncoder(FrozenResidueEncoder):
    """ESM-C masked-LM checkpoint exposed as a frozen residue encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        max_length: int,
        device: torch.device,
        dtype: str = "float16",
    ):
        super().__init__(checkpoint_path, max_length)
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers>=4.57.6 is required for the ESM-C checkpoint"
            ) from exc

        self.device = device
        load_dtype = _torch_dtype(dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
            dtype=load_dtype,
        ).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_size")
        self.hidden_size = int(hidden_size)
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> ResidueBatch:
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                "The installed Transformers ESM-C implementation did not return "
                "last_hidden_state. Use transformers>=4.57.6."
            )
        return _pad_residue_hidden(
            hidden=hidden,
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=self.special_token_ids,
            expected_lengths=[len(sequence) for sequence in cleaned],
            max_length=self.max_length,
        )


class TCRBertResidueEncoder(FrozenResidueEncoder):
    """TCR-BERT classifier checkpoint with its legacy head removed."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        max_length: int,
        device: torch.device,
        dtype: str = "float16",
    ):
        super().__init__(checkpoint_path, max_length)
        try:
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError("transformers is required for TCR-BERT") from exc

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
        if missing or unexpected:
            raise RuntimeError(
                "TCR-BERT state_dict does not match its config: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.model = classifier_model.base_model.to(device=device, dtype=load_dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> ResidueBatch:
        cleaned = [_normalise_sequence(sequence) for sequence in sequences]
        _validate_lengths(cleaned, self.max_length, "TCR-BERT")
        spaced = [" ".join(sequence) for sequence in cleaned]
        tokens = self.tokenizer(
            spaced,
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
        outputs = self.model(**model_inputs, return_dict=True)
        return _pad_residue_hidden(
            hidden=outputs.last_hidden_state,
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            special_token_ids=self.special_token_ids,
            expected_lengths=[len(sequence) for sequence in cleaned],
            max_length=self.max_length,
        )


class OnlineEncoderBundle:
    """Frozen online encoders for explicitly limited debug training."""

    def __init__(
        self,
        *,
        esmc_path: str | Path,
        tcr_bert_path: str | Path,
        peptide_max_length: int,
        hla_max_length: int,
        tcr_max_length: int,
        device: torch.device,
        dtype: str,
    ):
        self.peptide_max_length = peptide_max_length
        self.hla_max_length = hla_max_length
        self.esmc = ESMCResidueEncoder(
            esmc_path,
            max_length=hla_max_length,
            device=device,
            dtype=dtype,
        )
        self.tcr_bert = TCRBertResidueEncoder(
            tcr_bert_path,
            max_length=tcr_max_length,
            device=device,
            dtype=dtype,
        )

    def encode(
        self,
        task: str,
        peptide_sequences: Sequence[str],
        receptor_sequences: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        self.esmc.max_length = self.peptide_max_length
        peptide = self.esmc.encode(peptide_sequences)
        if task == "phla":
            self.esmc.max_length = self.hla_max_length
            receptor = self.esmc.encode(receptor_sequences)
        elif task == "ptcr":
            receptor = self.tcr_bert.encode(receptor_sequences)
        else:
            raise ValueError(f"Unknown task: {task}")
        return {
            "peptide_hidden": peptide.hidden,
            "peptide_mask": peptide.mask,
            "receptor_hidden": receptor.hidden,
            "receptor_mask": receptor.mask,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "online_debug",
            "esmc": encoder_metadata(self.esmc),
            "tcr_bert": encoder_metadata(self.tcr_bert),
        }


def _normalise_sequence(sequence: str) -> str:
    normalised = "".join(str(sequence).split()).replace("-", "").upper()
    if not normalised:
        raise ValueError("Encountered an empty sequence")
    return normalised


def _load_trusted_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load the user-provided local TCR-BERT checkpoint across PyTorch versions."""

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state_dict mapping in {path}")
    return state


def _validate_lengths(
    sequences: Iterable[str], max_length: int, encoder_name: str
) -> None:
    too_long = [sequence for sequence in sequences if len(sequence) > max_length]
    if too_long:
        preview = ", ".join(f"{seq[:12]}…({len(seq)})" for seq in too_long[:3])
        raise ValueError(
            f"{encoder_name} received sequences longer than {max_length}: {preview}"
        )


def encoder_metadata(encoder: FrozenResidueEncoder) -> dict[str, Any]:
    return {
        "checkpoint_path": str(encoder.checkpoint_path),
        "checkpoint_fingerprint": encoder.checkpoint_fingerprint,
        "hidden_size": encoder.hidden_size,
        "max_length": encoder.max_length,
        "encoder_class": type(encoder).__name__,
    }


def write_encoder_metadata(path: str | Path, encoder: FrozenResidueEncoder) -> None:
    target = Path(path)
    target.write_text(
        json.dumps(encoder_metadata(encoder), indent=2, sort_keys=True),
        encoding="utf-8",
    )
