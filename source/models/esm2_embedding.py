"""
ESM2 Embedding Module for Peptide, HLA, and TCR sequences.

This module provides:
- ESM2TokenizerWrapper: Wrapper for ESM2 tokenizer
- ESM2Embedding: ESM2-based embedding layer with projection
- Modified Encoder classes for ESM2 integration
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
from typing import Optional, List

# ESM2 model path
ESM2_MODEL_PATH = "/home/mjp/model/esm2_t33_650M_UR50D"

# Model parameters
pep_max_len = 15
hla_max_len = 34
tcr_max_len = 34

# ESM2 max length (excluding cls/eos tokens)
# For 34 output positions, we use 32 amino acids + cls + eos = 34
esm2_max_len = 32

d_model = 64
d_ff = 512
d_k = d_v = 64
n_heads = 1
n_layers = 1
esm2_hidden_size = 1280

batch_size = 64  # Reduced for ESM2 memory consumption
epochs = 30
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class ESM2TokenizerWrapper:
    """
    Wrapper for ESM2 tokenizer to convert amino acid sequences to ESM2 tokens.
    ESM2 adds <cls> at position 0 and <eos> at the end.
    """

    def __init__(self, model_path: str = ESM2_MODEL_PATH):
        from transformers import EsmTokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_path)
        self.pad_token_id = 1  # ESM2's <pad> token
        self.cls_token_id = 0  # ESM2's <cls> token
        self.eos_token_id = 2  # ESM2's <eos> token

    def encode_sequences(
        self, sequences: List[str], max_length: int = esm2_max_len
    ) -> dict:
        """
        Convert amino acid sequences to ESM2 token IDs.

        Args:
            sequences: List of amino acid strings
            max_length: Maximum sequence length for amino acids
                       (ESM2 will output max_length + 2 for cls and eos)

        Returns:
            dict with 'input_ids' and 'attention_mask' tensors
        """
        # ESM2 tokenizer adds <cls> at start and <eos> at end
        # Replace '-' padding with proper handling
        processed_sequences = []
        for seq in sequences:
            # Remove '-' padding characters (ESM2 will handle padding)
            seq_clean = seq.replace("-", "")
            processed_sequences.append(seq_clean)

        encoded = self.tokenizer(
            processed_sequences,
            padding="max_length",
            truncation=True,
            max_length=max_length + 2,  # +2 for <cls> and <eos>
            return_tensors="pt",
        )
        return encoded

    def encode_single_sequence(self, sequence: str, max_length: int = esm2_max_len):
        """Convert a single amino acid sequence to ESM2 token IDs."""
        seq_clean = sequence.replace("-", "")
        encoded = self.tokenizer(
            seq_clean,
            padding="max_length",
            truncation=True,
            max_length=max_length + 2,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)


class ESM2Embedding(nn.Module):
    """
    ESM2-based embedding layer that replaces nn.Embedding.
    Outputs 64-dimensional embeddings (projected from ESM2's 1280 dimensions).
    ESM2 parameters are frozen, only projection layer is trainable.
    """

    def __init__(
        self,
        model_path: str = ESM2_MODEL_PATH,
        output_dim: int = d_model,
        freeze: bool = True,
        max_seq_length: int = hla_max_len,
    ):
        super(ESM2Embedding, self).__init__()
        from transformers import EsmModel

        self.esm2 = EsmModel.from_pretrained(model_path)
        self.esm2_hidden_size = esm2_hidden_size
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        self.freeze = freeze

        # Freeze ESM2 parameters
        if freeze:
            for param in self.esm2.parameters():
                param.requires_grad = False
            self.esm2.eval()  # Set to eval mode for frozen model

        # Projection layer: 1280 -> 64
        self.projection = nn.Linear(self.esm2_hidden_size, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through ESM2 and projection.

        Args:
            input_ids: [batch_size, seq_len] ESM2 token IDs
            attention_mask: [batch_size, seq_len] mask for padding

        Returns:
            [batch_size, seq_len, output_dim] embeddings
        """
        # ESM2 forward pass (frozen, so no gradients needed)
        if self.freeze:
            with torch.no_grad():
                outputs = self.esm2(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                hidden_states = outputs.last_hidden_state  # [batch, seq_len, 1280]
        else:
            outputs = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

        # Project to output dimension
        projected = self.projection(hidden_states)  # [batch, seq_len, 64]
        normalized = self.layer_norm(projected)

        return normalized


def get_attn_pad_mask_esm2(seq_q, seq_k):
    """
    Padding mask for ESM2 tokens.
    ESM2 uses <pad> token with id=1.
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # ESM2 <pad> token id is 1
    pad_attn_mask = seq_k.data.eq(1).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = nn.Dropout(0.1)(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        enc_outputs1 = enc_inputs + self.dropout(enc_outputs)
        enc_outputs1 = self.layer_norm(enc_outputs1)
        enc_outputs = self.pos_ffn(enc_outputs1)
        return enc_outputs, attn


class Encoder_ESM2(nn.Module):
    """
    Encoder using ESM2 embeddings.
    For HLA/TCR sequences (padded to 34 length).
    Note: No PositionalEncoding added since ESM2 has rotary position embeddings.
    """

    def __init__(
        self,
        model_path: str = ESM2_MODEL_PATH,
        output_dim: int = d_model,
        freeze_esm2: bool = True,
        n_layers: int = n_layers,
    ):
        super(Encoder_ESM2, self).__init__()

        # ESM2 embedding (replaces nn.Embedding + PositionalEncoding)
        self.src_emb = ESM2Embedding(
            model_path=model_path, output_dim=output_dim, freeze=freeze_esm2
        )

        # Transformer layers (optional, can be removed if ESM2 output is sufficient)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, attention_mask=None):
        """
        Args:
            enc_inputs: [batch_size, seq_len] ESM2 token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            enc_outputs: [batch_size, seq_len, d_model]
            enc_self_attns: list of attention weights
        """
        # Get ESM2 embeddings (already includes position info)
        enc_outputs = self.src_emb(enc_inputs, attention_mask)

        # Create padding mask for transformer layers
        enc_self_attn_mask = get_attn_pad_mask_esm2(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Encoder_padding_ESM2(nn.Module):
    """
    Encoder using ESM2 embeddings for peptide sequences.
    Handles shorter peptide sequences padded to match HLA/TCR length (34).
    """

    def __init__(
        self,
        model_path: str = ESM2_MODEL_PATH,
        output_dim: int = d_model,
        freeze_esm2: bool = True,
        n_layers: int = n_layers,
        pep_max_len: int = pep_max_len,
        target_max_len: int = hla_max_len,
    ):
        super(Encoder_padding_ESM2, self).__init__()

        self.pep_max_len = pep_max_len
        self.target_max_len = target_max_len

        # ESM2 embedding
        self.src_emb = ESM2Embedding(
            model_path=model_path, output_dim=output_dim, freeze=freeze_esm2
        )

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, attention_mask=None):
        """
        Args:
            enc_inputs: [batch_size, pep_seq_len] ESM2 token IDs for peptides
            attention_mask: [batch_size, pep_seq_len] attention mask

        Returns:
            enc_outputs: [batch_size, 34, d_model] - padded to target length
            enc_self_attns: list of attention weights
        """
        # Get ESM2 embeddings
        enc_outputs = self.src_emb(enc_inputs, attention_mask)

        # Pad to target_max_len (34)
        batch_size = enc_outputs.shape[0]
        actual_seq_len = enc_outputs.shape[1]
        enc_pad = torch.zeros(batch_size, self.target_max_len, d_model).to(device)
        enc_pad[:, :actual_seq_len, :] = enc_outputs
        enc_outputs = enc_pad

        # Create attention mask for padded sequence
        # Mask padding positions (beyond actual sequence length)
        enc_self_attn_mask = torch.zeros(
            batch_size, self.target_max_len, self.target_max_len, dtype=torch.bool
        ).to(device)
        # Mark positions beyond actual_seq_len as masked (True)
        enc_self_attn_mask[:, actual_seq_len:, :] = True
        enc_self_attn_mask[:, :, actual_seq_len:] = True

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(0.1)

    def forward(self, pep_inputs, hla_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(
            pep_inputs, hla_inputs, hla_inputs, dec_self_attn_mask
        )
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class Cross_Attention(nn.Module):
    def __init__(self):
        super(Cross_Attention, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = hla_max_len

    def forward(self, pep_inputs, hla_inputs):
        pep_outputs = pep_inputs.to(device)
        hla_outputs = hla_inputs.to(device)
        dec_self_attn_pad_mask = (
            torch.LongTensor(
                np.zeros((pep_inputs.shape[0], hla_max_len, hla_max_len))
            )
            .bool()
            .to(device)
        )
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                pep_outputs, hla_outputs, dec_self_attn_pad_mask
            )
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


class Mymodel_HLA_ESM2(nn.Module):
    """
    HLA binding model using ESM2 embeddings.
    """

    def __init__(
        self,
        model_path: str = ESM2_MODEL_PATH,
        freeze_esm2: bool = True,
    ):
        super(Mymodel_HLA_ESM2, self).__init__()

        self.encoder_H = Encoder_ESM2(model_path, freeze_esm2=freeze_esm2).to(device)
        self.encoder_P = Encoder_padding_ESM2(
            model_path, freeze_esm2=freeze_esm2
        ).to(device)
        self.cross_1 = Cross_Attention().to(device)

        self.projection = nn.Sequential(
            nn.Linear(hla_max_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
        ).to(device)

    def forward(
        self, pep_inputs, hla_inputs, pep_attention_mask=None, hla_attention_mask=None
    ):
        hla_enc, hla_attn = self.encoder_H(hla_inputs, attention_mask=hla_attention_mask)
        pep_enc, enc1_attn = self.encoder_P(pep_inputs, attention_mask=pep_attention_mask)
        pep_hla, pep_hla_attn = self.cross_1(pep_enc, hla_enc)
        pep_hla_outputs = pep_hla.view(pep_hla.shape[0], -1)
        pep_hla_logits = self.projection(pep_hla_outputs)
        return pep_hla_logits.view(-1, pep_hla_logits.size(-1)), pep_hla_attn


class Mymodel_TCR_ESM2(nn.Module):
    """
    TCR binding model using ESM2 embeddings.
    """

    def __init__(
        self,
        model_path: str = ESM2_MODEL_PATH,
        freeze_esm2: bool = True,
    ):
        super(Mymodel_TCR_ESM2, self).__init__()

        self.encoder_T = Encoder_ESM2(model_path, freeze_esm2=freeze_esm2).to(device)
        self.encoder_P = Encoder_padding_ESM2(
            model_path, freeze_esm2=freeze_esm2
        ).to(device)
        self.cross_2 = Cross_Attention().to(device)

        self.projection = nn.Sequential(
            nn.Linear(tcr_max_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
        ).to(device)

    def forward(
        self, pep_inputs, tcr_inputs, pep_attention_mask=None, tcr_attention_mask=None
    ):
        tcr_enc, tcr_attn = self.encoder_T(tcr_inputs, attention_mask=tcr_attention_mask)
        pep_enc, enc1_attn = self.encoder_P(pep_inputs, attention_mask=pep_attention_mask)
        pep_tcr, pep_tcr_attn = self.cross_2(pep_enc, tcr_enc)
        pep_tcr_outputs = pep_tcr.view(pep_tcr.shape[0], -1)
        pep_tcr_logits = self.projection(pep_tcr_outputs)
        return pep_tcr_logits.view(-1, pep_tcr_logits.size(-1)), pep_tcr_attn


# Dataset classes for ESM2
class MyDataSet_HLA_ESM2(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels, pep_masks, hla_masks):
        super(MyDataSet_HLA_ESM2, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels
        self.pep_masks = pep_masks
        self.hla_masks = hla_masks

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return (
            self.pep_inputs[idx],
            self.hla_inputs[idx],
            self.labels[idx],
            self.pep_masks[idx],
            self.hla_masks[idx],
        )


class MyDataSet_TCR_ESM2(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks):
        super(MyDataSet_TCR_ESM2, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels
        self.pep_masks = pep_masks
        self.tcr_masks = tcr_masks

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return (
            self.pep_inputs[idx],
            self.tcr_inputs[idx],
            self.labels[idx],
            self.pep_masks[idx],
            self.tcr_masks[idx],
        )


def data_process_HLA_ESM2(data, tokenizer_wrapper: ESM2TokenizerWrapper):
    """
    Process HLA data using ESM2 tokenizer.
    """
    pep_sequences = []
    hla_sequences = []
    labels = []

    for pep, hla, label in zip(data.peptide, data.HLA, data.label):
        # Truncate to appropriate lengths
        pep_truncated = pep[:pep_max_len] if len(pep) > pep_max_len else pep
        hla_truncated = hla[:esm2_max_len] if len(hla) > esm2_max_len else hla
        pep_sequences.append(pep_truncated)
        hla_sequences.append(hla_truncated)
        labels.append(label)

    # Encode with ESM2 tokenizer
    pep_encoded = tokenizer_wrapper.encode_sequences(pep_sequences, max_length=pep_max_len)
    hla_encoded = tokenizer_wrapper.encode_sequences(hla_sequences, max_length=esm2_max_len)

    return (
        pep_encoded["input_ids"],
        hla_encoded["input_ids"],
        torch.LongTensor(labels),
        pep_encoded["attention_mask"],
        hla_encoded["attention_mask"],
    )


def data_process_TCR_ESM2(data, tokenizer_wrapper: ESM2TokenizerWrapper):
    """
    Process TCR data using ESM2 tokenizer.
    """
    pep_sequences = []
    tcr_sequences = []
    labels = []

    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep_truncated = pep[:pep_max_len] if len(pep) > pep_max_len else pep
        tcr_truncated = tcr[:esm2_max_len] if len(tcr) > esm2_max_len else tcr
        pep_sequences.append(pep_truncated)
        tcr_sequences.append(tcr_truncated)
        labels.append(label)

    pep_encoded = tokenizer_wrapper.encode_sequences(pep_sequences, max_length=pep_max_len)
    tcr_encoded = tokenizer_wrapper.encode_sequences(tcr_sequences, max_length=esm2_max_len)

    return (
        pep_encoded["input_ids"],
        tcr_encoded["input_ids"],
        torch.LongTensor(labels),
        pep_encoded["attention_mask"],
        tcr_encoded["attention_mask"],
    )


def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


# Global tokenizer wrapper instance
_tokenizer_wrapper = None

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))


def get_tokenizer_wrapper():
    """Get or create global tokenizer wrapper instance."""
    global _tokenizer_wrapper
    if _tokenizer_wrapper is None:
        _tokenizer_wrapper = ESM2TokenizerWrapper()
    return _tokenizer_wrapper


def data_load_HLA_ESM2(type_="train", fold=None, batch_size=batch_size):
    """
    Load HLA data with ESM2 tokenization.
    """
    data_dir = os.path.join(_project_root, "data", "data_HLA")
    tokenizer_wrapper = get_tokenizer_wrapper()

    if type_ != "train" and type_ != "val":
        data = pd.read_csv(os.path.join(data_dir, "{}_set.csv".format(type_)))
    elif type_ == "train":
        data = pd.read_csv(os.path.join(data_dir, "train_fold_{}.csv".format(fold)))
    elif type_ == "val":
        data = pd.read_csv(os.path.join(data_dir, "val_fold_{}.csv".format(fold)))

    pep_inputs, hla_inputs, labels, pep_masks, hla_masks = data_process_HLA_ESM2(
        data, tokenizer_wrapper
    )
    loader = Data.DataLoader(
        MyDataSet_HLA_ESM2(pep_inputs, hla_inputs, labels, pep_masks, hla_masks),
        batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    return loader


def data_load_TCR_ESM2(type_="train", fold=None, batch_size=batch_size):
    """
    Load TCR data with ESM2 tokenization.
    """
    data_dir = os.path.join(_project_root, "data", "data_TCR")
    tokenizer_wrapper = get_tokenizer_wrapper()

    if type_ != "train" and type_ != "val":
        data = pd.read_csv(os.path.join(data_dir, "{}_set.csv".format(type_))).dropna()
    elif type_ == "train":
        data = pd.read_csv(
            os.path.join(data_dir, "train_fold_{}.csv".format(fold))
        ).dropna()
    elif type_ == "val":
        data = pd.read_csv(
            os.path.join(data_dir, "val_fold_{}.csv".format(fold))
        ).dropna()

    pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks = data_process_TCR_ESM2(
        data, tokenizer_wrapper
    )
    loader = Data.DataLoader(
        MyDataSet_TCR_ESM2(pep_inputs, tcr_inputs, labels, pep_masks, tcr_masks),
        batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    return loader