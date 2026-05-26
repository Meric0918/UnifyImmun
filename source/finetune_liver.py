"""
Fine-tuning script for pTCR model using liver data (TCR_Peptide_balanced.csv)
Based on fine-tuning.txt requirements:
- Train: TCR encoder, peptide encoder, pTCR cross-attention, pTCR FC head
- Freeze: HLA encoder, pHLA cross-attention, pHLA FC head
- Learning rates: FC head (1e-4~1e-3), cross-attention (1e-4), encoder (1e-5~5e-5)
"""
import time
import math
import numpy as np
import pandas as pd
import random
import os
import copy
from collections import Counter
from tqdm import tqdm
from typing import Optional
import warnings
from datetime import datetime

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import swanlab

warnings.filterwarnings("ignore")

# ==================== Configuration ====================
pep_max_len = 15
tcr_max_len = 34
hla_max_len = 34

vocab = np.load('../data/data_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
n_heads = 1
d_model = 64
d_ff = 512
d_k = d_v = 64
n_layers = 1

batch_size = 64  # Reduced for smaller dataset
epochs = 50
threshold = 0.5
seed = 66

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"Using device: {device}")

# ==================== Learning Rates (from fine-tuning.txt) ====================
lr_fc_head = 5e-4          # pTCR FC head: 1e-4 ~ 1e-3
lr_cross_attention = 1e-4  # pTCR cross-attention
lr_tcr_encoder = 3e-5      # TCR encoder: 1e-5 ~ 5e-5
lr_peptide_encoder = 1e-5  # peptide encoder

# ==================== Model Components ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=34):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding_padding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding_padding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pad = torch.zeros(34, d_model)
        pad[:pe.shape[0], :] = pe
        pe = pad.unsqueeze(0).transpose(0, 1).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(device) + self.pe[:x.size(0), :].to(device)
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
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

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = nn.Dropout(0.1)(output)
        return nn.LayerNorm(d_model).to(device)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(0.1)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs1 = enc_inputs + self.dropout(enc_outputs)
        enc_outputs1 = nn.LayerNorm(d_model).to(device)(enc_outputs1)
        enc_outputs = self.pos_ffn(enc_outputs1)
        return enc_outputs, attn


class Encoder(nn.Module):
    """TCR Encoder - Will be trained"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Encoder_padding(nn.Module):
    """Peptide Encoder - Will be trained"""
    def __init__(self):
        super(Encoder_padding, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb_padding = PositionalEncoding_padding(d_model, max_len=pep_max_len)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_pad = torch.zeros(enc_inputs.shape[0], tcr_max_len, d_model).to(device)
        enc_pad[:, :enc_outputs.shape[1], :] = enc_outputs
        enc_outputs = enc_pad
        enc_outputs = self.pos_emb_padding(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
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

    def forward(self, pep_inputs, tcr_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(pep_inputs, tcr_inputs, tcr_inputs, dec_self_attn_mask)
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class Cross_Attention(nn.Module):
    """pTCR Cross Attention - Will be trained"""
    def __init__(self):
        super(Cross_Attention, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.pos_peptide = PositionalEncoding_padding(d_model, max_len=15)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tcr_max_len

    def forward(self, pep_inputs, tcr_inputs):
        pep_outputs = pep_inputs.to(device)
        tcr_outputs = tcr_inputs.to(device)
        dec_self_attn_pad_mask = torch.LongTensor(
            np.zeros((pep_inputs.shape[0], tcr_max_len, tcr_max_len))
        ).bool().to(device)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(pep_outputs, tcr_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


class FineTuningModel(nn.Module):
    """
    Fine-tuning model for pTCR prediction.
    Architecture:
    - encoder_P: Peptide encoder (trainable)
    - encoder_T: TCR encoder (trainable)
    - cross_2: pTCR cross-attention (trainable)
    - projection: FC head (trainable)

    Frozen components from HLA model (not used in this model but weights are loaded):
    - encoder_H: HLA encoder
    - cross_1: pHLA cross-attention
    """
    def __init__(self):
        super(FineTuningModel, self).__init__()
        self.encoder_P = Encoder_padding().to(device)  # Peptide encoder - trainable
        self.encoder_T = Encoder().to(device)          # TCR encoder - trainable
        self.cross_2 = Cross_Attention().to(device)    # pTCR cross-attention - trainable
        self.projection = nn.Sequential(
            nn.Linear(tcr_max_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        ).to(device)

    def forward(self, pep_inputs, tcr_inputs):
        tcr_enc, tcr_attn = self.encoder_T(tcr_inputs)
        pep_enc, enc1_attn = self.encoder_P(pep_inputs)
        pep_tcr, pep_tcr_attn = self.cross_2(pep_enc, tcr_enc)
        pep_tcr_outputs = pep_tcr.view(pep_tcr.shape[0], -1)
        pep_tcr_logits = self.projection(pep_tcr_outputs)
        return pep_tcr_logits.view(-1, pep_tcr_logits.size(-1)), pep_tcr_attn

    def load_pretrained_peptide_encoder(self, pretrained_path):
        """Load pretrained peptide encoder weights from HLA model"""
        print(f"Loading pretrained peptide encoder from: {pretrained_path}")
        try:
            hla_model_state = torch.load(pretrained_path, map_location=device)
            # Extract encoder_P weights from HLA model
            encoder_P_state = {}
            for key, value in hla_model_state.items():
                if key.startswith('encoder_P.'):
                    # Remove 'encoder_P.' prefix to match our model's keys
                    new_key = key.replace('encoder_P.', '')
                    encoder_P_state[new_key] = value

            if encoder_P_state:
                self.encoder_P.load_state_dict(encoder_P_state)
                print(f"Loaded pretrained peptide encoder with {len(encoder_P_state)} parameters")
                # Print loaded keys for verification
                print("Loaded encoder_P keys:", list(encoder_P_state.keys())[:3], "...")
            else:
                print("Warning: Could not find encoder_P weights in pretrained model")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")

    def set_parameter_requires_grad(self, freeze_peptide_encoder=False):
        """
        Set parameter freezing based on fine-tuning strategy.
        By default:
        - TCR encoder: trainable
        - Peptide encoder: trainable (but initialized from pretrained)
        - pTCR cross-attention: trainable
        - FC head: trainable

        If freeze_peptide_encoder=True, peptide encoder will be frozen.
        """
        # All parameters are trainable by default in this fine-tuning setup
        # according to fine-tuning.txt requirements
        for param in self.encoder_T.parameters():
            param.requires_grad = True
        for param in self.encoder_P.parameters():
            param.requires_grad = not freeze_peptide_encoder
        for param in self.cross_2.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def get_parameter_groups(self):
        """
        Get parameter groups with different learning rates.
        Returns parameter groups for optimizer with layered learning rates.
        """
        return [
            {'params': self.projection.parameters(), 'lr': lr_fc_head},
            {'params': self.cross_2.parameters(), 'lr': lr_cross_attention},
            {'params': self.encoder_T.parameters(), 'lr': lr_tcr_encoder},
            {'params': self.encoder_P.parameters(), 'lr': lr_peptide_encoder},
        ]


# ==================== Data Processing ====================
def data_process_liver(data):
    """Process liver TCR-Peptide data"""
    pep_inputs, tcr_inputs, labels = [], [], []
    for pep, tcr, label in zip(data.peptide, data.tcr, data.label):
        pep = pep.ljust(tcr_max_len, '-')
        tcr = tcr.ljust(tcr_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]
        tcr_input = [[vocab[n] for n in tcr]]
        pep_inputs.extend(pep_input)
        tcr_inputs.extend(tcr_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(tcr_inputs), torch.LongTensor(labels)


class LiverDataSet(Data.Dataset):
    def __init__(self, pep_inputs, tcr_inputs, labels):
        super(LiverDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.tcr_inputs = tcr_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.tcr_inputs[idx], self.labels[idx]


def split_and_save_data(data_path, output_dir, train_ratio=7, val_ratio=1.5, test_ratio=1.5, n_splits=5):
    """
    Split liver data into train/val/test sets using stratified split.
    Ratio: train:val:test = 7:1.5:1.5

    IMPORTANT: Removes duplicate TCR-peptide pairs to prevent data leakage.
    Same TCR-peptide pair will always appear in the same split.

    Args:
        data_path: Path to input CSV file
        output_dir: Directory to save split files
        train_ratio: Training set ratio (default 7)
        val_ratio: Validation set ratio (default 1.5)
        test_ratio: Test set ratio (default 1.5)
        n_splits: Number of cross-validation folds for training/validation within train+val portion
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold

    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path).dropna()
    print(f"Total samples: {len(data)}")
    print(f"Label distribution: {Counter(data.label)}")

    # Remove duplicates to prevent data leakage
    # Keep first occurrence of each unique TCR-peptide pair
    duplicate_count = len(data) - len(data.drop_duplicates(subset=['peptide', 'tcr']))
    if duplicate_count > 0:
        print(f"\nWARNING: Found {duplicate_count} duplicate TCR-peptide pairs. Removing duplicates to prevent data leakage.")
        data = data.drop_duplicates(subset=['peptide', 'tcr'], keep='first')
        print(f"After removing duplicates: {len(data)} samples")
        print(f"Label distribution after dedup: {Counter(data.label)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    train_size = train_ratio / total_ratio  # 7/10 = 70%
    val_size = val_ratio / total_ratio       # 1.5/10 = 15%
    test_size = test_ratio / total_ratio     # 1.5/10 = 15%

    print(f"\nSplit ratio: train:{train_ratio} | val:{val_ratio} | test:{test_ratio}")
    print(f"Split percentages: train={train_size*100:.1f}% | val={val_size*100:.1f}% | test={test_size*100:.1f}%")

    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=seed, stratify=data.label
    )

    # Save test set
    test_path = os.path.join(output_dir, 'test.csv')
    test_data.to_csv(test_path, index=False)
    print(f"\nTest set saved: {len(test_data)} samples ({Counter(test_data.label)})")

    # Second split: separate train and val from train_val
    # Use Stratified K-Fold for multiple folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_data, train_val_data.label), 1):
        fold_train_data = train_val_data.iloc[train_idx]
        fold_val_data = train_val_data.iloc[val_idx]

        train_path = os.path.join(output_dir, f'train_fold_{fold}.csv')
        val_path = os.path.join(output_dir, f'val_fold_{fold}.csv')

        fold_train_data.to_csv(train_path, index=False)
        fold_val_data.to_csv(val_path, index=False)

        print(f"Fold {fold}: Train={len(fold_train_data)} ({dict(Counter(fold_train_data.label))}), "
              f"Val={len(fold_val_data)} ({dict(Counter(fold_val_data.label))})")

    # Also save the full train+val data for reference
    train_val_path = os.path.join(output_dir, 'train_val_full.csv')
    train_val_data.to_csv(train_val_path, index=False)
    print(f"\nTrain+Val full set saved: {len(train_val_data)} samples")

    print(f"\nAll data splits saved to: {output_dir}")

    # Print summary
    expected_train = int(len(data) * train_size)
    expected_val = int(len(data) * val_size)
    expected_test = int(len(data) * test_size)
    print(f"\nExpected sizes (approx): Train={expected_train}, Val={expected_val}, Test={expected_test}")

    return output_dir


def data_load_liver(type_='train', fold=None, batch_size=batch_size, data_dir='../data/data_liver'):
    """Load liver dataset"""
    if type_ == 'train':
        data = pd.read_csv(f'{data_dir}/train_fold_{fold}.csv').dropna()
    elif type_ == 'val':
        data = pd.read_csv(f'{data_dir}/val_fold_{fold}.csv').dropna()
    elif type_ == 'test':
        data = pd.read_csv(f'{data_dir}/test.csv').dropna()
    else:
        data = pd.read_csv(f'{data_dir}/{type_}.csv').dropna()

    pep_inputs, tcr_inputs, labels = data_process_liver(data)
    loader = Data.DataLoader(
        LiverDataSet(pep_inputs, tcr_inputs, labels),
        batch_size=batch_size,
        shuffle=(type_ == 'train'),
        num_workers=0,
        drop_last=False
    )
    return loader, data


# ==================== Training Utilities ====================
def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


class EarlyStopping:
    """Early stopping helper to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def performance(y_true, y_pred, y_pred_transfer):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_transfer)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_transfer, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0
    prec, reca, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true, y_pred_transfer)

    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
    print(f'y_pred: 0={Counter(y_pred_transfer)[0]} | 1={Counter(y_pred_transfer)[1]}')
    print(f'y_true: 0={Counter(y_true)[0]} | 1={Counter(y_true)[1]}')
    print(f'auc={roc_auc:.4f}|sensitivity={sensitivity:.4f}|specificity={specificity:.4f}|acc={accuracy:.4f}|mcc={mcc:.4f}')
    print(f'precision={precision:.4f}|recall={recall:.4f}|f1={f1:.4f}|aupr={aupr:.4f}')

    return (roc_auc, accuracy, mcc, f1, aupr, sensitivity, specificity, precision, recall)


f_mean = lambda l: sum(l) / len(l)


def performance_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'aupr', 'sensitivity', 'specificity', 'precision', 'recall']
    performance_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performance_pd.loc['mean'] = performance_pd.mean(axis=0)
    performance_pd.loc['std'] = performance_pd.std(axis=0)
    return performance_pd


class FGM():
    """Adversarial training helper"""
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_names=['encoder_T.src_emb', 'encoder_P.src_emb']):
        """Attack multiple embedding layers at once"""
        for name, param in self.model.named_parameters():
            for emb_name in emb_names:
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_names=['encoder_T.src_emb', 'encoder_P.src_emb']):
        """Restore multiple embedding layers at once"""
        for name, param in self.model.named_parameters():
            for emb_name in emb_names:
                if param.requires_grad and emb_name in name:
                    if name in self.backup:
                        param.data = self.backup[name]
        self.backup = {}


def train_finetune(model, train_loader, fold, epoch, epochs, optimizer, criterion, fgm):
    """Training function for fine-tuning"""
    train_time = 0
    model.train()
    y_true_list, y_pred_list = [], []
    loss_list = []

    for train_pep_inputs, train_tcr_inputs, train_labels in tqdm(train_loader, colour='yellow'):
        train_pep_inputs = train_pep_inputs.to(device)
        train_tcr_inputs = train_tcr_inputs.to(device)
        train_labels = train_labels.to(device)

        start = time.time()
        train_outputs, _ = model(train_pep_inputs, train_tcr_inputs)
        train_loss = criterion(train_outputs, train_labels)
        train_loss.backward()
        train_time += time.time() - start

        # Adversarial training
        fgm.attack(emb_names=['encoder_T.src_emb', 'encoder_P.src_emb'])
        train_outputs2, _ = model(train_pep_inputs, train_tcr_inputs)
        loss_sum = criterion(train_outputs2, train_labels)
        loss_sum.backward()
        fgm.restore(emb_names=['encoder_T.src_emb', 'encoder_P.src_emb'])

        optimizer.step()
        optimizer.zero_grad()

        y_true_train = train_labels.cpu().numpy()
        y_pred_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_list.extend(y_true_train)
        y_pred_list.extend(y_pred_train)
        loss_list.append(train_loss.item())

    y_pred_transfer_train_list = transfer(y_pred_list, threshold)
    print(f'Fold-{fold} Train: Epoch:{epoch}/{epochs} Loss={f_mean(loss_list):.4f} | Time={train_time:.4f} sec')
    performance_train = performance(y_true_list, y_pred_list, y_pred_transfer_train_list)

    # Log training metrics to swanlab
    swanlab.log({
        f"train/fold_{fold}/loss": f_mean(loss_list),
        f"train/fold_{fold}/auc": performance_train[0],
        f"train/fold_{fold}/accuracy": performance_train[1],
        f"train/fold_{fold}/mcc": performance_train[2],
        f"train/fold_{fold}/f1": performance_train[3],
        f"train/fold_{fold}/aupr": performance_train[4],
        f"train/fold_{fold}/sensitivity": performance_train[5],
        f"train/fold_{fold}/specificity": performance_train[6],
        f"train/fold_{fold}/precision": performance_train[7],
        f"train/fold_{fold}/recall": performance_train[8],
        "epoch": epoch,
        "fold": fold,
    })

    return performance_train, train_time


def valid_finetune(model, val_loader, fold, epoch, epochs, criterion):
    """Validation function for fine-tuning"""
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)

    with torch.no_grad():
        y_true_val_list, y_pred_val_list = [], []
        loss_val_list = []

        for val_pep_inputs, val_tcr_inputs, val_labels in tqdm(val_loader, colour='blue'):
            val_pep_inputs = val_pep_inputs.to(device)
            val_tcr_inputs = val_tcr_inputs.to(device)
            val_labels = val_labels.to(device)

            val_outputs, _ = model(val_pep_inputs, val_tcr_inputs)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_pred_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val_list.extend(y_true_val)
            y_pred_val_list.extend(y_pred_val)
            loss_val_list.append(val_loss.item())

        y_pred_transfer_val_list = transfer(y_pred_val_list, threshold)
        print(f'Fold-{fold} Valid: Epoch:{epoch}/{epochs} Loss={f_mean(loss_val_list):.4f}')
        performance_val = performance(y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)

    # Log validation metrics to swanlab
    swanlab.log({
        f"val/fold_{fold}/loss": f_mean(loss_val_list),
        f"val/fold_{fold}/auc": performance_val[0],
        f"val/fold_{fold}/accuracy": performance_val[1],
        f"val/fold_{fold}/mcc": performance_val[2],
        f"val/fold_{fold}/f1": performance_val[3],
        f"val/fold_{fold}/aupr": performance_val[4],
        f"val/fold_{fold}/sensitivity": performance_val[5],
        f"val/fold_{fold}/specificity": performance_val[6],
        f"val/fold_{fold}/precision": performance_val[7],
        f"val/fold_{fold}/recall": performance_val[8],
    })

    return performance_val


def test_finetune(model, test_loader, criterion, fold=None):
    """Test function for final evaluation on test set"""
    model.eval()

    with torch.no_grad():
        y_true_test_list, y_pred_test_list = [], []
        loss_test_list = []

        for test_pep_inputs, test_tcr_inputs, test_labels in tqdm(test_loader, colour='green', desc='Testing'):
            test_pep_inputs = test_pep_inputs.to(device)
            test_tcr_inputs = test_tcr_inputs.to(device)
            test_labels = test_labels.to(device)

            test_outputs, _ = model(test_pep_inputs, test_tcr_inputs)
            test_loss = criterion(test_outputs, test_labels)

            y_true_test = test_labels.cpu().numpy()
            y_pred_test = nn.Softmax(dim=1)(test_outputs)[:, 1].cpu().detach().numpy()
            y_true_test_list.extend(y_true_test)
            y_pred_test_list.extend(y_pred_test)
            loss_test_list.append(test_loss.item())

        y_pred_transfer_test_list = transfer(y_pred_test_list, threshold)
        print(f'\nTest: Loss={f_mean(loss_test_list):.4f}')
        performance_test = performance(y_true_test_list, y_pred_test_list, y_pred_transfer_test_list)

    # Log test metrics to swanlab
    log_dict = {
        "test/loss": f_mean(loss_test_list),
        "test/auc": performance_test[0],
        "test/accuracy": performance_test[1],
        "test/mcc": performance_test[2],
        "test/f1": performance_test[3],
        "test/aupr": performance_test[4],
        "test/sensitivity": performance_test[5],
        "test/specificity": performance_test[6],
        "test/precision": performance_test[7],
        "test/recall": performance_test[8],
    }
    if fold is not None:
        log_dict["test/fold"] = fold
    swanlab.log(log_dict)

    return performance_test


# ==================== Main Training Function ====================
def run_fine_tuning(data_path=None,
                    pretrained_encoder_path='../trained_model/HLA_2/model_HLA.pkl',
                    output_dir='../trained_model/finetune_liver',
                    n_folds=5,
                    freeze_peptide_encoder=False,
                    swanlab_project="UnifyImmun-finetune",
                    swanlab_experiment="liver-pTCR"):
    """
    Run fine-tuning process.

    Args:
        data_path: Path to input dataset CSV file (REQUIRED, user must specify)
        pretrained_encoder_path: Path to pretrained HLA model for peptide encoder
        output_dir: Base directory to save fine-tuned models (timestamp will be appended)
        n_folds: Number of cross-validation folds
        freeze_peptide_encoder: Whether to freeze peptide encoder (default False per fine-tuning.txt)
        swanlab_project: SwanLab project name
        swanlab_experiment: SwanLab experiment name (timestamp will be appended)
    """
    if data_path is None:
        raise ValueError("data_path must be specified by the user. Example: '../data/data_liver/TCR_Peptide.csv'")

    start_time = time.time()

    # Add timestamp to output directory to avoid overwriting previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract dataset folder name from data_path
    dataset_folder = os.path.basename(os.path.dirname(data_path))
    output_dir = f"{output_dir}_{dataset_folder}_{timestamp}"
    swanlab_experiment = f"{swanlab_experiment}_{dataset_folder}_{timestamp}"
    print(f"Output directory: {output_dir}")
    print(f"SwanLab experiment: {swanlab_experiment}")

    # Initialize SwanLab
    swanlab.init(
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "n_folds": n_folds,
            "lr_fc_head": lr_fc_head,
            "lr_cross_attention": lr_cross_attention,
            "lr_tcr_encoder": lr_tcr_encoder,
            "lr_peptide_encoder": lr_peptide_encoder,
            "seed": seed,
            "threshold": threshold,
            "freeze_peptide_encoder": freeze_peptide_encoder,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "data_path": data_path,
            "pretrained_encoder_path": pretrained_encoder_path,
        },
        description="Fine-tuning pTCR model for liver data prediction"
    )

    # Step 1: Split data into train/val/test with ratio 7:1.5:1.5
    # Use the same directory as the input data file for saving splits
    data_dir = os.path.dirname(data_path)
    split_and_save_data(data_path, data_dir, train_ratio=7, val_ratio=1.5, test_ratio=1.5, n_splits=n_folds)

    # Step 2: Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 3: Training loop
    all_fold_performances = []
    all_test_performances = []

    # Load test data
    test_loader, test_data = data_load_liver(type_='test', fold=None, batch_size=batch_size, data_dir=data_dir)
    print(f'Test set: {len(test_data)} samples ({Counter(test_data.label)})')

    for fold in range(1, n_folds + 1):
        print(f'\n{"="*60}')
        print(f'Fold-{fold}: Fine-tuning Started')
        print(f'{"="*60}')

        # Initialize model
        model = FineTuningModel()

        # Load pretrained peptide encoder (shared component from HLA model)
        # Note: We extract encoder_P from the HLA model, other components are newly initialized
        try:
            hla_model_state = torch.load(pretrained_encoder_path, map_location=device)
            # Extract encoder_P weights
            encoder_P_state = {}
            for key, value in hla_model_state.items():
                if key.startswith('encoder_P.'):
                    new_key = key.replace('encoder_P.', '')
                    encoder_P_state[new_key] = value

            if encoder_P_state:
                model.encoder_P.load_state_dict(encoder_P_state)
                print(f"Loaded pretrained peptide encoder with {len(encoder_P_state)} parameters")
                print("Loaded encoder_P keys:", list(encoder_P_state.keys())[:3], "...")
            else:
                print("Warning: Could not find encoder_P weights in pretrained model, using random initialization")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}, using random initialization")

        # Note: encoder_T (TCR encoder), cross_2 (pTCR cross-attention), and projection (FC head)
        # are newly initialized and will be trained from scratch

        # Set freezing strategy
        model.set_parameter_requires_grad(freeze_peptide_encoder=freeze_peptide_encoder)

        # Setup optimizer with layered learning rates
        optimizer = optim.Adam(model.get_parameter_groups())
        criterion = nn.CrossEntropyLoss()
        fgm = FGM(model)

        # Load data
        print(f'Loading liver data for fold {fold}')
        train_loader, train_data = data_load_liver(type_='train', fold=fold, batch_size=batch_size, data_dir=data_dir)
        val_loader, val_data = data_load_liver(type_='val', fold=fold, batch_size=batch_size, data_dir=data_dir)
        print(f'Fold-{fold} Label: Train={Counter(train_data.label)} | Val={Counter(val_data.label)}')

        # Training
        save_path = os.path.join(output_dir, f'model_finetune_fold{fold}.pkl')
        performance_best, epoch_best = 0, -1
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')

        for epoch in range(1, epochs + 1):
            performance_train, train_time = train_finetune(
                model, train_loader, fold, epoch, epochs, optimizer, criterion, fgm
            )
            performance_val = valid_finetune(model, val_loader, fold, epoch, epochs, criterion)

            # Use average of key metrics for best model selection
            performance_avg = (performance_val[0] + performance_val[1] + performance_val[2] + performance_val[3]) / 4

            if performance_avg > performance_best:
                performance_best, epoch_best = performance_avg, epoch
                print(f'****Saving model: Best epoch={epoch_best} | metrics_avg={performance_best:.4f}')
                torch.save(model.state_dict(), save_path)

            # Check early stopping
            if early_stopping(performance_avg):
                print(f'Early stopping triggered at epoch {epoch}')
                break

        # Load best model and evaluate
        print(f'\n-----Fold-{fold} Final Evaluation-----')
        model.load_state_dict(torch.load(save_path))
        model.eval()

        # Final validation
        final_val_loader, _ = data_load_liver(type_='val', fold=fold, batch_size=batch_size, data_dir=data_dir)
        final_performance = valid_finetune(model, final_val_loader, fold, epoch_best, epochs, criterion)
        all_fold_performances.append(final_performance)

        # Test evaluation
        print(f'\n-----Fold-{fold} Test Evaluation-----')
        test_performance = test_finetune(model, test_loader, criterion, fold=fold)
        all_test_performances.append(test_performance)

        print(f'Fold-{fold} Best epoch: {epoch_best}')
        print(f'Fold-{fold} Val performance: AUC={final_performance[0]:.4f}, ACC={final_performance[1]:.4f}')
        print(f'Fold-{fold} Test performance: AUC={test_performance[0]:.4f}, ACC={test_performance[1]:.4f}')

    # Final summary
    end_time = time.time()
    use_time = end_time - start_time
    print(f'\n{"="*60}')
    print(f'Fine-tuning Completed')
    print(f'{"="*60}')
    print(f'Total time: {use_time:.2f} seconds')

    # Log final summary to swanlab
    swanlab.log({
        "summary/total_time_seconds": use_time,
        "summary/num_folds": n_folds,
    })

    print(f'\n{"="*60}')
    print(f'Cross-validation Validation Results:')
    print(f'{"="*60}')
    val_results_df = performance_pd(all_fold_performances)
    print(val_results_df.to_string())

    # Log validation mean/std to swanlab
    swanlab.log({
        "summary/val_mean_auc": val_results_df.loc['mean', 'roc_auc'],
        "summary/val_mean_accuracy": val_results_df.loc['mean', 'accuracy'],
        "summary/val_mean_f1": val_results_df.loc['mean', 'f1'],
        "summary/val_std_auc": val_results_df.loc['std', 'roc_auc'],
        "summary/val_std_accuracy": val_results_df.loc['std', 'accuracy'],
    })

    print(f'\n{"="*60}')
    print(f'Test Set Results (All Folds):')
    print(f'{"="*60}')
    test_results_df = performance_pd(all_test_performances)
    print(test_results_df.to_string())

    # Log test mean/std to swanlab
    swanlab.log({
        "summary/test_mean_auc": test_results_df.loc['mean', 'roc_auc'],
        "summary/test_mean_accuracy": test_results_df.loc['mean', 'accuracy'],
        "summary/test_mean_f1": test_results_df.loc['mean', 'f1'],
        "summary/test_mean_mcc": test_results_df.loc['mean', 'mcc'],
        "summary/test_std_auc": test_results_df.loc['std', 'roc_auc'],
        "summary/test_std_accuracy": test_results_df.loc['std', 'accuracy'],
    })

    # Save test results to file
    test_results_path = os.path.join(output_dir, 'test_results.csv')
    test_results_df.to_csv(test_results_path)
    print(f'\nTest results saved to: {test_results_path}')

    # Finish swanlab run
    swanlab.finish()

    return all_fold_performances, all_test_performances


# ==================== Entry Point ====================
if __name__ == '__main__':
    # User must specify the data_path before running
    # Example: data_path='../data/data_liver/TCR_Peptide_balanced.csv'
    data_path = '../data/data_liver/TCR_Peptide_balanced.csv'  # TODO: Modify this path as needed

    val_performances, test_performances = run_fine_tuning(
        data_path=data_path,
        pretrained_encoder_path='../trained_model/HLA_2/model_HLA.pkl',
        output_dir='../trained_model/finetune_liver',
        n_folds=5,
        freeze_peptide_encoder=False,  # Per fine-tuning.txt, peptide encoder should be trainable
        swanlab_project="UnifyImmun-finetune",
        swanlab_experiment="liver-pTCR"
    )