import os
import argparse
import warnings
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from models.TCR import Mymodel_tcr, vocab

warnings.filterwarnings("ignore")

pep_max_len = 15
tcr_max_len = 34


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TCR-peptide binding probability matrix"
    )
    parser.add_argument(
        "--peptide-csv",
        type=str,
        required=True,
        help="CSV file containing 'peptide' column",
    )
    parser.add_argument(
        "--cdr3-csv",
        type=str,
        required=True,
        help="CSV file containing 'cdr3' column",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../trained_model/TCR_2/model_TCR.pkl",
        help="Path to trained TCR model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="probability_matrix.csv",
        help="Output CSV file for the probability matrix",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for prediction (default: auto-tune based on GPU memory)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for data loading (default: CPU cores * 0.7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision for faster inference",
    )
    parser.add_argument(
        "--cuda-streams",
        type=int,
        default=2,
        help="Number of CUDA streams for async execution",
    )
    return parser.parse_args()


def encode_sequence(seq, max_len, vocab_dict):
    seq = str(seq).ljust(max_len, "-")
    return [vocab_dict.get(c, vocab_dict.get("-", 0)) for c in seq[:max_len]]


def encode_pair_worker(pep_cdr3_pair, tcr_max_len, vocab_dict):
    i, j, pep, cdr3 = pep_cdr3_pair
    pep_encoded = encode_sequence(pep, tcr_max_len, vocab_dict)
    cdr3_encoded = encode_sequence(cdr3, tcr_max_len, vocab_dict)
    return (i, j, pep_encoded, cdr3_encoded)


def load_sequences(csv_path, column_name):
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in {csv_path}")
    sequences = df[column_name].dropna().unique().tolist()
    return sequences


def get_optimal_batch_size(device, model):
    if device.type != "cuda":
        return 1024

    total_memory = torch.cuda.get_device_properties(0).total_memory

    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - reserved_memory

    target_usage = 0.7
    target_memory = free_memory * target_usage

    sample_pep = torch.randint(0, 25, (1, tcr_max_len), dtype=torch.long).to(device)
    sample_tcr = torch.randint(0, 25, (1, tcr_max_len), dtype=torch.long).to(device)

    torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        _ = model(sample_pep, sample_tcr)
    single_sample_memory = torch.cuda.max_memory_allocated(0)

    overhead = 50 * 1024 * 1024
    single_sample_memory += overhead

    optimal_batch_size = int(target_memory / single_sample_memory)
    optimal_batch_size = max(512, min(optimal_batch_size, 16384))

    return optimal_batch_size


def get_gpu_utilization():
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(",")
                gpu_util = float(parts[0].strip())
                mem_used = float(parts[1].strip())
                mem_total = float(parts[2].strip())
                mem_util = (mem_used / mem_total) * 100
                return gpu_util, mem_util, mem_used, mem_total
    except:
        pass
    return None, None, None, None


def prepare_batch_parallel(peptides, cdr3s, batch_size, num_workers, device):
    n_pep = len(peptides)
    n_cdr3 = len(cdr3s)
    total = n_pep * n_cdr3

    print(f"\n{'=' * 60}")
    print(f"Preparing data with {num_workers} workers...")
    print(f"{'=' * 60}")

    start_time = time.time()

    pairs = []
    for i, pep in enumerate(peptides):
        for j, cdr3 in enumerate(cdr3s):
            pairs.append((i, j, pep, cdr3))

    encode_func = partial(
        encode_pair_worker,
        tcr_max_len=tcr_max_len,
        vocab_dict=vocab,
    )

    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(encode_func, pairs, chunksize=1000),
                total=total,
                desc="Encoding pairs",
                colour="yellow",
            )
        )

    all_pep_inputs = []
    all_tcr_inputs = []
    pep_indices = []
    cdr3_indices = []

    for i, j, pep_enc, cdr3_enc in results:
        all_pep_inputs.append(pep_enc)
        all_tcr_inputs.append(cdr3_enc)
        pep_indices.append(i)
        cdr3_indices.append(j)

    pep_tensor = torch.LongTensor(all_pep_inputs)
    tcr_tensor = torch.LongTensor(all_tcr_inputs)

    pin_memory = device.type == "cuda"

    dataset = torch.utils.data.TensorDataset(pep_tensor, tcr_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type == "cuda" else num_workers,
        pin_memory=pin_memory,
    )

    elapsed = time.time() - start_time
    print(f"Data preparation completed in {elapsed:.2f}s")
    print(f"{'=' * 60}\n")

    return loader, n_pep, n_cdr3, pep_indices, cdr3_indices


class AsyncPredictor:
    def __init__(self, model, device, num_streams=2, use_fp16=False):
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.num_streams = num_streams

        if device.type == "cuda":
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
            if use_fp16:
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            self.streams = None

    def predict_batch(self, pep_inputs, tcr_inputs, stream_idx=0):
        if self.streams:
            with torch.cuda.stream(self.streams[stream_idx]):
                pep_inputs = pep_inputs.to(self.device, non_blocking=True)
                tcr_inputs = tcr_inputs.to(self.device, non_blocking=True)

                if self.use_fp16:
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs, _ = self.model(pep_inputs, tcr_inputs)
                else:
                    outputs, _ = self.model(pep_inputs, tcr_inputs)

                batch_probs = torch.softmax(outputs.float(), dim=1)[:, 1]
                return batch_probs.cpu()
        else:
            pep_inputs = pep_inputs.to(self.device)
            tcr_inputs = tcr_inputs.to(self.device)
            outputs, _ = self.model(pep_inputs, tcr_inputs)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            return batch_probs.cpu()


def predict(model, loader, device, total_pairs):
    model.eval()
    probs = []

    print(f"\n{'=' * 60}")
    print(f"Starting prediction on {device}...")
    print(f"{'=' * 60}")

    start_time = time.time()
    batch_count = 0
    total_batches = len(loader)

    with torch.no_grad():
        for pep_inputs, tcr_inputs in loader:
            batch_start = time.time()

            pep_inputs = pep_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)

            outputs, _ = model(pep_inputs, tcr_inputs)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy().tolist())

            batch_count += 1
            elapsed = time.time() - start_time
            batch_time = time.time() - batch_start

            pairs_done = batch_count * loader.batch_size
            pairs_done = min(pairs_done, total_pairs)
            progress_pct = (pairs_done / total_pairs) * 100

            if batch_count % 10 == 0 or batch_count == total_batches:
                pairs_per_sec = pairs_done / elapsed if elapsed > 0 else 0
                eta = (
                    (total_pairs - pairs_done) / pairs_per_sec
                    if pairs_per_sec > 0
                    else 0
                )

                gpu_util, mem_util, mem_used, mem_total = get_gpu_utilization()
                gpu_info = ""
                if gpu_util is not None:
                    gpu_info = f"GPU: {gpu_util:.0f}% | Mem: {mem_util:.0f}% ({mem_used:.0f}/{mem_total:.0f}MB) | "

                print(
                    f"Batch {batch_count}/{total_batches} | "
                    f"Progress: {progress_pct:.1f}% ({pairs_done}/{total_pairs}) | "
                    f"{gpu_info}"
                    f"Speed: {pairs_per_sec:.0f} pairs/s | "
                    f"Batch time: {batch_time * 1000:.1f}ms | "
                    f"ETA: {eta:.1f}s"
                )

    elapsed = time.time() - start_time
    print(f"\nPrediction completed in {elapsed:.2f}s")
    print(f"Average speed: {total_pairs / elapsed:.0f} pairs/s")

    if device.type == "cuda":
        gpu_util, mem_util, _, _ = get_gpu_utilization()
        if gpu_util is not None:
            print(f"Final GPU utilization: {gpu_util:.0f}%")
            print(f"Final Memory utilization: {mem_util:.0f}%")

    print(f"{'=' * 60}\n")

    return probs


def build_matrix(probs, n_pep, n_cdr3, pep_indices, cdr3_indices):
    print("Building probability matrix...")
    matrix = np.zeros((n_pep, n_cdr3))
    for idx, prob in enumerate(probs):
        i = pep_indices[idx]
        j = cdr3_indices[idx]
        matrix[i, j] = prob
    return matrix


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.7))
    else:
        num_workers = args.num_workers

    print(f"\n{'#' * 60}")
    print(f"TCR-Peptide Probability Matrix Generator")
    print(f"{'#' * 60}")
    print(f"Device: {device}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"DataLoader workers: {num_workers} (~70% CPU)")
    print(f"FP16: {args.fp16}")
    print(f"CUDA Streams: {args.cuda_streams}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    peptides = load_sequences(args.peptide_csv, "peptide")
    cdr3s = load_sequences(args.cdr3_csv, "cdr3")

    print(f"\nLoaded {len(peptides)} peptides, {len(cdr3s)} CDR3 sequences")
    print(f"Total pairs to predict: {len(peptides) * len(cdr3s):,}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(current_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model from: {model_path}")
    model = Mymodel_tcr().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Model loaded successfully!")

    if args.batch_size is None and device.type == "cuda":
        batch_size = get_optimal_batch_size(device, model)
        print(f"\nAuto-tuned batch size: {batch_size} (target 70% GPU memory)")
    else:
        batch_size = args.batch_size if args.batch_size else 1024

    loader, n_pep, n_cdr3, pep_indices, cdr3_indices = prepare_batch_parallel(
        peptides, cdr3s, batch_size, num_workers, device
    )

    total_pairs = n_pep * n_cdr3
    probs = predict(model, loader, device, total_pairs)

    matrix = build_matrix(probs, n_pep, n_cdr3, pep_indices, cdr3_indices)

    df_matrix = pd.DataFrame(matrix, index=peptides, columns=cdr3s)
    df_matrix.index.name = "peptide"

    df_matrix.to_csv(args.output)

    print(f"\n{'#' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'#' * 60}")
    print(f"Probability matrix saved to: {args.output}")
    print(f"Matrix shape: {matrix.shape} (peptides × CDR3)")
    print(f"Probability range: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print(f"Mean probability: {matrix.mean():.4f}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
