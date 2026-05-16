import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.TCR import Mymodel_tcr, MyDataSet_tcr, data_process_tcr, vocab, device

pep_max_len = 15
tcr_max_len = 34
batch_size = 1024
threshold = 0.5

seed = 66
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_model(model_path):
    model = Mymodel_tcr().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_binding_scores(model, tcr_sequences, peptide_sequences):
    data = pd.DataFrame(
        {
            "tcr": tcr_sequences,
            "peptide": peptide_sequences,
            "label": [0] * len(tcr_sequences),
        }
    )

    pep_inputs, tcr_inputs, labels = data_process_tcr(data)
    dataset = MyDataSet_tcr(pep_inputs, tcr_inputs, labels)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    all_scores = []
    with torch.no_grad():
        for pep_batch, tcr_batch, _ in tqdm(
            loader, desc="Predicting", colour="green", leave=False
        ):
            pep_batch = pep_batch.to(device)
            tcr_batch = tcr_batch.to(device)
            outputs, _ = model(pep_batch, tcr_batch)
            probs = nn.Softmax(dim=1)(outputs)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())

    return np.array(all_scores)


def rank_peptides_by_avg_binding(model, df, tcr_col="tcr", peptide_col="peptide"):
    unique_peptides = df[peptide_col].unique()

    results = []
    for peptide in tqdm(unique_peptides, desc="Processing peptides", colour="blue"):
        peptide_data = df[df[peptide_col] == peptide]
        tcrs = peptide_data[tcr_col].tolist()

        tcr_sequences = tcrs
        peptide_sequences = [peptide] * len(tcrs)

        scores = predict_binding_scores(model, tcr_sequences, peptide_sequences)
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        n_tcrs = len(tcrs)

        broad_tcr_score = avg_score * n_tcrs / (std_score + 0.1)

        results.append(
            {
                "peptide": peptide,
                "avg_binding_score": avg_score,
                "std_binding_score": std_score,
                "max_binding_score": max_score,
                "min_binding_score": min_score,
                "n_tcrs": n_tcrs,
                "broad_tcr_score": broad_tcr_score,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("broad_tcr_score", ascending=False).reset_index(
        drop=True
    )
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[
        [
            "rank",
            "peptide",
            "broad_tcr_score",
            "avg_binding_score",
            "std_binding_score",
            "max_binding_score",
            "min_binding_score",
            "n_tcrs",
        ]
    ]

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Peptide Ranking by Broad TCR Binding Score"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../trained_model/TCR_2/model_TCR.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file (with tcr and peptide columns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="peptide_ranking_results.csv",
        help="Output CSV file",
    )
    parser.add_argument("--tcr_col", type=str, default="tcr", help="TCR column name")
    parser.add_argument(
        "--peptide_col", type=str, default="peptide", help="Peptide column name"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    print("Model loaded successfully!")

    print(f"\nLoading data from {args.input}...")
    df = pd.read_csv(args.input).dropna(subset=[args.tcr_col, args.peptide_col])
    print(f"Loaded {len(df)} TCR-peptide pairs")
    print(f"Unique peptides: {df[args.peptide_col].nunique()}")
    print(f"Unique TCRs: {df[args.tcr_col].nunique()}")

    print("\nCalculating binding scores and ranking peptides...")
    results = rank_peptides_by_avg_binding(model, df, args.tcr_col, args.peptide_col)

    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 80)
    print("Top 10 Peptides by Broad TCR Score:")
    print("=" * 80)
    print(results.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("Bottom 10 Peptides by Broad TCR Score:")
    print("=" * 80)
    print(results.tail(10).to_string(index=False))
