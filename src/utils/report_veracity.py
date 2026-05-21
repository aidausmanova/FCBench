"""
Aggregate veracity-prediction results across seeds and report mean ± std.

Reads prediction JSONs from:
    experiment_results/{dataset}/{model}_seed{seed}_veracity_prediction.json

Gold labels are loaded from:
    data/cleaned_datasets/{dataset}/dev.pkl

Usage:
    python -m src.utils.report_veracity
    python -m src.utils.report_veracity --datasets averitec scifact
    python -m src.utils.report_veracity --models bm25 tfidf
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


DATASETS = ["averitec", "scifact", "climatecheck", "climatefever"]

METRICS = [
    ("accuracy",           "Accuracy"),
    ("macro_precision",    "Macro-P"),
    ("macro_recall",       "Macro-R"),
    ("macro_f1",           "Macro-F1"),
    ("weighted_precision", "Wtd-P"),
    ("weighted_recall",    "Wtd-R"),
    ("weighted_f1",        "Wtd-F1"),
]


# ---------------------------------------------------------------------------
# Gold label loaders — replicate the DatasetBuilder processing for dev.pkl
# ---------------------------------------------------------------------------

def _scifact_claim_label(evidence: dict) -> str:
    if not evidence:
        return "Not Enough Evidence"
    labels = set()
    for doc in evidence.values():
        if doc is None:
            continue
        for e in doc:
            labels.add(e["label"])
    if "CONTRADICT" in labels:
        return "Refuted"
    if "SUPPORT" in labels:
        return "Supported"
    return "Not Enough Evidence"


_CLIMATEFEVER_LABEL_MAP = {
    0: "Supported",
    1: "Refuted",
    2: "Not Enough Evidence",
    3: "Conflicting Evidence/Cherrypicking",
}


def load_gold_labels(dataset_name: str, data_dir: str) -> list:
    """Return an ordered list of gold labels matching the prediction JSON claim_ids."""
    path = os.path.join(data_dir, dataset_name, "dev.pkl")
    df = pd.read_parquet(path)

    if dataset_name == "scifact":
        labels = df["evidence"].apply(_scifact_claim_label).reset_index(drop=True).tolist()

    elif dataset_name == "climatefever":
        df["label"] = df["claim_label"].map(_CLIMATEFEVER_LABEL_MAP)
        df = df.drop_duplicates(subset="claim_id", keep="first")
        labels = df["label"].reset_index(drop=True).tolist()

    elif dataset_name == "climatecheck":
        df = df[["text", "label"]].copy()
        df = df.dropna(subset=["label"])
        df = df[df["label"] != ""]
        labels = df["label"].reset_index(drop=True).tolist()

    else:  # averitec and any dataset with a plain label column
        labels = df["label"].reset_index(drop=True).tolist()

    return labels


# ---------------------------------------------------------------------------
# Prediction file scanner
# ---------------------------------------------------------------------------

def parse_prediction_files(results_dir: str, datasets: list) -> dict:
    """Scan JSON files and return {dataset: {model: {seed: [pred_label, ...]}}}.

    Files matching OLD_* are skipped.
    """
    pattern = re.compile(r"^(?!OLD_)(.+?)_seed(\d+)_veracity_prediction\.json$")
    data = defaultdict(lambda: defaultdict(dict))

    for dataset in datasets:
        ds_dir = os.path.join(results_dir, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            m = pattern.match(fname)
            if not m:
                continue
            model, seed = m.group(1), int(m.group(2))
            with open(os.path.join(ds_dir, fname)) as f:
                records = json.load(f)
            records.sort(key=lambda r: int(r["claim_id"]))
            data[dataset][model][seed] = [r["pred_label"] for r in records]

    return data


# ---------------------------------------------------------------------------
# Metric computation and aggregation
# ---------------------------------------------------------------------------

def _compute(y_true: list, y_pred: list) -> dict:
    n = min(len(y_true), len(y_pred))
    report = classification_report(y_true[:n], y_pred[:n], output_dict=True, zero_division=0.0)
    return {
        "accuracy":           report["accuracy"],
        "macro_precision":    report["macro avg"]["precision"],
        "macro_recall":       report["macro avg"]["recall"],
        "macro_f1":           report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall":    report["weighted avg"]["recall"],
        "weighted_f1":        report["weighted avg"]["f1-score"],
    }


def aggregate(predictions: dict, gold: dict) -> pd.DataFrame:
    rows = []
    for dataset, models in sorted(predictions.items()):
        gold_labels = gold[dataset]
        for model, seeds_dict in sorted(models.items()):
            per_seed = []
            for seed, preds in sorted(seeds_dict.items()):
                m = _compute(gold_labels, preds)
                m["seed"] = seed
                per_seed.append(m)

            row = {"dataset": dataset, "model": model,
                   "n_seeds": len(per_seed),
                   "seeds": [m["seed"] for m in per_seed]}
            for col, label in METRICS:
                vals = [m[col] for m in per_seed]
                if len(vals) == 1:
                    row[label] = f"{vals[0]:.4f}"
                else:
                    row[label] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(summary: pd.DataFrame) -> None:
    metric_labels = [label for _, label in METRICS]
    col_w = 20
    header = (f"{'Dataset':<16} {'Model':<28} {'Seeds':<16}"
              + "".join(f"{m:>{col_w}}" for m in metric_labels))
    print(header)
    print("-" * len(header))

    prev_dataset = None
    for _, row in summary.iterrows():
        ds = row["dataset"] if row["dataset"] != prev_dataset else ""
        prev_dataset = row["dataset"]
        line = f"{ds:<16} {row['model']:<28} {str(row['seeds']):<16}"
        for label in metric_labels:
            line += f"{row.get(label, '—'):>{col_w}}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Report veracity mean ± std from prediction JSON files."
    )
    parser.add_argument("--results_dir", default="experiment_results",
                        help="Root directory containing per-dataset subdirs (default: experiment_results)")
    parser.add_argument("--data_dir", default="data/cleaned_datasets",
                        help="Root data directory with {dataset}/dev.pkl (default: data/cleaned_datasets)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+",
                        help="Keep only models whose name contains one of these strings")
    args = parser.parse_args()

    predictions = parse_prediction_files(args.results_dir, args.datasets)

    gold = {}
    for dataset in args.datasets:
        gold[dataset] = load_gold_labels(dataset, args.data_dir)
        print(f"{dataset}: {len(gold[dataset])} gold labels")

    if args.models:
        for dataset in predictions:
            predictions[dataset] = {
                m: v for m, v in predictions[dataset].items()
                if any(pat in m for pat in args.models)
            }

    summary = aggregate(predictions, gold)
    print()
    print_table(summary)
    return summary


if __name__ == "__main__":
    main()
