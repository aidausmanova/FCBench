"""
Claim veracity prediction using Logistic Regression and fine-tuned Longformer.

Training features: claim + gold evidence texts from the training split.
Test features:     claim + TF-IDF-retrieved evidence texts from the dev split.

Feature representation: concatenation of the claim and the top-n retrieved/gold
evidence passages separated by " [SEP] ".
"""

import json
import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.model.baseline import train_baselines
from src.model.distilbert import train_distilRoBERTa
from src.model.longformer import train_longformer
from src.utils.logger import Logger

# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

_CLIMATEFEVER_LABEL_MAP = {
    0: "Supported",
    1: "Refuted",
    2: "Not Enough Evidence",
    3: "Conflicting Evidence/Cherrypicking",
}

_SCIFACT_LABEL_MAP = {
    "SUPPORT":   "Supported",
    "CONTRADICT": "Refuted",
}


# ---------------------------------------------------------------------------
# Feature string builder
# ---------------------------------------------------------------------------

def _build_feature_str(claim: str, evidence_texts: List[str], n: int = 5) -> str:
    """Concatenate claim and top-n evidence passages into a single string."""
    parts = [claim] + evidence_texts[:n]
    return " [SEP] ".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Training-split loaders  (gold evidence → feature strings)
# ---------------------------------------------------------------------------

def _load_train_averitec(data_path: str, n: int = 5) -> Tuple[List[str], List[str]]:
    """Load AVeriTeC training split.

    Evidence text is built from question–answer pairs in the gold annotation.
    Returns (features, labels).
    """
    with open(data_path) as f:
        claims = json.load(f)

    features, labels = [], []
    for claim in claims:
        evidence_texts = []
        for q in claim["questions"]:
            for a in q["answers"]:
                text = a.get("answer", "").strip()
                if text:
                    evidence_texts.append(text)
        features.append(_build_feature_str(claim["claim"], evidence_texts, n))
        labels.append(claim["label"])
    return features, labels


def _load_train_climatefever(data_path: str, n: int = 5) -> Tuple[List[str], List[str]]:
    """Load ClimateFever training split.

    Multiple rows per claim; evidence texts and label are grouped by claim_id.
    Returns (features, labels).
    """
    from collections import defaultdict

    with open(data_path) as f:
        rows = json.load(f)

    grouped = defaultdict(lambda: {"claim": "", "evidences": [], "label": None})
    for row in rows:
        cid = row["claim_id"]
        grouped[cid]["claim"] = row["claim"]
        grouped[cid]["evidences"].append(row["evidence"])
        grouped[cid]["label"] = _CLIMATEFEVER_LABEL_MAP.get(row["claim_label"],
                                                              "Not Enough Evidence")

    features, labels = [], []
    for entry in grouped.values():
        features.append(_build_feature_str(entry["claim"], entry["evidences"], n))
        labels.append(entry["label"])
    return features, labels


def _load_train_scifact(data_path: str, corpus_path: str,
                         n: int = 5) -> Tuple[List[str], List[str]]:
    """Load SciFact training split.

    Abstract text is fetched from corpus.jsonl by cited_doc_ids.
    Claim-level label: CONTRADICT > SUPPORT > NEI (if evidence dict is empty).
    Returns (features, labels).
    """
    corpus: dict = {}
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            corpus[str(obj["doc_id"])] = " ".join(obj.get("abstract", []))

    features, labels = [], []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            evidence_texts = [
                corpus[str(did)]
                for did in row["cited_doc_ids"]
                if str(did) in corpus
            ]
            # Derive claim-level label from evidence dict
            raw_labels = set()
            for entries in row["evidence"].values():
                for e in entries:
                    raw_labels.add(e["label"])
            if "CONTRADICT" in raw_labels:
                label = "Refuted"
            elif "SUPPORT" in raw_labels:
                label = "Supported"
            else:
                label = "Not Enough Information"

            features.append(_build_feature_str(row["claim"], evidence_texts, n))
            labels.append(label)
    return features, labels


def _load_train_climatecheck(data_path: str, n: int = 5) -> Tuple[List[str], List[str]]:
    """Load ClimateCheck training split.

    Each unique claim_id gets one feature string built from its gold abstract.
    Returns (features, labels).
    """
    df = pd.read_parquet(data_path)
    df = df[df["label"].notna() & (df["label"] != "")]

    features, labels = [], []
    for _, group in df.groupby("claim_id"):
        claim_text = group["text"].iloc[0]
        evidence_texts = group["abstract"].dropna().tolist()
        label = group["label"].iloc[0]
        features.append(_build_feature_str(claim_text, evidence_texts, n))
        labels.append(label)
    return features, labels


# ---------------------------------------------------------------------------
# Dev-split loader using TF-IDF retrieval output
# ---------------------------------------------------------------------------

def _load_dev_retrieval(
    retrieval_path: str,
    gold_labels: dict,
    n: int = 5,
) -> Tuple[List[str], List[str], List[dict]]:
    """Build dev features from TF-IDF retrieval results.

    Args:
        retrieval_path: path to {dataset}/tfidf_retrieval.json
        gold_labels:    {claim_id: label_str} for the dev split
        n:              number of retrieved passages to include

    Returns:
        (features, labels, retrieved_docs) where retrieved_docs is a list of
        {"claim": str, "evidences": [{"id": str, "text": str}, ...]} dicts,
        one per claim, with evidence_id mapped to "id" for clarity.
    """
    with open(retrieval_path) as f:
        retrieval = json.load(f)

    features, labels, retrieved_docs = [], [], []
    for cid, entry in retrieval.items():
        if cid not in gold_labels:
            continue
        evidences = entry.get("evidences", [])[:n]
        evidence_texts = [e["text"] for e in evidences if e.get("text")]
        features.append(_build_feature_str(entry["claim"], evidence_texts, n))
        labels.append(gold_labels[cid])
        retrieved_docs.append({
            "claim_id": cid,
            "claim": entry["claim"],
            "evidences": [
                {"id": e.get("evidence_id", ""), "text": e.get("text", "")}
                for e in evidences
            ],
        })
    return features, labels, retrieved_docs


# ---------------------------------------------------------------------------
# Gold-label loaders for the dev split
# ---------------------------------------------------------------------------

def _dev_labels_averitec(data_path: str) -> dict:
    """Return {claim_id (str index): label} for AVeriTeC dev."""
    with open(data_path) as f:
        claims = json.load(f)
    return {str(i): c["label"] for i, c in enumerate(claims)}


def _dev_labels_climatefever(data_path: str) -> dict:
    """Return {claim_id: label} for ClimateFever dev (one entry per claim)."""
    from collections import defaultdict
    with open(data_path) as f:
        rows = json.load(f)
    labels: dict = {}
    for row in rows:
        cid = str(row["claim_id"])
        if cid not in labels:
            labels[cid] = _CLIMATEFEVER_LABEL_MAP.get(row["claim_label"],
                                                        "Not Enough Evidence")
    return labels


def _dev_labels_scifact(data_path: str) -> dict:
    """Return {claim_id: label} for SciFact dev."""
    labels = {}
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            raw = set()
            for entries in row["evidence"].values():
                for e in entries:
                    raw.add(e["label"])
            if "CONTRADICT" in raw:
                label = "Refuted"
            elif "SUPPORT" in raw:
                label = "Supported"
            else:
                label = "Not Enough Information"
            labels[str(row["id"])] = label
    return labels


def _dev_labels_climatecheck(data_path: str) -> dict:
    """Return {claim_id: label} for ClimateCheck dev."""
    df = pd.read_parquet(data_path)
    df = df[df["label"].notna() & (df["label"] != "")]
    return {str(cid): grp["label"].iloc[0]
            for cid, grp in df.groupby("claim_id")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train veracity classifiers using TF-IDF retrieval output as "
                    "test-time evidence. Supports Logistic Regression and Longformer."
    )
    parser.add_argument("--n_evidence", type=int, default=5,
                        help="Number of retrieved passages to use as features (default: 5).")
    parser.add_argument("--seed", type=int, default=42)
    # Longformer-specific arguments
    parser.add_argument("--longformer", action="store_true",
                        help="Also fine-tune Longformer in addition to the baseline.")
    parser.add_argument("--distilbert", action="store_true",
                        help="Also fine-tune DistilRoBERTa in addition to the baseline.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size for transformer training.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps for transformer training.")
    parser.add_argument("--hub_token", default="",
                        help="HuggingFace token for push_to_hub (required by transformer trainers).")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Fraction of training data to use as validation (default: 0.1).")
    args = parser.parse_args()

    if args.longformer or args.distilbert:
        os.environ.setdefault("HUB_TOKEN", args.hub_token)

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    N = args.n_evidence

    DATASETS = {
        "averitec": {
            "train_loader": lambda: _load_train_averitec(
                os.path.join(BASE, "data", "averitec", "train.json"), N),
            "dev_labels":   lambda: _dev_labels_averitec(
                os.path.join(BASE, "data", "averitec", "dev.json")),
            "retrieval":    os.path.join(BASE, "experiment_results",
                                          "averitec", "tfidf_retrieval.json"),
        },
        "climatefever": {
            "train_loader": lambda: _load_train_climatefever(
                os.path.join(BASE, "data", "climatefever", "train.json"), N),
            "dev_labels":   lambda: _dev_labels_climatefever(
                os.path.join(BASE, "data", "climatefever", "dev.json")),
            "retrieval":    os.path.join(BASE, "experiment_results",
                                          "climatefever", "tfidf_retrieval.json"),
        },
        "scifact": {
            "train_loader": lambda: _load_train_scifact(
                os.path.join(BASE, "data", "SciFact", "claims_train.jsonl"),
                os.path.join(BASE, "knowledge_store", "scifact", "corpus.jsonl"), N),
            "dev_labels":   lambda: _dev_labels_scifact(
                os.path.join(BASE, "data", "SciFact", "claims_dev.jsonl")),
            "retrieval":    os.path.join(BASE, "experiment_results",
                                          "scifact", "tfidf_retrieval.json"),
        },
        "climatecheck": {
            "train_loader": lambda: _load_train_climatecheck(
                os.path.join(BASE, "data", "cleaned_datasets",
                              "climatecheck", "train.pkl"), N),
            "dev_labels":   lambda: _dev_labels_climatecheck(
                os.path.join(BASE, "data", "cleaned_datasets",
                              "climatecheck", "dev.pkl")),
            "retrieval":    os.path.join(BASE, "experiment_results",
                                          "climatecheck", "tfidf_retrieval.json"),
        },
    }

    for d in ["model_save", "model_save/longformer", "model_save/distilbert",
              "experiment_results/performances",
              "experiment_results/performances/y_pred",
              "experiment_results/cartography",
              "experiment_results/cartography/distilRoBERTa",
              "experiment_results/distilRoBERTa"]:
        os.makedirs(os.path.join(BASE, d), exist_ok=True)
    os.chdir(BASE)  # all model saves use paths relative to cwd

    logger = Logger("retrieval_veracity", reset=True)
    logger.set_seed(args.seed)

    for dataset_name, cfg in DATASETS.items():
        print(f"\nLoading {dataset_name}...")
        X_train_full, y_train_full = cfg["train_loader"]()
        dev_labels = cfg["dev_labels"]()
        X_test, y_test, retrieved_docs = _load_dev_retrieval(cfg["retrieval"], dev_labels, n=N)

        if not X_test:
            print(f"  No dev claims found for {dataset_name}, skipping.")
            continue

        # Split training data into train / validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=y_train_full,
        )
        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # --- Logistic Regression baseline ---
        train_baselines(
            X_train=pd.Series(X_train),
            y_train=pd.Series(y_train),
            X_test=pd.Series(X_test),
            y_test=pd.Series(y_test),
            dataset_name=dataset_name,
            logger=logger,
            seed=args.seed,
            retrieved_docs=retrieved_docs,
        )

        # --- Longformer ---
        if args.longformer:
            print(f"\n  Fine-tuning Longformer on {dataset_name}...")
            train_longformer(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                model_save_path=f"model_save/longformer/{dataset_name}",
                logging_dir=f"model_save/longformer/{dataset_name}/logs",
                dataset_name=dataset_name,
                logger=logger,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                seed=args.seed,
                weighted_training=True,
            )

        # --- DistilRoBERTa ---
        if args.distilbert:
            print(f"\n  Fine-tuning DistilRoBERTa on {dataset_name}...")
            train_distilRoBERTa(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                model_save_path=f"model_save/distilbert/{dataset_name}",
                logging_dir=f"model_save/distilbert/{dataset_name}/logs",
                dataset_name=dataset_name,
                logger=logger,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                seed=args.seed,
                weighted_training=True,
            )

        logger.save()

    # Summary table  (micro F1 == accuracy for single-label classification)
    if not logger.performances.empty:
        model_types = ["tfidf + LogReg"]
        if args.longformer:
            model_types.append("longformer")
        if args.distilbert:
            model_types.append("distilRoBERTa")

        col_w = 12
        header = (f"{'Dataset':<16} {'Model':<18}"
                  f"{'Macro P':>{col_w}}"
                  f"{'Macro R':>{col_w}}"
                  f"{'Macro F1':>{col_w}}"
                  f"{'Micro F1':>{col_w}}"
                  f"{'Weighted F1':>{col_w}}")
        print(f"\n{'='*len(header)}")
        print(header)
        print("-" * len(header))
        for model_type in model_types:
            rows = logger.performances[
                logger.performances["model_type"] == model_type
            ]
            for _, row in rows.iterrows():
                print(f"{row['dataset_name']:<16} {model_type:<18}"
                      f"{row['precision']:>{col_w}.4f}"
                      f"{row['recall']:>{col_w}.4f}"
                      f"{row['performance']:>{col_w}.4f}"
                      f"{row['accuracy']:>{col_w}.4f}"
                      f"{row['weighted_f1']:>{col_w}.4f}")
