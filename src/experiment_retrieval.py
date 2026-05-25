"""
experiment_retrieval.py — Veracity prediction with TF-IDF-retrieved evidence.

Output
------
  experiment_results/{dataset}/tfidf_{model}_seed{seed}_veracity_prediction.json

  CUDA_VISIBLE_DEVICES=0 python3 -m src.experiment_retrieval \\
      --log logs/retrieval/all --reset -l -d \\
      --dataset_list averitec scifact climatecheck climatefever \\
      --seed_list 42 43 27 \\
      --batch_size 4 --accumulation_steps 8
"""

import argparse
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as _tts
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils.logger import Logger

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASETS = ["averitec", "scifact", "climatecheck", "climatefever"]
DEFAULT_SEEDS = [42, 43, 27]

_CLIMATEFEVER_LABEL_MAP = {
    0: "Supported",
    1: "Refuted",
    2: "Not Enough Evidence",
    3: "Conflicting Evidence/Cherrypicking",
}

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int, is_longformer: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Longformer uses non-deterministic CUDA ops; relax strict mode for it
    torch.use_deterministic_algorithms(not is_longformer)


def is_main_process() -> bool:
    """True on rank-0 under torchrun/DDP, always True for single-process runs."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _scifact_label(evidence: dict) -> str:
    """Derive claim-level veracity label from SciFact's per-doc evidence dict."""
    if not evidence:
        return "Not Enough Evidence"
    labels = set()
    for doc in evidence.values():
        if doc is None:
            continue
        for entry in doc:
            labels.add(entry["label"])
    if "CONTRADICT" in labels:
        return "Refuted"
    if "SUPPORT" in labels:
        return "Supported"
    return "Not Enough Evidence"


def load_splits(dataset_name: str, seed: int):
    """Return (train_df, dev_df) each with columns [text, label, claim_id].

    train_df : training split — claim text only (no retrieved evidence).
    dev_df   : dev split — claim_id values align with tfidf_retrieval.json keys.
               Deduplicated to one row per unique claim.
    """
    folder = os.path.join("data", "cleaned_datasets", dataset_name)

    if dataset_name == "averitec":
        train_full = pd.read_parquet(os.path.join(folder, "train.pkl"))
        dev = pd.read_parquet(os.path.join(folder, "dev.pkl"))
        train_full = train_full[["text", "label"]].copy()
        dev = dev[["text", "label"]].copy()
        # tfidf_retrieval.json uses sequential row index "0","1",... as keys
        dev["claim_id"] = [str(i) for i in range(len(dev))]
        train, _ = _tts(
            train_full, test_size=0.2, random_state=seed, stratify=train_full["label"]
        )
        train = train.reset_index(drop=True)
        train["claim_id"] = train.index.astype(str)
        return train, dev.reset_index(drop=True)

    if dataset_name == "scifact":
        train_full = pd.read_parquet(os.path.join(folder, "train.pkl"))
        dev = pd.read_parquet(os.path.join(folder, "dev.pkl"))
        train_full["label"] = train_full["evidence"].apply(_scifact_label)
        dev["label"] = dev["evidence"].apply(_scifact_label)
        train_full["claim_id"] = train_full["id"].astype(str)
        dev["claim_id"] = dev["id"].astype(str)
        train_full = train_full[["text", "label", "claim_id"]].copy()
        dev = dev[["text", "label", "claim_id"]].copy()
        train, _ = _tts(
            train_full, test_size=0.2, random_state=seed, stratify=train_full["label"]
        )
        return train.reset_index(drop=True), dev.reset_index(drop=True)

    if dataset_name == "climatecheck":
        train_full = pd.read_parquet(os.path.join(folder, "train.pkl"))
        dev = pd.read_parquet(os.path.join(folder, "dev.pkl"))
        for df in (train_full, dev):
            df.dropna(subset=["label"], inplace=True)
            df.drop(index=df[df["label"] == ""].index, inplace=True)
        train_full["claim_id"] = train_full["claim_id"].astype(str)
        dev["claim_id"] = dev["claim_id"].astype(str)
        train_full = train_full[["text", "label", "claim_id"]].copy()
        dev = dev[["text", "label", "claim_id"]].copy()
        # Multiple rows can share the same claim_id — keep one per claim
        train_full.drop_duplicates(subset="claim_id", keep="first", inplace=True)
        dev.drop_duplicates(subset="claim_id", keep="first", inplace=True)
        train, _ = _tts(
            train_full, test_size=0.2, random_state=seed, stratify=train_full["label"]
        )
        return train.reset_index(drop=True), dev.reset_index(drop=True)

    if dataset_name == "climatefever":
        train = pd.read_parquet(os.path.join(folder, "train.pkl"))
        dev = pd.read_parquet(os.path.join(folder, "dev.pkl"))
        for df in (train, dev):
            df["label"] = df["claim_label"].map(_CLIMATEFEVER_LABEL_MAP)
            df.drop_duplicates(subset="claim_id", keep="first", inplace=True)
        train["claim_id"] = train["claim_id"].astype(str)
        dev["claim_id"] = dev["claim_id"].astype(str)
        train = train[["text", "label", "claim_id"]].copy().reset_index(drop=True)
        dev = dev[["text", "label", "claim_id"]].copy().reset_index(drop=True)
        return train, dev

    raise ValueError(f"Unknown dataset: {dataset_name!r}")


def load_retrieval(dataset_name: str) -> dict:
    """Load experiment_results/{dataset}/tfidf_retrieval.json."""
    path = os.path.join("experiment_results", dataset_name, "tfidf_retrieval.json")
    with open(path) as fh:
        return json.load(fh)


def build_input_text(claim: str, evidences: list) -> str:
    """Concatenate claim with retrieved passages: 'claim [SEP] ev1 ev2 ...'
    Falls back to claim-only when no evidence is available for that claim."""
    ev_parts = [e["text"] for e in evidences if e.get("text", "").strip()]
    return (claim + " [SEP] " + " ".join(ev_parts)) if ev_parts else claim


def save_predictions(
    dataset_name: str,
    model_key: str,
    seed: int,
    dev_df: pd.DataFrame,
    retrieval: dict,
    y_pred,
) -> None:
    """Write per-claim predictions to
      experiment_results/{dataset}/tfidf_{model}_seed{seed}_veracity_prediction.json
    """
    records = []
    for (_, row), pred in zip(dev_df.iterrows(), y_pred):
        cid = str(row["claim_id"])
        records.append(
            {
                "claim_id": cid,
                "claim": row["text"],
                "evidence": retrieval.get(cid, {}).get("evidences", []),
                "pred_label": str(pred),
            }
        )

    out_dir = os.path.join("experiment_results", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"tfidf_{model_key}_seed{seed}_veracity_prediction.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"    → {len(records)} predictions saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Transformer training + evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f1}


def run_transformer(
    hf_model_id: str,
    max_length: int,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    retrieval: dict,
    dataset_name: str,
    model_key: str,
    seed: int,
    logger: Logger,
    batch_size: int,
    accumulation_steps: int,
) -> None:
    """Fine-tune a transformer on training claims; evaluate on retrieval-augmented dev claims.

    Training input  : raw claim text only.
    Validation input: claim + tfidf-retrieved evidence  (early stopping signal).
    Test input      : claim + tfidf-retrieved evidence  (final evaluation).
    """
    # ── Build texts ──────────────────────────────────────────────────────────
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()

    X_dev = []
    for _, row in dev_df.iterrows():
        cid = str(row["claim_id"])
        evidences = retrieval.get(cid, {}).get("evidences", [])
        X_dev.append(build_input_text(row["text"], evidences))
    y_dev = dev_df["label"].tolist()

    # ── Label mappings ────────────────────────────────────────────────────────
    unique_labels = sorted(set(y_train))
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    y_train_ids = [label2id[y] for y in y_train]
    y_dev_ids = [label2id[y] for y in y_dev]

    # ── Tokenise ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_dict(
        {"text": X_train, "label": y_train_ids}
    ).map(tokenize, batched=True)

    dev_ds = Dataset.from_dict(
        {"text": X_dev, "label": y_dev_ids}
    ).map(tokenize, batched=True)

    # ── Training ──────────────────────────────────────────────────────────────
    save_dir = os.path.join("model_save", f"{dataset_name}_{model_key}_seed{seed}")

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=500,
        fp16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        seed=seed,
        disable_tqdm=True,
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            hf_model_id,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,       # retrieval-augmented → early stopping on target distribution
        compute_metrics=_compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    trainer.train()

    # ── Inference on retrieval-augmented dev set ───────────────────────────────
    output = trainer.predict(dev_ds)
    y_pred_ids = output.predictions.argmax(axis=-1)
    y_pred = [id2label[i] for i in y_pred_ids]

    if is_main_process():
        # Log macro-F1 to the performance CSV
        logger.add_precomputed_f1_score(
            y_pred=y_pred,
            y_test=y_dev,
            dataset_name=dataset_name,
            model_type=f"tfidf_{model_key}",
            n_labels=len(unique_labels),
        )
        # Save per-claim predictions with evidence to JSON
        save_predictions(dataset_name, model_key, seed, dev_df, retrieval, y_pred)

    # Clean up checkpoint to free disk space
    if is_main_process() and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Veracity prediction with TF-IDF-retrieved evidence"
    )
    parser.add_argument(
        "--log", required=True,
        help="Path stem for the CSV performance log (no .csv extension). "
             "Example: logs/retrieval/longformer",
    )
    parser.add_argument("--seed_list", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--dataset_list", nargs="+", type=str, default=DEFAULT_DATASETS
    )
    parser.add_argument("--reset", action="store_true", help="Overwrite existing log")
    parser.add_argument("-l", "--longformer", action="store_true",
                        help="Run tfidf + Longformer")
    parser.add_argument("-d", "--distilroberta", action="store_true",
                        help="Run tfidf + DistilROBERTa")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    args = parser.parse_args()

    if not args.longformer and not args.distilroberta:
        parser.error("Specify at least one model: -l (Longformer) or -d (DistilROBERTa)")

    logger = Logger(log_filename=args.log, reset=args.reset)

    for seed in args.seed_list:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")
        logger.set_seed(seed)
        set_seed(seed, is_longformer=args.longformer)

        for dataset_name in args.dataset_list:
            print(f"\n  [Dataset: {dataset_name}]")
            train_df, dev_df = load_splits(dataset_name, seed)
            retrieval = load_retrieval(dataset_name)
            print(
                f"  train={len(train_df)}  dev={len(dev_df)}"
                f"  retrieval_keys={len(retrieval)}"
            )

            if args.longformer:
                print("  Running: tfidf + Longformer")
                run_transformer(
                    hf_model_id="allenai/longformer-base-4096",
                    max_length=4096,
                    train_df=train_df,
                    dev_df=dev_df,
                    retrieval=retrieval,
                    dataset_name=dataset_name,
                    model_key="longformer",
                    seed=seed,
                    logger=logger,
                    batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps,
                )

            if args.distilroberta:
                print("  Running: tfidf + DistilROBERTa")
                run_transformer(
                    hf_model_id="distilbert/distilroberta-base",
                    max_length=512,
                    train_df=train_df,
                    dev_df=dev_df,
                    retrieval=retrieval,
                    dataset_name=dataset_name,
                    model_key="distilroberta",
                    seed=seed,
                    logger=logger,
                    batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps,
                )

            if is_main_process():
                logger.save()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
