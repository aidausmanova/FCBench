import os
import sys
import json
import argparse
import datetime
import re
import time

from openai import OpenAI

BASE_URL = 'https://chat-ai.academiccloud.de/v1'
CHAT_AI_KEY = "c65f0464fa6fe9251d15f09b120f3583"

def _make_instructions(labels: list[str], with_evidence: bool) -> str:
    label_descriptions = {
        "Supported": "The claim is fully backed by clear, consistent evidence.",
        "Refuted": "The claim is directly contradicted by reliable evidence, or there is no concrete evidence to support the claim.",
        "Not Enough Evidence": "There is insufficient evidence to support or refute the claim.",
        "Conflicting Evidence/Cherrypicking": "The evidence has both supporting and opposing arguments for the claim or is selectively presented to favor the claim.",
    }
    label_lines = "\n".join(f"{l}: {label_descriptions[l]}" for l in labels)
    label_choices = " | ".join(labels)
    preamble = (
        "Given a claim and retrieved evidence passages, classify the claim using the provided evidence."
        if with_evidence else
        "Given a claim, classify the claim to the best of your knowledge."
    )
    return (
        f"You are a professional fact checker.\n{preamble}\n"
        f"You must assign one of the following labels:\n{label_lines}\n\n"
        f'Output Format (JSON only, no extra text):\n{{"label": "{label_choices}"}}\n'
    )


# Per-dataset label configuration.
# scifact and climatecheck are 3-class; averitec and climatefever are 4-class.
_LABELS_3 = ["Supported", "Refuted", "Not Enough Evidence"]
_LABELS_4 = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]

_DATASET_CONFIG = {
    "averitec":     {"valid_labels": set(_LABELS_4), "label_list": _LABELS_4},
    "scifact":      {"valid_labels": set(_LABELS_3), "label_list": _LABELS_3},
    "climatecheck": {"valid_labels": set(_LABELS_3), "label_list": _LABELS_3},
    "climatefever": {"valid_labels": set(_LABELS_4), "label_list": _LABELS_4},
}

# climatecheck stores labels with slightly different strings than the canonical
# LLM output space.  Normalize y_true before metric computation so label strings match.
_GOLD_LABEL_NORM = {
    "climatecheck": {
        "Supports": "Supported",
        "Refutes": "Refuted",
        "Not Enough Information": "Not Enough Evidence",
    }
}

LLM_MODEL = "meta-llama-3.1-8b-instruct" #"meta-llama-3.1-70b-instruct"

# Derive short model name once: "meta-llama-3.1-70b-instruct" -> "llama-3.1-70b"
_short_name = re.sub(r"^meta-", "", LLM_MODEL)
_short_name = re.sub(r"-instruct$", "", _short_name)


def _pred_filename(retrieval: str) -> str:
    return f"{_short_name}_{retrieval}_veracity_prediction.json"


def load_dataset(dataset_name: str, retrieval: str):
    """Return dev DataFrame with columns [text, label] and optionally [claim_id].

    When retrieval != 'none', imports load_splits from experiment_retrieval so
    that claim_id values align with the retrieval JSON keys.
    """
    if retrieval == "none":
        from src.utils.builder import DatasetBuilder
        builder = DatasetBuilder(seed=42)
        _, _, dev = builder.datasets[dataset_name]()
        return dev  # has [text, label]

    from src.experiment_retrieval import load_splits
    _, dev = load_splits(dataset_name, seed=42)
    return dev  # has [text, label, claim_id]


def load_retrieval_file(dataset_name: str, retrieval: str) -> dict:
    """Load experiment_results/{dataset}/{retrieval}_retrieval.json."""
    path = os.path.join("experiment_results", dataset_name, f"{retrieval}_retrieval.json")
    with open(path) as fh:
        return json.load(fh)


def build_evidence_text(evidences: list, top_k: int) -> str:
    """Return a numbered list of the top-k evidence passages for the prompt."""
    parts = []
    for i, ev in enumerate(evidences[:top_k]):
        text = ev.get("text", "").strip()
        if text:
            parts.append(f"[{i+1}] {text}")
    return "\n".join(parts)


def predict_claim(client, claim: str, evidences: list | None, top_k: int,
                  valid_labels: set, instructions: str,
                  retries: int = 3) -> str:
    has_evidence = evidences is not None and len(evidences) > 0

    if has_evidence:
        ev_text = build_evidence_text(evidences, top_k)
        user_content = f"Evidence passages:\n{ev_text}\n\nClaim: {claim}"
    else:
        user_content = f"Claim: {claim}"

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": user_content},
                ],
                model=LLM_MODEL,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                parsed = json.loads(json_match.group())
                label = parsed.get("label", "").strip()
                if label in valid_labels:
                    return label

            print(f"  [warn] Unexpected response: {raw!r}")
            return "Refuted"

        except Exception as e:
            print(f"  [error] attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return "Refuted"


def _predictions_path(dataset_name: str, retrieval: str) -> str:
    return os.path.join("experiment_results", dataset_name, _pred_filename(retrieval))


def _load_cached_predictions(dataset_name: str, retrieval: str):
    path = _predictions_path(dataset_name, retrieval)
    if os.path.exists(path):
        with open(path) as fh:
            records = json.load(fh)
        claims = [r["claim"] for r in records]
        y_pred = [r["pred_label"] for r in records]
        print(f"  Loaded {len(records)} cached predictions from {path}")
        return claims, y_pred
    return None, None


def _save_predictions(dataset_name: str, retrieval: str, claims, y_pred):
    out_dir = os.path.join("experiment_results", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    path = _predictions_path(dataset_name, retrieval)
    records = [
        {"claim_id": str(i), "claim": claim, "pred_label": pred}
        for i, (claim, pred) in enumerate(zip(claims, y_pred))
    ]
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"  Saved {len(records)} predictions -> {path}")


def run_llm_on_dataset(client, dataset_name: str, logger, retrieval: str, top_k: int,
                       log_results: bool = True):
    print(f"\n=== Dataset: {dataset_name} | retrieval: {retrieval} ===")
    dev = load_dataset(dataset_name, retrieval)
    print(f"  Dev set size: {len(dev)}")

    label_norm = _GOLD_LABEL_NORM.get(dataset_name, {})
    y_true = [label_norm.get(l, l) for l in dev["label"].tolist()]

    cfg = _DATASET_CONFIG[dataset_name]
    valid_labels = cfg["valid_labels"]
    has_retrieval = retrieval != "none"
    instructions = _make_instructions(cfg["label_list"], with_evidence=has_retrieval)
    print(f"  Labels ({len(cfg['label_list'])}): {cfg['label_list']}")

    retrieval_data = None
    if has_retrieval:
        retrieval_data = load_retrieval_file(dataset_name, retrieval)

    claims, y_pred = _load_cached_predictions(dataset_name, retrieval)
    if y_pred is None:
        claims = dev["text"].tolist()
        y_pred = []
        for i, claim in enumerate(claims):
            if i % 50 == 0:
                print(f"  [{i}/{len(claims)}] predicting...")

            evidences = None
            if retrieval_data is not None:
                cid = str(dev["claim_id"].iloc[i]) if "claim_id" in dev.columns else str(i)
                evidences = retrieval_data.get(cid, {}).get("evidences", [])

            y_pred.append(predict_claim(client, claim, evidences, top_k,
                                        valid_labels=valid_labels, instructions=instructions))

        _save_predictions(dataset_name, retrieval, claims, y_pred)

    from sklearn.metrics import classification_report
    from src.utils.logger import bootstrap_confidence_interval
    import numpy as np

    report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0.0, output_dict=True)

    acc   = report["accuracy"]
    m_p   = report["macro avg"]["precision"]
    m_r   = report["macro avg"]["recall"]
    m_f1  = report["macro avg"]["f1-score"]
    w_p   = report["weighted avg"]["precision"]
    w_r   = report["weighted avg"]["recall"]
    w_f1  = report["weighted avg"]["f1-score"]

    print(f"\n  {'Metric':<28} {'Macro':>8}  {'Weighted':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy':<28} {acc:>8.4f}")
    print(f"  {'Precision':<28} {m_p:>8.4f}  {w_p:>10.4f}")
    print(f"  {'Recall':<28} {m_r:>8.4f}  {w_r:>10.4f}")
    print(f"  {'F1':<28} {m_f1:>8.4f}  {w_f1:>10.4f}")

    f1_lower, f1_upper = bootstrap_confidence_interval(y_true=np.array(y_true), y_pred=np.array(y_pred))
    print(f"  {'Macro-F1 95% CI':<28} [{f1_lower:.4f}, {f1_upper:.4f}]")

    if log_results:
        n_labels = len(set(y_true))
        model_tag = f"LLM_{retrieval}"
        logger.add_record(dataset_name, model_tag, report, n_labels, "f1_score", f1_upper, f1_lower)
        logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM veracity prediction (claim-only or retrieval-augmented)")
    parser.add_argument("--log", type=str, required=True, help="Output log file name (without .csv)")
    parser.add_argument("--seed_list", nargs="+", type=int, default=[42])
    parser.add_argument("--dataset_list", nargs="+", type=str,
                        default=["averitec", "scifact", "climatecheck", "climatefever"])
    parser.add_argument("--retrieval", choices=["none", "bm25", "tfidf"], default="none",
                        help="Evidence source: 'none' = claim only, 'bm25' or 'tfidf' = use retrieved passages")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of retrieved evidence passages to include in the prompt (default: 5)")
    parser.add_argument("--reset", action="store_true", help="Clear existing log and re-run all datasets")
    args = parser.parse_args()

    print(f"Start time: {datetime.datetime.now()}")
    print(f"Datasets:   {args.dataset_list}")
    print(f"Seeds:      {args.seed_list}")
    print(f"Model:      {LLM_MODEL}")
    print(f"Retrieval:  {args.retrieval}")
    print(f"Top-K:      {args.top_k}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    client = OpenAI(api_key=CHAT_AI_KEY, base_url=BASE_URL)

    from src.utils.logger import Logger
    logger = Logger(log_filename=args.log, reset=args.reset)

    for seed in args.seed_list:
        print(f"\n############ Seed {seed} ############")
        logger.set_seed(seed)
        already_done = set(logger.get_already_trained_datasets())

        for dataset_name in args.dataset_list:
            if dataset_name in already_done:
                cached_path = _predictions_path(dataset_name, args.retrieval)
                if os.path.exists(cached_path):
                    print(f"Already logged {dataset_name} — showing cached results:")
                    run_llm_on_dataset(client, dataset_name, logger,
                                       retrieval=args.retrieval, top_k=args.top_k,
                                       log_results=False)
                else:
                    print(f"Skipping {dataset_name} (already logged, no prediction cache found)")
                continue
            run_llm_on_dataset(client, dataset_name, logger,
                               retrieval=args.retrieval, top_k=args.top_k)

    print(f"\nDone. End time: {datetime.datetime.now()}")
