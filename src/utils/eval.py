"""
Retrieval evaluation metrics for fact-checking benchmarks.

Each function operates on document IDs (e.g. evidence_id strings).
retrieved_ids: ordered list of retrieved document IDs (rank 1 first)
gold_ids:      collection of relevant/gold document IDs for a claim
n:             cutoff rank
"""

from typing import Sequence, Collection
from pathlib import Path


def recall_at_n(
    retrieved_ids: Sequence[str],
    gold_ids: Collection[str],
    n: int,
) -> float:
    """Fraction of gold documents found in the top-n retrieved documents.

    Returns 0.0 when gold_ids is empty.
    """
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    top_n = set(retrieved_ids[:n])
    return len(top_n & gold_set) / len(gold_set)


def precision_at_n(
    retrieved_ids: Sequence[str],
    gold_ids: Collection[str],
    n: int,
) -> float:
    """Fraction of top-n retrieved documents that are gold documents.

    Returns 0.0 when the retrieved list is empty or n == 0.
    """
    if not retrieved_ids or n == 0:
        return 0.0
    gold_set = set(gold_ids)
    top_n = retrieved_ids[:n]
    return sum(1 for doc_id in top_n if doc_id in gold_set) / len(top_n)


def f1_at_n(
    retrieved_ids: Sequence[str],
    gold_ids: Collection[str],
    n: int,
) -> float:
    """Harmonic mean of Precision@n and Recall@n.

    Returns 0.0 when both precision and recall are 0.
    """
    p = precision_at_n(retrieved_ids, gold_ids, n)
    r = recall_at_n(retrieved_ids, gold_ids, n)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def macro_f1(
    all_retrieved: Sequence[Sequence[str]],
    all_gold: Sequence[Collection[str]],
    n: int,
) -> float:
    """Mean F1@n across all claims (macro average).

    Args:
        all_retrieved: One ordered list of retrieved IDs per claim.
        all_gold:      One collection of gold IDs per claim.
        n:             Rank cutoff applied to each claim.

    Returns:
        Macro-averaged F1@n. Returns 0.0 for an empty claim set.
    """
    if not all_retrieved:
        return 0.0
    scores = [
        f1_at_n(retrieved, gold, n)
        for retrieved, gold in zip(all_retrieved, all_gold)
    ]
    return sum(scores) / len(scores)



# ---------------------------------------------------------------------------
# Gold-data loaders
# ---------------------------------------------------------------------------

def _load_gold_climatefever(data_path: str) -> dict:
    """Return {claim_id: set of gold evidence_ids} from climatefever dev split."""
    import json
    from collections import defaultdict
    with open(data_path) as f:
        rows = json.load(f)
    gold: dict = defaultdict(set)
    for row in rows:
        gold[str(row["claim_id"])].add(row["evidence_id"])
    return dict(gold)


def _load_gold_scifact(data_path: str) -> dict:
    """Return {claim_id: set of gold doc_ids (strings)} from SciFact dev split."""
    import json
    gold = {}
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            gold[str(row["id"])] = {str(d) for d in row["cited_doc_ids"]}
    return gold


def _load_climatecheck_baseline_predictions(csv_path: str) -> dict:
    """Return retrieval dict from ClimateCheckBaselinePredictions.csv.

    Format: {claim_id: {"evidences": [{"evidence_id": str, "rank": int}, ...]}}
    Evidences are ordered by rank (ascending).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.sort_values(["claim_id", "rank"])
    retrieval: dict = {}
    for claim_id, group in df.groupby("claim_id"):
        retrieval[str(claim_id)] = {
            "evidences": [
                {"evidence_id": str(row["abstract_id"]), "rank": int(row["rank"])}
                for _, row in group.iterrows()
            ]
        }
    return retrieval


def _load_gold_climatecheck(data_path: str) -> dict:
    """Return {claim_id: set of gold abstract_ids (strings)} from climatecheck dev split."""
    import pandas as pd
    df = pd.read_parquet(data_path)
    df = df[df["abstract_id"] != -1]
    gold: dict = {}
    for claim_id, group in df.groupby("claim_id"):
        gold[str(claim_id)] = {str(aid) for aid in group["abstract_id"]}
    return gold


def _load_gold_averitec(data_path: str) -> dict:
    """Return {claim_id: set of gold source_urls} from AVeriTeC dev split.

    Claim IDs are zero-based indices (matching experiment_results keys).
    Gold evidence URLs are collected from all questions[].answers[].source_url.
    """
    import json
    with open(data_path) as f:
        claims = json.load(f)
    gold = {}
    for idx, claim in enumerate(claims):
        urls = set()
        for question in claim["questions"]:
            for answer in question["answers"]:
                url = answer.get("source_url") or answer.get("cached_source_url", "")
                if url:
                    urls.add(url)
        if urls:
            gold[str(idx)] = urls
    return gold


# ---------------------------------------------------------------------------
# Gold-data loaders with text and labels (for Ev2R)
# ---------------------------------------------------------------------------

# Map normalized labels → ClimateCheck label space used by the proxy scorer
_TO_CC_LABEL = {
    "Supported":                           "Supports",
    "Refuted":                             "Refutes",
    "Not Enough Evidence":                 "Not Enough Information",
    "Conflicting Evidence/Cherrypicking":  "Not Enough Information",
}


def _load_gold_climatefever_ev2r(data_path: str) -> dict:
    """Return {claim_id: {"claim": str, "texts": [str], "labels": [str]}} from climatefever dev.

    Sentence-level gold evidences are grouped per claim. Label is derived from
    claim_label (integer) via the dataset's LABEL_MAPS entry.
    """
    import json
    from collections import defaultdict

    _CLAIM_LABEL_MAP = {0: "Supported", 1: "Refuted", 2: "Not Enough Evidence",
                        3: "Conflicting Evidence/Cherrypicking"}

    with open(data_path) as f:
        rows = json.load(f)

    grouped: dict = defaultdict(lambda: {"claim": "", "texts": [], "labels": [], "_seen": set()})
    for row in rows:
        cid = str(row["claim_id"])
        eid = row["evidence_id"]
        if eid in grouped[cid]["_seen"]:
            continue
        grouped[cid]["_seen"].add(eid)
        grouped[cid]["claim"] = row["claim"]
        grouped[cid]["texts"].append(row["evidence"])
        normalized = _CLAIM_LABEL_MAP.get(row["claim_label"], "Not Enough Evidence")
        grouped[cid]["labels"].append(_TO_CC_LABEL.get(normalized, "Not Enough Information"))

    return {cid: {"claim": v["claim"], "texts": v["texts"], "labels": v["labels"]}
            for cid, v in grouped.items()}


def _load_gold_scifact_ev2r(data_path: str, corpus_path: str) -> dict:
    """Return {claim_id: {"claim": str, "texts": [str], "labels": [str]}} from SciFact dev.

    Abstract text is looked up in corpus.jsonl. Label comes from the evidence dict
    (SUPPORT / CONTRADICT); cited docs without an annotation entry are treated as NEI.
    """
    import json

    _SCIFACT_LABEL_MAP = {"SUPPORT": "Supports", "CONTRADICT": "Refutes"}

    # Build corpus lookup: doc_id (str) → full abstract text
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            corpus[str(obj["doc_id"])] = " ".join(obj.get("abstract", []))

    gold = {}
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            cid = str(row["id"])
            texts, labels = [], []
            for doc_id in row["cited_doc_ids"]:
                sid = str(doc_id)
                text = corpus.get(sid, "")
                if not text:
                    continue
                doc_evidence = row["evidence"].get(sid, row["evidence"].get(str(doc_id), []))
                if doc_evidence:
                    raw_label = doc_evidence[0]["label"]
                    label = _SCIFACT_LABEL_MAP.get(raw_label, "Not Enough Information")
                else:
                    label = "Not Enough Information"
                texts.append(text)
                labels.append(label)
            if texts:
                gold[cid] = {"claim": row["claim"], "texts": texts, "labels": labels}
    return gold


def _load_gold_climatecheck_ev2r(data_path: str) -> dict:
    """Return {claim_id: {"claim": str, "texts": [str], "labels": [str]}} from climatecheck dev."""
    import pandas as pd

    _CC_LABEL_MAP = {"Supports": "Supports", "Refutes": "Refutes",
                     "Not Enough Information": "Not Enough Information"}

    df = pd.read_parquet(data_path)
    df = df[(df["abstract_id"] != -1) & (df["abstract"].str.strip() != "")]

    gold = {}
    for claim_id, group in df.groupby("claim_id"):
        cid = str(claim_id)
        texts = group["abstract"].tolist()
        labels = [_CC_LABEL_MAP.get(str(lbl).strip(), "Not Enough Information")
                  for lbl in group["label"]]
        claim_text = group["text"].iloc[0]
        gold[cid] = {"claim": claim_text, "texts": texts, "labels": labels}
    return gold


# ---------------------------------------------------------------------------
# Ev2R evaluation helper
# ---------------------------------------------------------------------------

def evaluate_ev2r(
    retrieval: dict,
    gold_ev2r: dict,
    scorer,
    n: int,
) -> dict:
    """Compute mean Ev2R@n for one method on one dataset.

    Args:
        retrieval:  {claim_id: {"claim": str, "evidences": [{"evidence_id": ..., "text": ...}]}}
        gold_ev2r:  {claim_id: {"claim": str, "texts": [str], "labels": [str]}}
        scorer:     ClimateCheckEv2RScorer instance
        n:          rank cutoff — only top-n retrieved texts are scored

    Returns:
        {"mean_ev2r": float, "n_claims": int}
    """
    common_ids = [cid for cid in retrieval if cid in gold_ev2r
                  and gold_ev2r[cid]["texts"]]

    scores = []
    for cid in common_ids:
        claim_text = retrieval[cid].get("claim") or gold_ev2r[cid]["claim"]
        retrieved_texts = [e["text"] for e in retrieval[cid]["evidences"][:n]
                           if e.get("text")]
        gold_entry = gold_ev2r[cid]

        # Swap gold labels into scorer for this claim
        scorer.gold_labels = gold_entry["labels"]

        result = scorer.score(
            claim=claim_text,
            retrieved_abstracts=retrieved_texts,
            gold_abstracts=gold_entry["texts"],
        )
        scores.append(result["Ev2R"])

    mean = sum(scores) / len(scores) if scores else 0.0
    return {"mean_ev2r": mean, "n_claims": len(common_ids)}


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retrieval: dict,
    gold: dict,
    ns: list,
) -> dict:
    """
    Compute retrieval metrics for one method on one dataset.

    Args:
        retrieval: {claim_id: {"evidences": [{"evidence_id": ...}, ...]}}
        gold:      {claim_id: set of gold evidence_ids}
        ns:        list of cutoff ranks to evaluate

    Returns:
        {n: {"recall": float, "precision": float, "f1": float, "macro_f1": float}}
    """
    # Only evaluate claims that appear in both retrieval and gold
    common_ids = [cid for cid in retrieval if cid in gold]

    if not common_ids:
        return {n: {"recall": 0.0, "precision": 0.0, "f1": 0.0, "n_claims": 0} for n in ns}

    all_retrieved = [
        [e["evidence_id"] for e in retrieval[cid]["evidences"]]
        for cid in common_ids
    ]
    all_gold = [gold[cid] for cid in common_ids]

    results = {}
    for n in ns:
        recalls    = [recall_at_n(r, g, n)    for r, g in zip(all_retrieved, all_gold)]
        precisions = [precision_at_n(r, g, n) for r, g in zip(all_retrieved, all_gold)]
        f1s        = [f1_at_n(r, g, n)        for r, g in zip(all_retrieved, all_gold)]
        results[n] = {
            "recall":    sum(recalls)    / len(recalls),
            "precision": sum(precisions) / len(precisions),
            "f1":        sum(f1s)        / len(f1s),
            # "macro_f1":  macro_f1(all_retrieved, all_gold, n),
            "n_claims":  len(common_ids),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Evaluate FCBench retrieval results (lexical metrics + optional Ev2R)."
    )
    parser.add_argument("--gemini_key", default=None,
                        help="Gemini API key. Required for Ev2R reference-based scoring.")
    parser.add_argument("--gemini_model", default="gemini-2.0-flash",
                        help="Gemini model name for the reference-based scorer.")
    parser.add_argument("--proxy_model",
                        default="rausch/deberta-climatecheck-2463191-step26000",
                        help="HuggingFace model for the proxy scorer.")
    parser.add_argument("--ev2r_n", type=int, default=5,
                        help="Rank cutoff for Ev2R evaluation (default: 5).")
    parser.add_argument("--cache_db", default="experiment_results/cache/ev2r_cache.db",
                        help="SQLite cache path for Ev2R results.")
    args = parser.parse_args()

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    DATASETS = {
        # "climatefever": {
        #     "results_dir": os.path.join(BASE, "experiment_results", "climatefever"),
        #     "gold_loader": lambda: _load_gold_climatefever(
        #         os.path.join(BASE, "data", "climatefever", "dev.json")
        #     ),
        #     "ev2r_loader": lambda: _load_gold_climatefever_ev2r(
        #         os.path.join(BASE, "data", "climatefever", "dev.json")
        #     ),
        # },
        # "scifact": {
        #     "results_dir": os.path.join(BASE, "experiment_results", "scifact"),
        #     "gold_loader": lambda: _load_gold_scifact(
        #         os.path.join(BASE, "data", "SciFact", "claims_dev.jsonl")
        #     ),
        #     "ev2r_loader": lambda: _load_gold_scifact_ev2r(
        #         os.path.join(BASE, "data", "SciFact", "claims_dev.jsonl"),
        #         os.path.join(BASE, "knowledge_store", "scifact", "corpus.jsonl"),
        #     ),
        # },
        "climatecheck": {
            "results_dir": os.path.join(BASE, "experiment_results", "climatecheck"),
            "gold_loader": lambda: _load_gold_climatecheck(
                os.path.join(BASE, "data", "cleaned_datasets", "climatecheck", "dev.pkl")
            ),
            "ev2r_loader": None,
        },
        "climatecheck_test": {
            "results_dir": None,
            "gold_loader": lambda: _load_gold_climatecheck(
                os.path.join(BASE, "data", "cleaned_datasets", "climatecheck", "test.pkl")
            ),
            "ev2r_loader": None,
            "extra_retrievals": {
                "baseline": lambda: _load_climatecheck_baseline_predictions(
                    os.path.join(BASE, "data", "ClimateCheckBaselinePredictions.csv")
                ),
            },
        },
        "averitec": {
            "results_dir": os.path.join(BASE, "experiment_results", "averitec"),
            "gold_loader": lambda: _load_gold_averitec(
                os.path.join(BASE, "data", "averitec", "dev.json")
            ),
            "ev2r_loader": None,
        },
    }

    METHODS = ["bm25", "tfidf", "random"]
    NS = [5, 10, 20]

    # ------------------------------------------------------------------
    # Optionally build Ev2R scorers
    # ------------------------------------------------------------------
    ev2r_scorer = None
    if args.gemini_key:
        auto_eval_dir = os.path.join(os.path.dirname(__file__), "..", "automatic_eval")
        sys.path.insert(0, os.path.abspath(auto_eval_dir))

        from google import genai
        from reference_based import Ev2RReferenceBasedScorer
        from proxy_based import Ev2RProxyScorer
        from final_score import ClimateCheckEv2RScorer
        from cache import Ev2RCache

        gemini_client = genai.Client(api_key=args.gemini_key)
        reference_scorer = Ev2RReferenceBasedScorer(
            gemini_client=gemini_client,
            prompt_path=Path(os.path.join(auto_eval_dir, "reference_based_prompt.txt")),
            gemini_model=args.gemini_model,
        )
        proxy_scorer = Ev2RProxyScorer(
            model_name_or_path=args.proxy_model,
            label2id={"Supports": 0, "Refutes": 1, "Not Enough Information": 2},
        )
        cache = Ev2RCache(args.cache_db)
        # gold_labels is set per-claim inside evaluate_ev2r; use [] as placeholder
        ev2r_scorer = ClimateCheckEv2RScorer(
            cache=cache,
            reference_scorer=reference_scorer,
            proxy_scorer=proxy_scorer,
            gold_labels=[],
        )

    # ------------------------------------------------------------------
    # Print lexical metrics table
    # ------------------------------------------------------------------
    col_w = 10
    metric_cols = []
    for n in NS:
        metric_cols += [f"R@{n}", f"P@{n}", f"F1@{n}"]
    ev2r_col = f"Ev2R@{args.ev2r_n}" if ev2r_scorer else ""
    header = (f"{'Dataset':<14} {'Method':<8}"
              + "".join(f"{c:>{col_w}}" for c in metric_cols)
              + (f"{ev2r_col:>{col_w}}" if ev2r_col else ""))
    print(header)
    print("-" * len(header))

    for dataset_name, cfg in DATASETS.items():
        gold = cfg["gold_loader"]()
        gold_ev2r = cfg["ev2r_loader"]() if ev2r_scorer else None
        first = True

        method_retrievals = []
        if cfg["results_dir"]:
            for method in METHODS:
                ret_path = os.path.join(cfg["results_dir"], f"{method}_retrieval.json")
                if os.path.exists(ret_path):
                    with open(ret_path) as f:
                        method_retrievals.append((method, json.load(f)))
        for method, retrieval in cfg.get("extra_retrievals", {}).items():
            method_retrievals.append((method, retrieval()))

        for method, retrieval in method_retrievals:
            metrics = evaluate_retrieval(retrieval, gold, NS)

            ev2r_val = None
            if ev2r_scorer and gold_ev2r:
                ev2r_result = evaluate_ev2r(retrieval, gold_ev2r, ev2r_scorer, n=args.ev2r_n)
                ev2r_val = ev2r_result["mean_ev2r"]

            ds_label = dataset_name if first else ""
            first = False
            row = f"{ds_label:<14} {method:<8}"
            for n in NS:
                m = metrics[n]
                row += (
                    f"{m['recall']:>{col_w}.4f}"
                    f"{m['precision']:>{col_w}.4f}"
                    f"{m['f1']:>{col_w}.4f}"
                    # f"{m['macro_f1']:>{col_w}.4f}"
                )
            if ev2r_scorer:
                row += f"{ev2r_val:>{col_w}.4f}" if ev2r_val is not None else f"{'N/A':>{col_w}}"
            print(row)

        print()