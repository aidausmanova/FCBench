import os
import csv
import json
import pickle
import tqdm
import pandas as pd

LABEL_MAPS = {
    "averitec": {
        "Supported": "Supported",
        "Refuted": "Refuted",
        "Conflicting Evidence/Cherrypicking": "Conflicting Evidence/Cherrypicking",
        "Not Enough Evidence": "Not Enough Evidence",
    },
    "climatecheck": {
        "Supports": "Supported",
        "Refutes": "Refuted",
        "Not Enough Information": "Not Enough Evidence",
    },
    "climatefever": {
        0: "Supported",
        1: "Refuted",
        2: "Not Enough Evidence",
        3: "Conflicting Evidence/Cherrypicking",
    },
    "fever": {
        "SUPPORTS": "Supported",
        "REFUTES": "Refuted",
        "NOT ENOUGH INFO": "Not Enough Evidence",
    },
    "feverous": {
        "SUPPORTS": "Supported",
        "REFUTES": "Refuted",
        "NOT ENOUGH INFO": "Not Enough Evidence",
    },
    "scifact": {
        "SUPPORT": "Supported",
        "CONTRADICT": "Refuted",
        "NOT_ENOUGH_INFO": "Not Enough Evidence",
    },
}

def normalize_label(label, dataset_type):
    """Map a dataset-specific label to the internal label space."""
    return LABEL_MAPS.get(dataset_type, {}).get(label, label)

def _load_pkl(path):
    """
    Load a tabular data file (.parquet or .pkl) and return a list of dicts.
    """
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

def load_claims(path, dataset_type):

    if path.endswith((".pkl", ".parquet")):
        rows = _load_pkl(path)
    elif path.endswith(".json") and dataset_type == "averitec":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    claims = []
    if dataset_type == "averitec":
        for i, obj in enumerate(rows):
            claims.append({
                "claim_id": i,
                "claim":    obj["text"],
                "label":    normalize_label(obj.get("label", ""), dataset_type),
            })
    elif dataset_type == "climatecheck":
        for row in rows:
            claims.append({
                "claim_id": int(row["claim_id"]),
                "claim":   row["text"],
                "label":  normalize_label(str(row["label"]).strip(), dataset_type),
                "evidence_id": row.get("abstract_id") or row.get("evidence_id") or row.get("doc_id") or row.get("source_url") or row.get("evidence_wiki_url"),
            })
    elif dataset_type == "scifact":
        for obj in rows:
            claims.append({
                "claim_id": obj["id"],
                "claim":    obj["text"],
                "label":    normalize_label(obj.get("label", ""), dataset_type),
            })
    elif dataset_type == "climatefever":
        for i, obj in enumerate(rows):
            raw_label = obj.get("claim_label", obj.get("label", ""))
            claims.append({
                "claim_id":    obj.get("claim_id", i),
                "claim":       obj["text"],
                "label":       normalize_label(raw_label, dataset_type),
                "evidence_id": obj.get("evidence_id"),
            })
    elif dataset_type in ("fever", "feverous"):
        for obj in rows:
            claims.append({
                "claim_id":          obj.get("id"),
                "claim":             obj["text"],
                "label":             normalize_label(obj.get("label", ""), dataset_type),
                "evidence_wiki_url": obj.get("evidence_wiki_url", ""),
                "evidence_sentence_id": obj.get("evidence_sentence_id", -1),
            })
    return claims

def load_shared_corpus(knowledge_store_path, dataset_type):
    """Load corpus from a directory of jsonl files (fever/feverous) or a single file.
    Returns (doc_ids, doc_texts) lists."""
    doc_ids, doc_texts = [], []

    if dataset_type in ("fever", "feverous"):
        import glob
        pattern = "wiki-*.jsonl" if dataset_type == "fever" else "wiki_*.jsonl"
        files = sorted(glob.glob(os.path.join(knowledge_store_path, pattern)))
        for fpath in tqdm(files, desc="Loading corpus"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if not obj:
                        continue
                    doc_id     = str(obj.get("id", obj.get("title", "")))
                    lines_text = obj.get("lines", "")
                    # Index each sentence separately: evidence_id = "doc_id::sentence_id"
                    for seg in lines_text.split("\n"):
                        if "\t" not in seg:
                            continue
                        sent_id, sent_text = seg.split("\t", 1)
                        sent_text = sent_text.strip()
                        if not sent_text:
                            continue
                        doc_ids.append(f"{doc_id}::{sent_id}")
                        doc_texts.append(sent_text)

    elif dataset_type == "scifact":
        with open(f"{knowledge_store_path}/corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_ids.append(str(obj["doc_id"]))
                doc_texts.append(" ".join(obj.get("abstract", [])))

    elif dataset_type == "climatecheck":
        data = pd.read_json(f"{knowledge_store_path}/corpus.json")
        for _, row in data.iterrows():
            doc_ids.append(str(row["abstract_id"]))
            doc_texts.append(str(row.get("abstract", "") or ""))

    elif dataset_type == "climatefever":
        with open(f"{knowledge_store_path}/corpus.json", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_ids.append(str(obj.get("doc_id", obj.get("id", ""))))
                doc_texts.append(obj.get("text", ""))

    return doc_ids, doc_texts