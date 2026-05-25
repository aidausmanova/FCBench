"""
80/20 train/eval splits for averitec, scifact, climatefever, climatecheck.
- Splits are done on the existing train file only (dev/test copied as-is).
- climatefever and climatecheck are split at claim_id level to prevent leakage.
- Random seed is fixed for reproducibility.
"""
import json
import os
import shutil
import random
import pandas as pd

SEED = 42
DATA_IN = "FCBench/data"
DATA_OUT = "FCBench/data_new"


def split_80_20(items, seed=SEED):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    return shuffled[:cut], shuffled[cut:]


def make_dir(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


# ── 1. averitec ──────────────────────────────────────────────────────────────
print("Processing averitec ...")
src = os.path.join(DATA_IN, "averitec")
dst = make_dir(DATA_OUT, "averitec")

train_data = json.load(open(os.path.join(src, "train.json")))
train_new, eval_new = split_80_20(train_data)

json.dump(train_new, open(os.path.join(dst, "train.json"), "w"), indent=2)
json.dump(eval_new,  open(os.path.join(dst, "eval.json"),  "w"), indent=2)
shutil.copy(os.path.join(src, "dev.json"),  os.path.join(dst, "dev.json"))
shutil.copy(os.path.join(src, "test.json"), os.path.join(dst, "test.json"))

print(f"  train: {len(train_data)} → new train: {len(train_new)}, eval: {len(eval_new)}")


# ── 2. SciFact ───────────────────────────────────────────────────────────────
print("Processing SciFact ...")
src = os.path.join(DATA_IN, "SciFact")
dst = make_dir(DATA_OUT, "SciFact")

train_data = [json.loads(l) for l in open(os.path.join(src, "claims_train.jsonl"))]
train_new, eval_new = split_80_20(train_data)

with open(os.path.join(dst, "claims_train.jsonl"), "w") as f:
    for r in train_new:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(dst, "claims_eval.jsonl"), "w") as f:
    for r in eval_new:
        f.write(json.dumps(r) + "\n")
shutil.copy(os.path.join(src, "claims_dev.jsonl"),  os.path.join(dst, "claims_dev.jsonl"))
shutil.copy(os.path.join(src, "claims_test.jsonl"), os.path.join(dst, "claims_test.jsonl"))

# Verify no id overlap between train and eval
train_ids = {r["id"] for r in train_new}
eval_ids  = {r["id"] for r in eval_new}
assert train_ids.isdisjoint(eval_ids), "ID leakage in SciFact!"
print(f"  train: {len(train_data)} → new train: {len(train_new)}, eval: {len(eval_new)} (no id overlap ✓)")


# ── 3. climatefever ──────────────────────────────────────────────────────────
print("Processing climatefever ...")
src = os.path.join(DATA_IN, "climatefever")
dst = make_dir(DATA_OUT, "climatefever")

train_data = json.load(open(os.path.join(src, "train.json")))

# Split at claim_id level to prevent leakage (5 evidence rows per claim)
unique_claim_ids = list({r["claim_id"] for r in train_data})
train_ids, eval_ids = split_80_20(unique_claim_ids)
train_ids_set = set(train_ids)
eval_ids_set  = set(eval_ids)

train_new = [r for r in train_data if r["claim_id"] in train_ids_set]
eval_new  = [r for r in train_data if r["claim_id"] in eval_ids_set]

assert not train_ids_set.intersection(eval_ids_set), "claim_id leakage in climatefever!"

json.dump(train_new, open(os.path.join(dst, "train.json"), "w"), indent=2)
json.dump(eval_new,  open(os.path.join(dst, "eval.json"),  "w"), indent=2)
shutil.copy(os.path.join(src, "dev.json"),  os.path.join(dst, "dev.json"))
shutil.copy(os.path.join(src, "test.json"), os.path.join(dst, "test.json"))

print(f"  train: {len(train_data)} rows / {len(unique_claim_ids)} claims → "
      f"new train: {len(train_new)} rows / {len(train_ids)} claims, "
      f"eval: {len(eval_new)} rows / {len(eval_ids)} claims (no claim_id overlap ✓)")


# ── 4. climatecheck ──────────────────────────────────────────────────────────
print("Processing climatecheck ...")
src = os.path.join(DATA_IN, "cleaned_datasets", "climatecheck")
dst = make_dir(DATA_OUT, "cleaned_datasets", "climatecheck")

train_df = pd.read_parquet(os.path.join(src, "train.pkl"))

# Split at claim_id level to prevent leakage
unique_claim_ids = train_df["claim_id"].unique().tolist()
train_ids, eval_ids = split_80_20(unique_claim_ids)
train_ids_set = set(train_ids)
eval_ids_set  = set(eval_ids)

train_new = train_df[train_df["claim_id"].isin(train_ids_set)]
eval_new  = train_df[train_df["claim_id"].isin(eval_ids_set)]

assert not train_ids_set.intersection(eval_ids_set), "claim_id leakage in climatecheck!"

train_new.to_parquet(os.path.join(dst, "train.pkl"), index=False)
eval_new.to_parquet( os.path.join(dst, "eval.pkl"),  index=False)
shutil.copy(os.path.join(src, "dev.pkl"),  os.path.join(dst, "dev.pkl"))
shutil.copy(os.path.join(src, "test.pkl"), os.path.join(dst, "test.pkl"))

print(f"  train: {len(train_df)} rows / {len(unique_claim_ids)} claims → "
      f"new train: {len(train_new)} rows / {len(train_ids)} claims, "
      f"eval: {len(eval_new)} rows / {len(eval_ids)} claims (no claim_id overlap ✓)")

print("\nDone. Output written to", DATA_OUT)
