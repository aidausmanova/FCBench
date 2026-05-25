import os
import re
import json
import nltk
import time
import torch
import argparse
import pandas as pd
from tqdm import *
from collections import Counter
from datasets import load_dataset, Dataset
from multiprocessing import Pool
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


def detect_language_and_gibberish(examples, batch_size=64):
    texts = examples['clean_text']
    lang_preds = pipe_language(texts, batch_size=batch_size)
    gib_preds  = pipe_gibberish(texts, batch_size=batch_size)
    return {
        'language':  [p['label'] for p in lang_preds],
        'gibberish': [p['label'] for p in gib_preds],
    }

def run_detection_fast(texts, batch_size=512):
    languages, gibberishes = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Detecting"):
        batch = texts[i:i+batch_size]
        lang_preds = pipe_language(batch, batch_size=batch_size, truncation=True)
        gib_preds  = pipe_gibberish(batch, batch_size=batch_size, truncation=True)
        languages.extend(p['label'] for p in lang_preds)
        gibberishes.extend(p['label'] for p in gib_preds)
    return languages, gibberishes

def download_nltk_data(package_name, download_dir='nltk_data'):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(download_dir)
    
    try:
        # Try to find the resource
        nltk.data.find(f'tokenizers/{package_name}')
        print(f"Package '{package_name}' is already downloaded")
    except LookupError:
        # If resource isn't found, download it
        print(f"Downloading {package_name}...")
        nltk.download(package_name, download_dir=download_dir)
        print(f"Successfully downloaded {package_name}")

def remove_duplicates(sentences, urls):
    df = pd.DataFrame({"document_in_sentences":sentences, "sentence_urls":urls})
    df['sentences'] = df['document_in_sentences'].str.strip().str.lower()
    df = df.drop_duplicates(subset="sentences").reset_index()
    return df['document_in_sentences'].tolist(), df['sentence_urls'].tolist()

def get_token_count(content, chars_per_token=4.0):
    content = re.sub(r'\s{2,}', ' ', content)
    content = content.strip()
    if not content:
        return 0
    token_count = int(len(content) / chars_per_token)
    return token_count

def make_paragraphs_for_url(sentences, target_tokens_per_paragraph=500):    
    paragraphs = []
    current_paragraph_sentences = []
    current_token_count = 0
    sentence_index = 0
    
    while sentence_index < len(sentences):
        sentence = sentences[sentence_index]
        s_tokens = get_token_count(sentence)
        
        # Add sentence to current paragraph if it's the first sentence
        if not current_paragraph_sentences:
            current_paragraph_sentences.append(sentence)
            current_token_count += s_tokens                      
            # print(f"Starting PARAGRAPH with S{sentence_index}; Token Count: {current_token_count}")
            sentence_index += 1
        else:
            # If adding the sentence doesn't increase the global paragraph token target
            if current_token_count + s_tokens <= target_tokens_per_paragraph:
                current_paragraph_sentences.append(sentence)
                current_token_count += s_tokens                       
                # print(f"Adding S{sentence_index}; Token Count: {current_token_count}")
                sentence_index += 1
            else:
                # Add sentence if it doesn't cause the paragraph to become too big!
                if current_token_count + s_tokens <= 1.25 * target_tokens_per_paragraph:
                    current_paragraph_sentences.append(sentence)
                    current_token_count += s_tokens
                    # print(f"Within acceptable range, Adding S{sentence_index}; Token Count: {current_token_count}")
                    sentence_index += 1

                # Wrap up the paragraph
                paragraph = " ".join(current_paragraph_sentences)
                if paragraph:
                    paragraphs.append(paragraph)
                        
                # Start a new paragraph
                current_paragraph_sentences = []
                current_token_count = 0
                    
    if current_paragraph_sentences:
        paragraph = " ".join(current_paragraph_sentences)
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs

def get_averitec_paragraphs_parallel(file_path, pool, avg_tokens_per_sentence=42, target_sentences_per_para=12):
    evidence_docs = []
    urls = []

    # target_tokens = avg_tokens_per_sentence * target_sentences_per_para
    # print(f"Target tokens per paragraph: {avg_tokens_per_sentence} x {target_sentences_per_para}")

    all_sentences_lists = []
    all_urls = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for evidence_idx, line in enumerate(f):
            json_obj = json.loads(line)
            if not json_obj:
                continue

            url = json_obj.get("url", f"fallback_url_{evidence_idx}") 
            sentences = json_obj["url2text"]  
            if not sentences:
                continue

            all_sentences_lists.append(sentences)
            all_urls.append(url)

    # make_paragraphs_partial = partial(make_paragraphs_for_url, target_tokens_per_paragraph=target_tokens)
    # results = pool.map(make_paragraphs_partial, all_sentences_lists)
    results = pool.map(make_paragraphs_for_url, all_sentences_lists)
    for url, paragraphs_for_url in zip(all_urls, results):
        evidence_docs.extend(paragraphs_for_url)
        urls.extend([url] * len(paragraphs_for_url))

    return evidence_docs, urls

def _load_claim_file(args):
    index_id, claim_object, knowledge_path = args
    claim_evidences = []
    try:
        claim_evidence_file = os.path.join(knowledge_path, f"{claim_object.get('claim_id', index_id)}.json")
        with open(claim_evidence_file, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                if not json_obj:
                    continue
                claim_evidences.append({
                    "claim_id": json_obj.get("claim_id"),
                    "url":      json_obj.get("url"),
                    "text":     " ".join(json_obj.get("url2text", []))
                })
    except Exception as e:
        print(f"ERROR processing claim {index_id}: {e}")
    return claim_evidences

def averitec_to_list(claims_path, knowledge_path):
    with open(claims_path, "r", encoding="utf-8") as fp:
        claims_dataset = json.load(fp)

    CHUNK = 50   # process 50 claim files at a time
    all_rows = []
    seen = set()

    for start in tqdm(range(0, len(claims_dataset), CHUNK)):
        chunk_claims = claims_dataset[start:start + CHUNK]
        chunk_args = [(start + i, c, knowledge_store_path) for i, c in enumerate(chunk_claims)]

        with ThreadPoolExecutor(max_workers=4) as ex:
            evidence_list = []
            for result in ex.map(_load_claim_file, chunk_args):
                evidence_list.extend(result)

        if not evidence_list:
            continue
    return evidence_list

def fever_to_list(knowledge_path):
    all_evidences = []

    for index_id in tqdm(range(1, 110)):
        wiki_index = f"wiki-{str(index_id).zfill(3)}"

        claim_evidences = []
        try:
            claim_evidence_file = os.path.join(knowledge_path, f"{wiki_index}.jsonl")
            with open(claim_evidence_file, 'r', encoding='utf-8') as f:
                # df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
                for line in f:
                    json_obj = json.loads(line)
                    if not json_obj:
                        continue
                    claim_evidences.append({
                        "doc_id": json_obj.get("id"),
                        "text":   json_obj.get("text")
                    })
            all_evidences.extend(claim_evidences)
        except Exception as e:
            print(f"ERROR processing claim {wiki_index}: {e}")
    return all_evidences

def feverous_to_list(knowledge_path):
    all_evidences = []
    
    for index_id in tqdm(range(0, 611)):
        wiki_index = f"wiki_{str(index_id).zfill(3)}"

        claim_evidences = []

        # output_path = os.path.join(knowledge_path, f"clean/feverous/{index_id}.json")
        # with open(output_path, 'w', encoding='utf-8') as out_f:
        try:
            claim_evidence_file = os.path.join(knowledge_path, f"feverous/{wiki_index}.jsonl")
            with open(claim_evidence_file, 'r', encoding='utf-8') as f:
                # df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
                for line in f:
                    json_obj = json.loads(line)
                    if not json_obj:
                        continue
                    sentences = []
                    for item in json_obj.get("order", []):
                        if item.startswith("sentence"):
                            sentences.append(json_obj.get(item))
                    record = {
                        "id":       index_id,
                        "title":    json_obj.get("title"),
                        "url2text": sentences,
                        "text":     " ".join(sentences)
                    }
                    # out_f.write(json.dumps(record) + '\n')
                    claim_evidences.append(record)
            all_evidences.extend(claim_evidences)
        except Exception as e:
            print(f"ERROR processing claim {wiki_index}: {e}")
    return all_evidences

def scifact_to_list(knowledge_path):
    all_evidences = []
    with open(knowledge_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            if not json_obj:
                continue
            all_evidences.append({
                "doc_id": json_obj.get("doc_id"),
                "title":  json_obj.get("title"),
                "text":   " ".join(json_obj.get("abstract", []))
            })

    return all_evidences

def climatecheck_to_list(knowledge_path):
    all_evidences = []
    with open(knowledge_path, "r") as f:
        corpus = json.load(f)

    for ab_id_k, ab_id_v in corpus['abstract_id'].items():
        all_evidences.append(
            {
                "doc_id": ab_id_v,
                "title": corpus['title_lowered'][ab_id_k],
                "text": corpus['abstract_lowered'][ab_id_k]
            }
        )

    return all_evidences

def climatefever_to_list():
    raw_dataset = load_dataset('tdiggelm/climate_fever', trust_remote_code=True)
    data = raw_dataset['test']

    # labels: 0: "supports", 1: "refutes", 2: "not enough info" and 3: "disputed"

    all_evidences = []
    with open("knowledge_store/climatefever/corpus.json", "w") as out_f:
        for item in data:
            for evidence in item["evidences"]:
                record = {
                    "doc_id": evidence["evidence_id"],
                    "title": evidence["article"],
                    "text": evidence["evidence"],
                    "entropy": evidence["entropy"],
                    "votes": evidence["votes"],
                }
                all_evidences.append(record)
                out_f.write(json.dumps(record) + '\n')

    return all_evidences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()

    dataset = args.dataset

    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')

    pipe_language = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection",
                         truncation=True, device="cuda", torch_dtype=torch.float16)
    pipe_gibberish = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457",
                          truncation=True, device="cuda", torch_dtype=torch.float16)


    if dataset == "averitec":
        claims_path = "FCBench/data/averitec/dev.json"
        knowledge_store_path = "FCBench/knowledge_store/averitec/"
        evidence_list = averitec_to_list(claims_path, knowledge_store_path)

    elif dataset == "scifact":
        knowledge_store_path = "FCBench/knowledge_store/scifact/corpus.jsonl"
        evidence_list = scifact_to_list(knowledge_store_path)

    elif dataset == "climatecheck":
        knowledge_store_path = "FCBench/knowledge_store/climatecheck/corpus.json"
        evidence_list = climatecheck_to_list(knowledge_store_path)

    elif dataset == "climatefever":
        evidence_list = climatefever_to_list()

    df = pd.DataFrame(evidence_list)
    # df['token_count'] = df['text'].apply(get_token_count)
    df['token_count'] = (df['text'].str.strip().str.replace(r'\s{2,}', ' ', regex=True).str.len() / 4).astype(int)
    df['text_norm'] = df['text'].str.strip().str.lower()
    n_dupes = df.duplicated(subset="text_norm").sum()
    df = df.drop_duplicates(subset="text_norm").reset_index()

    sentences = df['text'].tolist()
    evidence_dataset = Dataset.from_dict({"clean_text": sentences})
    # evidence_dataset = evidence_dataset.map(detect_language, batched=True, batch_size=64)
    # evidence_dataset = evidence_dataset.map(detect_gibberish, batched=True, batch_size=64)
    # evidence_dataset = evidence_dataset.map(detect_language_and_gibberish, batched=True, batch_size=64)
    df['language'], df['gibberish'] = run_detection_fast(sentences, batch_size=512)

    print(df['token_count'].describe())
    print(f"[INFO ]Total tokens: {df['token_count'].sum():,}")
    print(f"[INFO] Duplicates: {n_dupes:,} ({n_dupes/len(df)*100:.1f}%)")
    print(f"Language counts:\n{Counter(evidence_dataset['language']).most_common()}\n")
    print(f"Gibberish counts:\n{Counter(evidence_dataset['gibberish']).most_common()}\n")
        