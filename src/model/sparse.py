import os
import re
import time
import json
import nltk
import heapq
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from tqdm import *

from ..utils.basic import load_claims, load_shared_corpus
from ..utils.config import TOKENIZATION_WORKERS, MAX_SENTENCES, OVERLAP, TOP_N_PER_QUERY

print(f"TOP_N_PER_QUERY: {TOP_N_PER_QUERY}")
print(f"MAX_SENTENCES: {MAX_SENTENCES}")
print(f"TOKENIZATION_WORKERS: {TOKENIZATION_WORKERS}")

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

def initialize_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Cleared existing directory: {path}")
    os.makedirs(path, exist_ok=True)
    print(f"Directory ready: {path}")

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
    
def chunk_into_paragraphs_parallel(file_path, pool, avg_tokens_per_sentence=42, target_sentences_per_para=12):
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

def tokenize_paragraph(paragraph):
    return nltk.word_tokenize(paragraph)

def tokenize_paragraphs_parallel(paragraphs, pool):
    """Tokenise a list of paragraphs in parallel with an existing Pool."""
    return pool.map(tokenize_paragraph, paragraphs)

def build_bm25_index_parallel(paragraphs, pool):
    tokenized_paras = tokenize_paragraphs_parallel(paragraphs, pool)
    return BM25Okapi(tokenized_paras)

def bm25_retrieval(queries, bm25, pool, top_n_per_query):
    top_bm25_paras = set()    
    tokenized_queries = tokenize_paragraphs_parallel(queries, pool)
    for tokenized_query in tokenized_queries:
        scores = bm25.get_scores(tokenized_query)
        ranked_para_ids = heapq.nlargest(top_n_per_query, range(len(scores)), key=lambda x: x[1]) #key=scores.__getitem__)
        top_bm25_paras.update(ranked_para_ids)
    return top_bm25_paras


def build_tfidf_index(texts):
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def tfidf_retrieval(queries, vectorizer, matrix, top_n_per_query):
    top_ids = set()
    query_vec = vectorizer.transform(queries)
    scores = cosine_similarity(query_vec, matrix)   # (n_queries, n_docs)
    for row in scores:
        ranked = np.argpartition(row, -top_n_per_query)[-top_n_per_query:]
        top_ids.update(ranked.tolist())
    return top_ids

def random_retrieval(n_docs, top_n):
    return set(random.sample(range(n_docs), min(top_n, n_docs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--CLAIMS_PATH', required=True)
    parser.add_argument('-k', '--KNOWLEDGE_STORE_PATH', required=True)
    parser.add_argument('-d', '--dataset_type', default='averitec',
                        choices=['averitec', 'climatecheck', 'climatefever', 'scifact'])
    parser.add_argument('-n', '--top_n', type=int, default=20)
    parser.add_argument('-m', '--method', type=str, default="bm25", choices=['bm25', 'tfidf', 'random'])
    args = parser.parse_args()

    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')

    claims = load_claims(args.CLAIMS_PATH, args.dataset_type)
    print(f"Loaded {len(claims)} claims")

    output_dir = os.path.join("experiment_results", args.dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    results = {}   # { claim_id: {"claim": str, "evidences": [{"evidence_id": str, "text": str}, ...]} }

    def build_evidences_from_ids(top_ids, doc_ids, doc_texts):
        return [{"evidence_id": doc_ids[i], "text": doc_texts[i]} for i in sorted(top_ids)]

    with Pool(TOKENIZATION_WORKERS) as pool:

        if args.dataset_type == "averitec":
            # Per-claim knowledge store: one file per claim
            for claim in tqdm(claims, desc=f"{args.method.upper()} retrieval"):
                claim_id = claim['claim_id']
                query    = claim['claim']
                start    = time.time()
                try:
                    claim_file = os.path.join(args.KNOWLEDGE_STORE_PATH, f"{claim_id}.json")
                    paras, urls = chunk_into_paragraphs_parallel(claim_file, pool,
                                                                 target_sentences_per_para=MAX_SENTENCES)
                    paras, urls = remove_duplicates(paras, urls)
                    if not paras:
                        results[claim_id] = {"claim": query, "evidences": []}
                        continue

                    if args.method == "bm25":
                        index = build_bm25_index_parallel(paras, pool)
                        top_para_ids = bm25_retrieval([query], index, pool, top_n_per_query=args.top_n)
                    elif args.method == "tfidf":
                        vectorizer, matrix = build_tfidf_index(paras)
                        top_para_ids = tfidf_retrieval([query], vectorizer, matrix, top_n_per_query=args.top_n)
                    else:  # random
                        top_para_ids = random_retrieval(len(paras), top_n=args.top_n)

                    # Aggregate paragraphs by URL, preserving text
                    url_paras = defaultdict(list)
                    for pid in sorted(top_para_ids):
                        url_paras[urls[pid]].append(paras[pid])

                    evidences = [
                        {"evidence_id": url, "text": " ".join(url_paras[url])}
                        for url in url_paras
                    ]
                    results[claim_id] = {"claim": query, "evidences": evidences}
                    print(f"[{claim_id}] {len(evidences)} evidence URLs  ({time.time()-start:.1f}s)")
                except Exception as e:
                    print(f"Error for claim {claim_id}: {e}")
                    results[claim_id] = {"claim": claim['claim'], "evidences": []}

        else:
            # Shared corpus: build one index, query for every claim
            print("Loading shared corpus...")
            doc_ids, doc_texts = load_shared_corpus(args.KNOWLEDGE_STORE_PATH, args.dataset_type)
            print(f"Corpus size: {len(doc_ids):,} documents")

            if args.method == "bm25":
                print("Building BM25 index...")
                index = build_bm25_index_parallel(doc_texts, pool)
            elif args.method == "tfidf":
                print("Building TF-IDF index...")
                vectorizer, matrix = build_tfidf_index(doc_texts)

            for claim in tqdm(claims, desc=f"{args.method.upper()} retrieval"):
                claim_id = claim['claim_id']
                query    = claim['claim']
                try:
                    if args.method == "bm25":
                        top_ids = bm25_retrieval([query], index, pool, top_n_per_query=args.top_n)
                    elif args.method == "tfidf":
                        top_ids = tfidf_retrieval([query], vectorizer, matrix, top_n_per_query=args.top_n)
                    else:  # random
                        top_ids = random_retrieval(len(doc_ids), top_n=args.top_n)

                    results[claim_id] = {
                        "claim":    query,
                        "evidences": build_evidences_from_ids(top_ids, doc_ids, doc_texts),
                    }
                except Exception as e:
                    print(f"Error for claim {claim_id}: {e}")
                    results[claim_id] = {"claim": claim['claim'], "evidences": []}

    output_path = os.path.join(output_dir, f'{args.method}_retrieval.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {output_path}")
