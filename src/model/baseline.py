import json
import os

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


class BM25Transformer(BaseEstimator, TransformerMixin):
    """Convert a term-count matrix (from CountVectorizer) to BM25-weighted features.

    Implements the Okapi BM25 weighting scheme as a drop-in sklearn transformer,
    so it can be used in a pipeline:
        CountVectorizer() -> BM25Transformer() -> LogisticRegression()

    Parameters
    ----------
    k1 : float
        Controls term-frequency saturation (default 1.5).
    b : float
        Controls document-length normalisation (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, X, y=None):
        X = sp.csr_matrix(X, dtype=np.float64)
        n_docs, n_terms = X.shape

        # Document frequency per term
        df = np.diff(X.tocsc().indptr)  # number of docs each term appears in
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)  [BM25+ variant, always > 0]
        self.idf_ = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Average document length (in tokens)
        self.avgdl_ = X.sum(axis=1).mean()
        return self

    def transform(self, X, y=None):
        X = sp.csr_matrix(X, dtype=np.float64)
        # Document lengths
        dl = np.asarray(X.sum(axis=1)).ravel()
        # Length-normalisation factor per document: (1 - b + b * dl / avgdl)
        norm = 1.0 - self.b + self.b * (dl / self.avgdl_)

        # Apply BM25 numerator / denominator element-wise in sparse format
        # score = idf * tf * (k1 + 1) / (tf + k1 * norm)
        X = X.copy()
        rows, cols = X.nonzero()
        tf = np.asarray(X[rows, cols]).ravel()
        denom = tf + self.k1 * norm[rows]
        X.data = self.idf_[cols] * tf * (self.k1 + 1.0) / denom
        return X


def _save_tfidf_retrieval(retrieved_docs, dataset_name):
    """Save the TF-IDF-retrieved documents (with their IDs) for each test claim.

    Args:
        retrieved_docs: list of dicts, one per test claim:
            {"claim": str, "evidences": [{"id": str, "text": str}, ...]}
            The "id" key holds the dataset-specific identifier:
              - averitec:    URL
              - scifact:     doc_id
              - climatefever: doc_id
              - climatecheck: abstract_id
        dataset_name: used to name the output file.

    Output: experiment_results/tfidf_retrieved/{dataset_name}.json
    """
    out_dir = os.path.join("experiment_results", "tfidf_retrieved")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}.json")

    with open(out_path, "w") as f:
        json.dump(retrieved_docs, f, indent=2)
    print(f"Saved TF-IDF retrieval results to {out_path}")


def train_baselines(X_train, y_train, X_test, y_test, dataset_name, logger, seed,
                    retrieved_docs=None):
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train and y_train should have the same number of samples.")

    if X_test.shape[0] != len(y_test):
        raise ValueError("X_test and y_test should have the same number of samples.")

    pipelines = {
        "tfidf + LogReg": make_pipeline(TfidfVectorizer(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)),
        "bm25 + LogReg":  make_pipeline(CountVectorizer(), BM25Transformer(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)),
        "random": make_pipeline(DummyClassifier(strategy="uniform")),
        # "majority": make_pipeline(DummyClassifier(strategy="most_frequent"))
    }

    for model in pipelines.keys():
        pipe = pipelines[model]
        pipe.fit(X=X_train, y=y_train)

        logger.add_f1_score(
            pipe=pipe,
            X_test=X_test,
            y_test=y_test,
            dataset_name=dataset_name,
            model_type=model,
            n_labels=len(set(y_train))
        )

        if model == "tfidf + LogReg" and retrieved_docs is not None:
            _save_tfidf_retrieval(retrieved_docs, dataset_name)

        filename = f"model_save/{dataset_name}_{model.replace(' ', '_')}.joblib"
        joblib.dump(pipe, filename)
        print(f"Saved pipeline '{model}' as {filename}.\n")


def train_baselines_multilabel(X_train, y_train, X_test, y_test, dataset_name, logger, seed):
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train and y_train should have the same number of samples.")

    if X_test.shape[0] != len(y_test):
        raise ValueError("X_test and y_test should have the same number of samples.")

    pipelines = {
        "tfidf + LogReg": make_pipeline(TfidfVectorizer(), MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed))),
        "random": make_pipeline(DummyClassifier(strategy="uniform", random_state=seed)),
        "majority": make_pipeline(DummyClassifier(strategy="most_frequent", random_state=seed))
    }

    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    for model in pipelines.keys():
        pipe = pipelines[model]
        pipe.fit(X=X_train, y=y_train_bin)

        logger.add_f1_score(
            pipe=pipe,
            X_test=X_test,
            y_test=y_test_bin,
            dataset_name=dataset_name,
            model_type=model,
            n_labels=len(mlb.classes_),
        )

        filename = f"model_save/{dataset_name}_{model.replace(' ', '_')}.joblib"
        joblib.dump(pipe, filename)
        print(f"Saved pipeline '{model}' as {filename}.\n")

def train_baselines_query_onehot(X_train, y_train, X_test, y_test, dataset_name, logger, seed):
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'text'),
            ('query', OneHotEncoder(), ['query'])
        ]
    )

    pipelines = {
        "tfidf + LogReg": make_pipeline(preprocessor, LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)),
        "random": make_pipeline(DummyClassifier(strategy="uniform", random_state=seed)),
        "majority": make_pipeline(DummyClassifier(strategy="most_frequent", random_state=seed))
    }

    for model in pipelines.keys():
        pipe = pipelines[model]
        pipe.fit(X=X_train, y=y_train)

        logger.add_f1_score(
            pipe=pipe,
            X_test=X_test,
            y_test=y_test.reset_index(drop=True),
            dataset_name=dataset_name,
            model_type=model,
            n_labels=len(set(y_train)),
        )

        filename = f"model_save/{dataset_name}_{model.replace(' ', '_')}.joblib"
        joblib.dump(pipe, filename)
        print(f"Saved pipeline '{model}' as {filename}.\n")

def train_baselines_relation(X_train, y_train, X_test, y_test, dataset_name, logger, seed):
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'text'),
            ('query', TfidfVectorizer(), 'query')
        ]
    )

    pipelines = {
        "tfidf + LogReg": make_pipeline(preprocessor, LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)),
        "random": make_pipeline(DummyClassifier(strategy="uniform", random_state=seed)),
        "majority": make_pipeline(DummyClassifier(strategy="most_frequent", random_state=seed))
    }

    for model in pipelines.keys():
        pipe = pipelines[model]
        pipe.fit(X=X_train, y=y_train)

        logger.add_f1_score(
            pipe=pipe,
            X_test=X_test,
            y_test=y_test.reset_index(drop=True),
            dataset_name=dataset_name,
            model_type=model,
            n_labels=len(set(y_train)),
        )

        filename = f"model_save/{dataset_name}_{model.replace(' ', '_')}.joblib"
        joblib.dump(pipe, filename)
        print(f"Saved pipeline '{model}' as {filename}.\n")
