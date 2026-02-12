import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def train_baselines(X_train, y_train, X_test, y_test, dataset_name, logger, seed):
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train and y_train should have the same number of samples.")

    if X_test.shape[0] != len(y_test):
        raise ValueError("X_test and y_test should have the same number of samples.")

    pipelines = {
        "tfidf + LogReg": make_pipeline(TfidfVectorizer(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)),
        "random": make_pipeline(DummyClassifier(strategy="uniform")),
        "majority": make_pipeline(DummyClassifier(strategy="most_frequent"))
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
