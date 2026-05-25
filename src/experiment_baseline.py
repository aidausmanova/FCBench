import os
import ast
import argparse
import datetime
import pandas as pd
import numpy as np
import torch
import random


def is_main_process() -> bool:
    """Return True only on rank-0 when running under torchrun/DDP, always True otherwise."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

from src.utils.builder import DatasetBuilder
from src.model.longformer import train_longformer, train_multi_longformer
from src.model.distilbert import train_distilRoBERTa, train_multi_distilRoBERTa
from src.model.baseline import train_baselines, train_baselines_query_onehot, train_baselines_multilabel, train_baselines_relation
from src.utils.logger import Logger

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\n=== GPU {i}: ===")
        props = torch.cuda.get_device_properties(i)
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available")

start_time = datetime.datetime.now()
print(f"Start time: {start_time}")


def set_seed(seed, lonformer):
    """
    Everything should be reproducible
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # these are just for deterministic behaviour
    torch.backends.cudnn.benchmark = False
    if lonformer:
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_dataset(dataset_name):
    dataset_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", dataset_name)
    train = pd.read_parquet(os.path.join(dataset_path, "train.pkl"))
    test = pd.read_parquet(os.path.join(dataset_path, "test.pkl"))
    dev = pd.read_parquet(os.path.join(dataset_path, "dev.pkl"))

    if dataset_name in ["logicClimate"]:
        train["label"] = train["label"].apply(ast.literal_eval)
        test["label"] = test["label"].apply(ast.literal_eval)
        dev["label"] = dev["label"].apply(ast.literal_eval)

    return train, test, dev

def clean_datasets(train, test, dev, dataset_name, dataset_builder):
    if dataset_name in ["lobbymap_pages", "lobbymap_query", "lobbymap_stance"]:
        # We found that examples with less than 40 tokens are all labelled as 0, and correspond to pages with number or titles, but not actual content. Also we limit to 3900 because it is a query task (so there is the length of the text and the query)
        train = dataset_builder.filter(train, min_token=40, max_token=4000)
        test = dataset_builder.filter(test, min_token=40, max_token=4000)
        dev = dataset_builder.filter(dev, min_token=40, max_token=4000)
    else:
        train = dataset_builder.filter(train)
        test = dataset_builder.filter(test)
        dev = dataset_builder.filter(dev)

    train.drop_duplicates(inplace=True)
    test.drop_duplicates(inplace=True)
    dev.drop_duplicates(inplace=True)

    return train, test, dev

def generate_args(dataset_builder, dataset_list, logger):
    args = dict()

    for dataset_name in dataset_builder.datasets.keys():
        args[dataset_name] = {
            "function": dataset_builder.datasets[dataset_name],
            "training_function": train_baselines,
            "stratify_on": 'label',
            "label_columns": 'label',
            "input_columns": 'text',
            "classification_type": "classification",
            "balanced": "balanced",
            "weighted_loss": False,
        }

    # Keep a subset of datasets
    args = {key: args[key] for key in dataset_list if key in args}

    # Skip already trained datasets
    remove_datasets = logger.get_already_trained_datasets()
    args = {key: value for key, value in args.items() if key not in remove_datasets}

    print("Not done: ***************")
    print(args.keys())

    return args

def experiment_loop(dataset_builder, logger, seed, dataset_list, batch_size, accumulation_steps, dataset_max_size=10000, do_train_baselines=False, do_train_longformer=False, do_train_distilRoBERTa=False):
    args = generate_args(dataset_builder, dataset_list, logger)

    for dataset_name in args.keys():
        print("DATASET:", dataset_name)

        print(args[dataset_name])

        train, test, dev = args[dataset_name]["function"]()

        # Neural models: 80 % train / 20 % holdout val / dev test (no overlap)
        X_train_neural = train[args[dataset_name]["input_columns"]]
        y_train_neural = train[args[dataset_name]['label_columns']]
        X_val = test[args[dataset_name]["input_columns"]]
        y_val = test[args[dataset_name]['label_columns']]

        # Baselines: train on the full corpus (train + holdout merged)
        train = pd.concat([train, test], ignore_index=True)
        X_train = train[args[dataset_name]["input_columns"]]
        y_train = train[args[dataset_name]['label_columns']]

        X_dev = dev[args[dataset_name]["input_columns"]]
        y_dev = dev[args[dataset_name]['label_columns']]

        if do_train_baselines:
            print("Training Baselines:")
            args[dataset_name]["training_function"](
                X_train=X_train,
                y_train=y_train,
                X_test=X_dev,
                y_test=y_dev,
                dataset_name=dataset_name,
                logger=logger,
                seed=seed
            )

        if do_train_longformer:
            print("Training Longformer:")
            func = train_longformer
            if args[dataset_name]["classification_type"]=="multilabel":
                print("multilabel")
                func = train_multi_longformer
            func(
                X_train=X_train_neural,
                y_train=y_train_neural,
                X_val=X_val,
                y_val=y_val,
                X_test=X_dev,
                y_test=y_dev,
                model_save_path="model_save/distilbert/" + dataset_name,
                logging_dir="model_save/distilbert/" + dataset_name,
                dataset_name=dataset_name,
                logger=logger,
                classification_type=args[dataset_name]['classification_type'],
                seed=seed,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                weighted_training=args[dataset_name]["weighted_loss"],
            )

        if do_train_distilRoBERTa:
            print("Training distilRoBERTa")
            func = train_distilRoBERTa
            if args[dataset_name]["classification_type"] == "multilabel":
                print("multilabel")
                func = train_multi_distilRoBERTa
            func(
                X_train=X_train_neural,
                y_train=y_train_neural,
                X_val=X_val,
                y_val=y_val,
                X_test=X_dev,
                y_test=y_dev,
                model_save_path="model_save/distilbert/" + dataset_name,
                logging_dir="model_save/distilbert/" + dataset_name,
                dataset_name=dataset_name,
                logger=logger,
                classification_type=args[dataset_name]['classification_type'],
                seed=seed,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                weighted_training=args[dataset_name]["weighted_loss"],
            )

        if is_main_process():
            logger.save()

        print(f"* * * * * * * * * * * * * < END: {dataset_name} *")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments for greenwashing automatic detection, intermediary tasks")
    parser.add_argument("--log", type=str, help="The name of the output log file", required = True)
    parser.add_argument("--seed_list", nargs='+', type=int, help="List of seeds")
    parser.add_argument("--dataset_list", nargs='+', type=str, help="A list of strings")
    parser.add_argument("--reset", action="store_true", help="Clear existing log file and re-run all datasets")
    parser.add_argument("-b", "--baseline", action="store_true", help="Run the baseline experiments")
    parser.add_argument("-l", "--longformer", action="store_true", help="Run the longformer experiments")
    parser.add_argument("-d", "--distilRoBERTa", action="store_true", help="Run the distilRoBERTa experiments")
    parser.add_argument("--batch_size", type=int, help="The size of the batch used during training")
    parser.add_argument("--accumulation_steps", type=int, help="The gradient accumulation parameter during training")
    args = parser.parse_args()

    if args.longformer:
        if args.batch_size is None:
            parser.error("--batch_size required when running lonformer experiments")
        if args.accumulation_steps is None:
            parser.error("--accumulation_steps required when running lonformer experiments")

    print(f"Running baseline experiments: {args.baseline}")
    print(f"Running longformer experiments: {args.longformer}")

    if not args.seed_list:
        args.seed_list = [42, 26, 123]

    print(f"Running seeds: {args.seed_list}")

    if not args.dataset_list:
        args.dataset_list = [
            'averitec',
            'scifact',
            'climatecheck',
            'climatefever',
        ]

    print(f"Running datasets: {args.dataset_list}")

    set_seed(42, lonformer=args.longformer)

    logger = Logger(log_filename=args.log, reset=args.reset)
    dataset_builder = DatasetBuilder(seed=42)

    for seed in args.seed_list:
        print(f"############ Seed {seed} ############")
        set_seed(seed, lonformer=args.longformer)
        logger.set_seed(seed)
        experiment_loop(
            dataset_builder=dataset_builder,
            logger=logger,
            seed=seed,
            dataset_list=args.dataset_list,
            do_train_baselines=args.baseline,
            do_train_longformer=args.longformer,
            do_train_distilRoBERTa=args.distilRoBERTa,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps
        )
        print(f"############ End of Seed {seed} ############")
    print("END OF TRAINING")

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")
