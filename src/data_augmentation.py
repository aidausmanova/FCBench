import os
import argparse
from datasets import Dataset, DatasetDict
from transformers import pipeline

from src.experiment import generate_args
from src.utils.builder import DatasetBuilder
from src.utils.logger import Logger


def generate_dataset_names(dataset_builder):
    dataset_names = []

    for dataset_name in dataset_builder.datasets.keys():
        dataset_names += [dataset_name]

    for dataset_name in dataset_builder.relation_datasets.keys():
        dataset_names += [dataset_name]

    return dataset_names

def detect_language(examples):
    predictions = pipe_language(examples['clean_text'], batch_size=batch_size)
    return {'language': [pred['label'] for pred in predictions]}

def detect_gibberish(examples):
    predictions = pipe_gibberish(examples['clean_text'], batch_size=batch_size)
    return {'gibberish': [pred['label'] for pred in predictions]}

def process_dataset(ds_builder, dataset_name, save_path):
    print(f"Processing dataset: {dataset_name}")

    train, test, dev = args[dataset_name]["function"]()
    dataset = DatasetDict({
        'train':  Dataset.from_pandas(ds_builder.prepare_filter(train)),
        'test':  Dataset.from_pandas(ds_builder.prepare_filter(test)),
        'dev':  Dataset.from_pandas(ds_builder.prepare_filter(dev)),
    })

    dataset = dataset.map(detect_language, batched=True)
    dataset = dataset.map(detect_gibberish, batched=True)

    dataset_save_path = os.path.join(save_path, dataset_name)
    os.makedirs(dataset_save_path, exist_ok=True)
    dataset['train'].to_parquet(os.path.join(dataset_save_path, "train.pkl"))
    dataset['test'].to_parquet(os.path.join(dataset_save_path, "test.pkl"))
    dataset['dev'].to_parquet(os.path.join(dataset_save_path, "dev.pkl"))

    train_df = dataset['train'].to_pandas()
    print(f"Language counts in training set: {train_df['language'].value_counts()}")
    print(f"Gibberish counts in training set: {train_df['gibberish'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add details to the datasets, create a file that can be used later for analysis.")
    parser.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size')
    args = parser.parse_args()
    batch_size = args.batch_size

    # Initialize components
    ds_builder = DatasetBuilder()
    logger = Logger(log_filename="dataset_size")
    all_datasets = generate_dataset_names(ds_builder)
    print("[INFO] Tital # of dataset is ", len(all_datasets))
    args = generate_args(ds_builder, all_datasets, logger)
    save_path = os.path.join(os.getcwd(), "data", "cleaned_datasets")

    # Load pipelines
    pipe_language = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection",
                            truncation=True, device="cuda")
    pipe_gibberish = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457",
                            truncation=True, device="cuda")

    # Process each dataset
    for dataset_name in all_datasets:
        process_dataset(ds_builder, dataset_name, save_path)
