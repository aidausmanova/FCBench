import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
import math
import pickle
import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

from transformers import (Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, 
                          AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, EarlyStoppingCallback)

HF_REPO = "anonymous"

def save_mappings(mapping_file, label_map, dataset_name):
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as file:
            label_mappings = json.load(file)
    else:
        label_mappings = {}

    label_mappings[dataset_name] = label_map

    with open(mapping_file, 'w') as file:
        json.dump(label_mappings, file, indent=4)

class CartographyCallback(TrainerCallback):
    def __init__(self, trainer, dataset, dataset_name, seed):
        super().__init__()
        self.epoch_probabilities = []
        self.correct_predictions = []
        self.epoch_logits = []

        self._trainer = trainer
        self._dataset = dataset

        self.save_path = f"experiment_results/cartography/distilRoBERTa/{dataset_name}_{seed}.tsv"

    def load(self):
        self.epoch_probabilities = []
        self.correct_predictions = []
        self.epoch_logits = []

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logits, labels, _ = self._trainer.predict(self._dataset)

        # Logits
        self.epoch_logits.append(logits)

        # Probabilities
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        true_label_probs = probs[range(len(labels)), labels].tolist()
        self.epoch_probabilities.append(true_label_probs)

        # Correctness
        predictions = torch.argmax(probs, dim=-1)
        correct = (predictions == torch.tensor(labels)).tolist()
        self.correct_predictions.append(correct)

    def compute_metrics(self):
        # Calculate the mean probability (confidence) for each instance
        mean_probs = [sum(probs) / len(self.epoch_probabilities) for probs in zip(*self.epoch_probabilities)]
        # Calculate the variability (standard deviation) for each instance
        variability = []
        for index in range(len(mean_probs)):
            sum_squared_diff = sum((prob[index] - mean_probs[index]) ** 2 for prob in self.epoch_probabilities)
            variance = sum_squared_diff / len(self.epoch_probabilities)
            std_dev = math.sqrt(variance)
            variability.append(std_dev)
        # Calculate the correctness (fraction of correct predictions) for each instance
        correctness = [sum(correct) / len(self.correct_predictions) for correct in zip(*self.correct_predictions)]
        return mean_probs, variability, correctness

    def save(self):
        epoch_logits_list = [logits.tolist() for logits in self.epoch_logits]

        # Save raw data
        data = {
            'text': self._dataset['text'],
            'labels': self._dataset['label'],
            'probabilities': self.epoch_probabilities,
            'logits': epoch_logits_list
        }
        json_data = json.dumps(data)
        binary_data = pickle.dumps(json_data)
        with open(self.save_path[:-4]+'.pkl', 'wb') as file:
            file.write(binary_data)

        # Save computed metrics
        mean_probs, variability, correctness = self.compute_metrics()
        pd.DataFrame({
            "text":self._dataset['text'],
            "label":self._dataset['label'],
            "mean_probs":mean_probs,
            "variability":variability,
            "correctness":correctness
        }).to_csv(self.save_path, sep="\t", index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_weights(dataset):
    labels = [label for _, label in dataset]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get('logits')

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float)).to(
            self.model.device)
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_distilRoBERTa(X_train, y_train, X_val, y_val, X_test, y_test, model_save_path, logging_dir, dataset_name, logger, batch_size, accumulation_steps, classification_type="standard", weighted_training=False, seed=42):
    # Prepare datasets
    unique_labels = sorted(list(set(y_train)))
    label2id = {label: i for i, label in enumerate(unique_labels)}

    id2label = {v: k for k, v in label2id.items()}

    y_train = [label2id[y] for y in y_train]
    y_val = [label2id[y] for y in y_val]
    y_test = [label2id[y] for y in y_test]

    if classification_type == "relation/stance":
        train_dataset = Dataset.from_dict({'text': X_train['text'], 'query': X_train['query'], 'label': y_train})
        val_dataset = Dataset.from_dict({'text': X_val['text'], 'query': X_val['query'], 'label': y_val})
        test_dataset = Dataset.from_dict({'text': X_test['text'], 'query': X_test['query'], 'label': y_test})
    else:
        train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
        val_dataset = Dataset.from_dict({'text': X_val, 'label': y_val})
        test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("distilbert/distilroberta-base", num_labels=len(set(y_train)), id2label=id2label, label2id=label2id)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    # Preprocess the text data

    if classification_type == "relation/stance":
        print("Relation Tokenizer")
        def tokenize_function(examples):
            return tokenizer(examples['text'], examples['query'], padding="max_length", truncation=True, max_length=512)
    else:
        print("Classifical Tokenizer")
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        # warmup_steps=500,
        # weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        push_to_hub = True,
        hub_private_repo=True,
        hub_model_id = (f"{HF_REPO}/"+dataset_name+"_"+str(seed)+"_distilRoBERTa").replace("&", "and"),
        hub_token = os.environ["HUB_TOKEN"],
        disable_tqdm=True,

        fp16=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,

    )

    data_collator = DataCollatorWithPadding(tokenizer)

    # Initialize Trainer
    if weighted_training:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset['label']), y=train_dataset['label'])
        # To be removed if it does not work testing to be similar to climabench
        class_weights = class_weights / sum(class_weights)

        trainer = WeightedTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer,
            class_weights = class_weights
        )
    else:
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    # Add callbacks for cartography and early stopping
    training_carto_callback = CartographyCallback(trainer, train_dataset, dataset_name + "_train", seed)
    trainer.add_callback(training_carto_callback)
    validation_carto_callback = CartographyCallback(trainer, val_dataset, dataset_name + "_validation", seed)
    trainer.add_callback(validation_carto_callback)

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Train the model
    trainer.train()

    # Save the cartography metrics
    training_carto_callback.save()
    validation_carto_callback.save()

    trainer.push_to_hub(commit_message="Best model according to evaluation metric")

    # # Save the model and the tokenizer
    # model.save_pretrained(model_save_path)
    # tokenizer.save_pretrained(model_save_path)

    # Evaluate on test
    logger.add_trainer_f1_score(
        trainer=trainer,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        model_type='distilRoBERTa',
        n_labels=len(np.unique(y_train)),
        n_epoch=trainer.state.epoch
    )

    save_mappings(
        mapping_file="experiment_results/distilRoBERTa/label_mappings.json",
        label_map=label2id,
        dataset_name=dataset_name
    )

    if not os.path.exists(model_save_path):
        print("Folder not found")
    else:
        os.system(f'rm -r {model_save_path}')


# ### MULTI-LABEL

class CartographyCallbackMultilabel(TrainerCallback):
    def __init__(self, trainer, dataset, dataset_name, seed):
        super().__init__()
        self.epoch_logits = []

        self._trainer = trainer
        self._dataset = dataset

        self.save_path = f"experiment_results/cartography/distilRoBERTa/{dataset_name}_{seed}.tsv"

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logits, labels, _ = self._trainer.predict(self._dataset)
        # Logits
        self.epoch_logits.append(logits)

    def save(self):
        epoch_logits_list = [logits.tolist() for logits in self.epoch_logits]

        # Save raw data
        data = {
            'text': self._dataset['text'],
            'labels': self._dataset['labels'],
            'logits': epoch_logits_list
        }
        json_data = json.dumps(data)
        binary_data = pickle.dumps(json_data)
        with open(self.save_path[:-4]+'.pkl', 'wb') as file:
            file.write(binary_data)

def compute_metrics_multilabel(eval_pred):
    logits, labels = eval_pred
    logits = torch.sigmoid(torch.tensor(logits))
    predictions = (logits > 0.5).int().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class WeightedTrainerMulti(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")  # Expecting shape [batch_size, num_labels]

        outputs = model(**inputs)
        logits = outputs.get('logits')  # Shape [batch_size, num_labels]

        # For multilabel classification, we use BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(self.class_weights, dtype=torch.float)).to(self.model.device)
        loss = criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_weight_multi(train_dataset):
    """
    Computes the normalized inverse frequency of each class in a multilabel dataset,
    so that the majority class has a value of 1.

    Parameters:
    train_dataset (np.array): A 2D numpy array where each row represents a sample
                              and each column represents a class with binary labels (0 or 1).

    Returns:
    np.array: An array containing the normalized inverse frequency of each class.
    """
    # Sum the occurrences of each class (column-wise)
    label_counts = np.sum(train_dataset, axis=0)

    # Compute the inverse frequency
    inverse_freq = 1.0 / label_counts

    # Handle any division by zero (in case a label doesn't appear at all)
    inverse_freq = np.where(label_counts == 0, 0, inverse_freq)

    # Normalize by the inverse of the majority class (i.e., the minimum inverse frequency)
    min_inverse_freq = np.min(inverse_freq[np.nonzero(inverse_freq)])  # Ignore any zero values
    normalized_inverse_freq = inverse_freq / min_inverse_freq

    return normalized_inverse_freq


def train_multi_distilRoBERTa(X_train, y_train, X_val, y_val, X_test, y_test, model_save_path, logging_dir, dataset_name, logger, batch_size, accumulation_steps, classification_type="standard", weighted_training=False, seed=42):
    # hot-ecoding + id2label
    mlb = MultiLabelBinarizer()

    y_train = mlb.fit_transform(y_train)
    y_train = y_train.astype(float)

    y_val = mlb.transform(y_val)
    y_val = y_val.astype(float)

    y_test = mlb.transform(y_test)
    y_test = y_test.astype(float)

    # Update label2id and id2label accordingly
    label2id = {label: i for i, label in enumerate(mlb.classes_)}
    id2label = {i: label for label, i in label2id.items()}

    n_labels = len(label2id)

    train_dataset = Dataset.from_dict({'text': X_train, 'labels': y_train})
    val_dataset = Dataset.from_dict({'text': X_val, 'labels': y_val})
    test_dataset = Dataset.from_dict({'text': X_test, 'labels': y_test})

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained("distilbert/distilroberta-base", num_labels=n_labels, id2label=id2label, label2id=label2id, problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=20,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        # warmup_steps=500,
        # weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=seed,
        push_to_hub = True,
        hub_private_repo=True,
        hub_model_id = (f"{HF_REPO}/"+dataset_name+"_"+str(seed)+"_distilRoBERTa").replace("&", "and"),
        hub_token = os.environ["HUB_TOKEN"],
        disable_tqdm=True,

        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        warmup_ratio=0.1,
        weight_decay=0.01,

    )

    data_collator = DataCollatorWithPadding(tokenizer)

    if weighted_training:
        class_weights = compute_weight_multi(y_train)
        trainer = WeightedTrainerMulti(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_multilabel,
            data_collator=data_collator,
            tokenizer=tokenizer,
            class_weights=class_weights
        )
    else:
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_multilabel,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    # Add callbacks for cartography and early stopping
    training_carto_callback = CartographyCallbackMultilabel(trainer, train_dataset, dataset_name + "_train", seed)
    trainer.add_callback(training_carto_callback)
    validation_carto_callback = CartographyCallbackMultilabel(trainer, val_dataset, dataset_name + "_validation", seed)
    trainer.add_callback(validation_carto_callback)

    # trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Train the model
    trainer.train()

    # Save the cartography metrics
    training_carto_callback.save()
    validation_carto_callback.save()

    trainer.push_to_hub(commit_message="Best model according to evaluation metric")

    # Evaluate on test
    logger.add_trainer_f1_score_multi(
        trainer=trainer,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        model_type='distilRoBERTa',
        n_labels=n_labels,
        n_epoch=trainer.state.epoch
    )

    if not os.path.exists(model_save_path):
        print("Folder not found")
    else:
        os.system(f'rm -r {model_save_path}')
