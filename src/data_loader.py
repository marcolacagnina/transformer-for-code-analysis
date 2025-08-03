# src/data_loader.py

import json
import random
import torch
from collections import Counter
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_dataloaders(config):
    """
    Loads, processes, and splits the dataset, returning train, validation, and test dataloaders.
    """
    random.seed(config.SEED)

    with open(config.DATASET_PATH, "r") as f:
        processed_dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': config.CUSTOM_TAGS})

    filtered_dataset = [
        entry for entry in processed_dataset
        if len(tokenizer.tokenize(entry['solution_code'])) <= config.MAX_TOKEN_LENGTH
    ]

    unique_classes = sorted(set(ex['time_complexity'] for ex in filtered_dataset))
    label2id = {label: idx for idx, label in enumerate(unique_classes)}
    id2label = {idx: label for label, idx in label2id.items()}

    for example in filtered_dataset:
        example['label'] = label2id[example['time_complexity']]

    random.shuffle(filtered_dataset)
    train_split = int(0.8 * len(filtered_dataset))
    val_test_split = int(0.9 * len(filtered_dataset))

    train_data = filtered_dataset[:train_split]
    val_data = filtered_dataset[train_split:val_test_split]
    test_data = filtered_dataset[val_test_split:]

    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    # --- Tokenize ---
    def tokenize_function(examples):
        # Tokenize code
        tokenized_output = tokenizer(
            examples['solution_code'], padding='max_length', truncation=True, max_length=config.MAX_TOKEN_LENGTH
        )

        tokenized_output['labels'] = examples['label']
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                    remove_columns=['solution_code', 'time_complexity'])

    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=config.BATCH_SIZE,
                                num_workers=config.NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=config.BATCH_SIZE,
                                 num_workers=config.NUM_WORKERS, pin_memory=True)

    label_counts = Counter(ex['label'] for ex in train_data)
    num_classes = len(label_counts)
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor([total_samples / (num_classes * label_counts[i]) for i in range(num_classes)],
                                 device=config.DEVICE)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer, label2id, id2label, class_weights