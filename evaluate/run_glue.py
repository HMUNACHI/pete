#!/usr/bin/env python
# coding=utf-8

"""
Finetune a 🤗 Transformers model on a text classification task (GLUE) without using `accelerate` or `huggingface_hub`.
"""

import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.49.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Define a simple Args class to hold all parameters
class Args:
    def __init__(self):
        # Task and data parameters
        self.task_name = "sst2"  # e.g., "sst2", "mrpc", etc. Set to None if using custom files
        self.train_file = None  # Path to training file (CSV or JSON) if not using a GLUE task
        self.validation_file = None  # Path to validation file (CSV or JSON) if not using a GLUE task

        # Tokenization parameters
        self.max_length = 128
        self.pad_to_max_length = True

        # Model parameters
        self.model_name_or_path = "bert-base-uncased"  # Path or model identifier from Hugging Face
        self.use_slow_tokenizer = False

        # Training parameters
        self.per_device_train_batch_size = 8
        self.per_device_eval_batch_size = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.num_train_epochs = 3
        self.max_train_steps = None  # If set, overrides num_train_epochs
        self.gradient_accumulation_steps = 1

        # Scheduler parameters
        self.lr_scheduler_type = "linear"  # Options: "linear", "cosine", etc.
        self.num_warmup_steps = 0

        # Output and logging
        self.output_dir = "./results"  # Directory to save the final model
        self.seed = 42  # Random seed for reproducibility

        # Advanced parameters
        self.trust_remote_code = False
        self.ignore_mismatched_sizes = False

# Instantiate the Args class with desired parameters
args = Args()

def main():
    # Uncomment the following line if you want to send telemetry data
    # send_example_telemetry("run_glue_no_trainer", args)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    global logger
    logger = logging.getLogger(__name__)

    # If passed along, set the training seed now for reproducibility.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # Load the dataset
    # --------------------------------------------------------------------------
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if not data_files:
            raise ValueError("Either task_name or train_file/validation_file must be provided.")
        extension = (
            args.train_file.split(".")[-1]
            if args.train_file is not None
            else args.validation_file.split(".")[-1]
        )
        assert extension in ["csv", "json"], "`train_file` and `validation_file` should be CSV or JSON files."
        raw_datasets = load_dataset(extension, data_files=data_files)

    # --------------------------------------------------------------------------
    # Prepare label information
    # --------------------------------------------------------------------------
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak for your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # sort for determinism
            num_labels = len(label_list)

    # --------------------------------------------------------------------------
    # Load pretrained model & tokenizer
    # --------------------------------------------------------------------------
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    # Ensure pad token is defined if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    # --------------------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------------------
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Try to have some nice defaults but adapt to your own use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some models have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"Using the model's label correspondence: {label_name_to_id}."
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Model labels don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                result["labels"] = examples["label"]
        return result

    with datasets.utils.disable_progress_bar():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset_name = "validation_matched" if args.task_name == "mnli" else "validation"
    if eval_dataset_name not in processed_datasets:
        raise ValueError(f"Validation set not found for {args.task_name}.")
    eval_dataset = processed_datasets[eval_dataset_name]

    # --------------------------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------------------------
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # --------------------------------------------------------------------------
    # Optimizer & Scheduler
    # --------------------------------------------------------------------------
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and number of total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # --------------------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------------------
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step % args.gradient_accumulation_steps == 0) or (step == len(train_dataloader) - 1):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # ----------------------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------------------
        model.eval()
        metric.reset()
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            if not is_regression:
                predictions = outputs.logits.argmax(dim=-1)
            else:
                predictions = outputs.logits.squeeze()

            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, Eval: {eval_metric}")

        if completed_steps >= args.max_train_steps:
            break

    progress_bar.close()

    # --------------------------------------------------------------------------
    # Save the final model
    # --------------------------------------------------------------------------
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # --------------------------------------------------------------------------
    # (Optional) Evaluate on MNLI mismatch set if needed
    # --------------------------------------------------------------------------
    if args.task_name == "mnli":
        mismatched_dataset_name = "validation_mismatched"
        if mismatched_dataset_name in processed_datasets:
            eval_dataset_mm = processed_datasets[mismatched_dataset_name]
            eval_dataloader_mm = DataLoader(
                eval_dataset_mm,
                collate_fn=data_collator,
                batch_size=args.per_device_eval_batch_size,
            )

            model.eval()
            metric.reset()
            for batch in eval_dataloader_mm:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            eval_metric_mm = metric.compute()
            logger.info(f"MNLI (mismatched) Eval: {eval_metric_mm}")


if __name__ == "__main__":
    """
    Example usage:
    
    Simply run the script without any arguments:
    
    python run_glue.py
    """
    main()
