#!/usr/bin/env python
# coding: utf-8

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm

# Append the parent directory to sys.path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tan import TAN
from src.trainer import cosine_sim  # Assuming this is used elsewhere

# Configuration
weight_path = "/root/tiny-attention-networks/weights/tan_nano.pt"  # Update if necessary
vocab_size = 30552
d_model = 128
num_layers_and_heads = 2
max_seq_len = 128
batch_size = 32
learning_rate = 2e-5
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Update if using a different tokenizer

# Load the TAN model
tan = TAN(
    vocab_size=vocab_size,
    d_model=d_model,
    num_hidden_layers=num_layers_and_heads,
    num_attention_heads=num_layers_and_heads,
    max_seq_len=max_seq_len,
)

# Load model weights
state_dict = torch.load(weight_path, map_location=device)
tan.load_state_dict(state_dict)
tan.to(device)
tan.eval()  # Set to evaluation mode initially

# Define the Wrapper for embeddings
class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.criterion = nn.MSELoss()

    def forward(self, 
                input_ids_1: torch.Tensor, 
                attention_mask_1: torch.Tensor, 
                input_ids_2: torch.Tensor = None, 
                attention_mask_2: torch.Tensor = None
               ) -> torch.Tensor:
        embedding1 = self.model(input_ids=input_ids_1, attention_mask=attention_mask_1)[1]
        embedding2 = self.model(input_ids=input_ids_2, attention_mask=attention_mask_2)[1] if input_ids_2 is not None else None
        return embedding1, embedding2

# Define the Classification Wrapper
class ClassificationWrapper(nn.Module):
    def __init__(self, model, num_labels, task_type):
        super(ClassificationWrapper, self).__init__()
        self.model = model
        self.classifier = nn.Linear(d_model, num_labels)
        self.task_type = task_type  # 'classification' or 'regression'
        if task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Assuming input_ids is a dictionary with 'input_ids_1' and 'attention_mask_1'
        embeddings1, embeddings2 = self.model(
            input_ids_1=input_ids['input_ids_1'],
            attention_mask_1=input_ids['attention_mask_1'],
            input_ids_2=input_ids.get('input_ids_2', None),
            attention_mask_2=input_ids.get('attention_mask_2', None)
        )
        # For classification, typically use the embedding of the [CLS] token
        logits = self.classifier(embeddings1)
        loss = None
        if labels is not None:
            if self.task_type == 'classification':
                loss = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits.squeeze(), labels.float())
        return {'loss': loss, 'logits': logits}

# Define the Dataset
class GlueDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key in ['input_ids', 'attention_mask']}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Function to preprocess data
def preprocess_data(dataset, task):
    if task == 'stsb':
        # For STS-B, use 'sentence1' and 'sentence2' and 'label' as float
        def preprocess_function(examples):
            return tokenizer(
                examples['sentence1'],
                examples['sentence2'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_len
            )
    elif task in ['mnli', 'mnli_matched', 'mnli_mismatched']:
        # MNLI has different splits
        def preprocess_function(examples):
            return tokenizer(
                examples['premise'],
                examples['hypothesis'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_len
            )
    else:
        # For other classification tasks
        def preprocess_function(examples):
            return tokenizer(
                examples['sentence1'],
                examples['sentence2'],
                padding='max_length',
                truncation=True,
                max_length=max_seq_len
            )
    return dataset.map(preprocess_function, batched=True)

# Function to get number of labels and task type
def get_task_info(task):
    if task == 'cola':
        return 2, 'classification'
    elif task == 'sst2':
        return 2, 'classification'
    elif task == 'mrpc':
        return 2, 'classification'
    elif task == 'qqp':
        return 2, 'classification'
    elif task == 'stsb':
        return 1, 'regression'
    elif task == 'mnli':
        return 3, 'classification'
    elif task == 'qnli':
        return 2, 'classification'
    elif task == 'rte':
        return 2, 'classification'
    elif task == 'wnli':
        return 2, 'classification'
    else:
        raise ValueError(f"Unknown task: {task}")

# Main benchmarking function
def benchmark_glue():
    glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
    results = {}

    for task in glue_tasks:
        print(f"\n{'='*20} Benchmarking on {task.upper()} {'='*20}")
        # Load dataset
        if task in ['mnli_matched', 'mnli_mismatched']:
            dataset = load_dataset('glue', 'mnli')
            if task == 'mnli_matched':
                split = 'validation_matched'
            else:
                split = 'validation_mismatched'
            eval_dataset = dataset[split]
        else:
            dataset = load_dataset('glue', task)
            eval_split = 'validation_matched' if task == 'mnli_matched' else 'validation_mismatched' if task == 'mnli_mismatched' else 'validation'
            eval_dataset = dataset[eval_split]

        # Preprocess data
        encoded_dataset = preprocess_data(dataset, task)
        
        # Determine number of labels and task type
        num_labels, task_type = get_task_info(task)
        
        # Create PyTorch datasets
        if task in ['mnli_matched', 'mnli_mismatched']:
            train_dataset = GlueDataset(encoded_dataset['train'], encoded_dataset['train']['label'])
            eval_dataset_split = GlueDataset(encoded_dataset[split], encoded_dataset[split]['label'])
        else:
            train_dataset = GlueDataset(encoded_dataset['train'], encoded_dataset['train']['label'])
            eval_dataset_split = GlueDataset(encoded_dataset['validation'], encoded_dataset['validation']['label'])
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset_split, batch_size=batch_size)
        
        # Initialize the model
        model = ClassificationWrapper(tan, num_labels, task_type)
        model.to(device)
        
        # Define optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Define the appropriate metric using the 'evaluate' library
        if task == 'mnli_matched' or task == 'mnli_mismatched':
            metric = evaluate.load("glue", task="mnli")
        else:
            metric = evaluate.load("glue", task=task)
        
        # Training Loop
        model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc="Training", leave=False)
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = {
                    'input_ids_1': batch['input_ids'].to(device),
                    'attention_mask_1': batch['attention_mask'].to(device)
                }
                
                # For tasks with sentence pairs, include input_ids_2 and attention_mask_2
                # In this script, all tasks are treated as sentence-pair tasks
                # If your model handles single sentences differently, adjust accordingly
                # Here, 'input_ids_2' and 'attention_mask_2' are the same as 'input_ids_1' and 'attention_mask_1'
                input_ids['input_ids_2'] = batch['input_ids'].to(device)
                input_ids['attention_mask_2'] = batch['attention_mask'].to(device)
                
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, batch['attention_mask'].to(device), labels=labels)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Average Training Loss: {avg_loss:.4f}")
        
        # Evaluation Loop
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            progress_bar = tqdm(eval_loader, desc="Evaluating", leave=False)
            for batch in progress_bar:
                input_ids = {
                    'input_ids_1': batch['input_ids'].to(device),
                    'attention_mask_1': batch['attention_mask'].to(device)
                }
                
                # For tasks with sentence pairs, include input_ids_2 and attention_mask_2
                input_ids['input_ids_2'] = batch['input_ids'].to(device)
                input_ids['attention_mask_2'] = batch['attention_mask'].to(device)
                
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, batch['attention_mask'].to(device), labels=None)
                logits = outputs['logits']
                
                if task_type == 'classification':
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    predictions = logits.squeeze()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # Compute Metrics
        if task_type == 'classification':
            if task in ['mnli_matched', 'mnli_mismatched']:
                preds = all_predictions
                refs = all_labels
                # For MNLI, specify 'matched' or 'mismatched' during metric computation if needed
                # However, 'evaluate' handles it based on the task parameter
                result = metric.compute(predictions=preds, references=refs)
                accuracy = result.get('accuracy', None)
                f1 = result.get('f1', None)
                results[task] = {}
                if accuracy is not None:
                    results[task]['accuracy'] = accuracy
                    print(f"Validation Accuracy: {accuracy:.4f}")
                if f1 is not None:
                    results[task]['f1'] = f1
                    print(f"Validation F1: {f1:.4f}")
            elif task == 'sst2':
                preds = all_predictions
                refs = all_labels
                # SST-2 is binary classification
                accuracy = accuracy_score(refs, preds)
                f1 = f1_score(refs, preds)
                results[task] = {'accuracy': accuracy, 'f1': f1}
                print(f"Validation Accuracy: {accuracy:.4f}")
                print(f"Validation F1: {f1:.4f}")
            else:
                preds = all_predictions
                refs = all_labels
                accuracy = accuracy_score(refs, preds)
                f1 = f1_score(refs, preds, average='weighted')
                results[task] = {'accuracy': accuracy, 'f1': f1}
                print(f"Validation Accuracy: {accuracy:.4f}")
                print(f"Validation F1: {f1:.4f}")
        else:
            # Regression task (stsb)
            mse = mean_squared_error(all_labels, all_predictions)
            pearson = pearsonr(all_labels, all_predictions)[0]
            spearman = spearmanr(all_labels, all_predictions)[0]
            results[task] = {'mse': mse, 'pearson': pearson, 'spearman': spearman}
            print(f"Validation MSE: {mse:.4f}")
            print(f"Validation Pearson Correlation: {pearson:.4f}")
            print(f"Validation Spearman Correlation: {spearman:.4f}")
        
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    # Summary of Results
    print("\n" + "="*30 + " Benchmarking Results " + "="*30)
    for task, metrics in results.items():
        print(f"\nTask: {task.upper()}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    benchmark_glue()
