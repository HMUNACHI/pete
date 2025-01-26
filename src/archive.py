from enum import Enum
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.optim import AdamW

from src.trainer import setup_scaler, setup_scheduler


class Mode(Enum):
    MRPC = "mrpc"
    STSB = "stsb"
    AX = "ax"
    COLA = "cola"
    MNLI = "mnli"
    RTE = "rte"
    QQP = "qqp"
    QNLI = "qnli"
    SST2 = "sst2"
    WNLI = "wnli"
    PAWS = "paws"
    SNLI = "snli"


class Benchmarker(nn.Module):
    def __init__(self, model, d_model):
        super(Benchmarker, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([0.07]))

        # Regressor for regression tasks (e.g., STSB)
        self.regressor = nn.Linear(d_model, 1)

        # Binary classifier for binary classification tasks
        self.binary_classifier = nn.Linear(d_model, 1)

        # Multi-class classifier for multi-class classification tasks (e.g., MNLI, SNLI)
        # Input dimension is 3 * d_model due to concatenation of embeddings
        self.multi_classifier = nn.Linear(3 * d_model, 3)

    def forward(self, batch: torch.Tensor, mode: Mode) -> torch.Tensor:
        if mode == Mode.STSB:
            # Regression task (STS-B)
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
            embeddings1 = self.model(
                input_ids=input_ids1, attention_mask=attention_mask1
            )[1]
            embeddings2 = self.model(
                input_ids=input_ids2, attention_mask=attention_mask2
            )[1]
            combined = torch.abs(embeddings1 - embeddings2)
            preds = self.regressor(combined).squeeze()
            loss = self.regression_loss(preds, labels)
            return loss

        elif mode in [Mode.SST2, Mode.COLA]:
            # Single-sentence binary classification tasks
            input_ids, attention_mask, labels = batch
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)[
                1
            ]
            preds = self.binary_classifier(embeddings).squeeze()
            loss = self.classification_scores(preds, labels)
            return loss

        elif mode in [Mode.MRPC, Mode.QQP, Mode.QNLI, Mode.RTE, Mode.WNLI, Mode.PAWS]:
            # Sentence-pair binary classification tasks
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
            embeddings1 = self.model(
                input_ids=input_ids1, attention_mask=attention_mask1
            )[1]
            embeddings2 = self.model(
                input_ids=input_ids2, attention_mask=attention_mask2
            )[1]
            combined = torch.abs(embeddings1 - embeddings2)
            preds = self.binary_classifier(combined).squeeze()
            loss = self.classification_scores(preds, labels)
            return loss

        elif mode in [Mode.MNLI, Mode.SNLI]:
            # Sentence-pair multi-class classification tasks
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
            embeddings1 = self.model(
                input_ids=input_ids1, attention_mask=attention_mask1
            )[1]
            embeddings2 = self.model(
                input_ids=input_ids2, attention_mask=attention_mask2
            )[1]
            # Concatenate embeddings and their absolute difference
            combined = torch.cat(
                [embeddings1, embeddings2, torch.abs(embeddings1 - embeddings2)], dim=1
            )
            preds = self.multi_classifier(combined)
            loss = self.classification_scores(preds, labels, num_classes=3)
            return loss

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def regression_loss(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(preds, labels)

    def classification_scores(
        self, preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 2
    ) -> torch.Tensor:
        if num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        else:
            loss = F.cross_entropy(preds, labels.long())
        return loss

    def contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        similarity_matrix = (a @ b.T) / (self.temperature + 1e-12)
        labels = torch.arange(similarity_matrix.shape[0]).to(similarity_matrix.device)
        return F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(
            similarity_matrix.T, labels
        )

    def embed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)[
                1
            ]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings

    def compute_similarity(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> torch.Tensor:
        similarity = torch.mm(emb1, emb2.T)
        return similarity

    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: List[str] = None,
        num_samples: int = 1000,
        perplexity: int = 30,
    ) -> None:

        embeddings_np = embeddings.cpu().numpy()
        if num_samples > len(embeddings_np):
            num_samples = len(embeddings_np)
        selected_embeddings = embeddings_np[:num_samples]

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(selected_embeddings)

        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c="blue", alpha=0.5)
        if labels:
            for i, label in enumerate(labels[:num_samples]):
                plt.annotate(label, (tsne_results[i, 0], tsne_results[i, 1]))
        plt.title("t-SNE Visualization of Sentence Embeddings")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.show()


def glue_benchmark(model, experiment, name, d_model):
    benchmarker = Benchmarker(model, d_model)

    optimizer = experiment.tan_optimizer = AdamW(
        benchmarker.parameters(), lr=experiment.learning_rate
    )

    data = experiment.data
    dataset_names = experiment.train_datasets

    num_epochs = experiment.num_epochs
    warmup_steps = experiment.warmup_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmarker.to(device)

    total_steps_per_epoch = sum(
        len(data.data_loaders[dataset_name]["train"])
        for dataset_name in dataset_names
        if "train" in data.data_loaders[dataset_name]
    )
    total_steps = total_steps_per_epoch * num_epochs

    scheduler = setup_scheduler(optimizer, warmup_steps, total_steps)
    scaler = setup_scaler()

    global_step = 0

    dataset_to_mode: Dict[str, Mode] = {
        "mrpc": Mode.MRPC,
        "stsb": Mode.STSB,
        "ax": Mode.AX,
        "cola": Mode.COLA,
        "mnli": Mode.MNLI,
        "rte": Mode.RTE,
        "qqp": Mode.QQP,
        "qnli": Mode.QNLI,
        "sst2": Mode.SST2,
        "wnli": Mode.WNLI,
        "paws": Mode.PAWS,
        "snli": Mode.SNLI,
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for dataset_name in dataset_names:
            if dataset_name not in dataset_to_mode:
                raise ValueError(f"Dataset name '{dataset_name}' is not supported.")

            mode = dataset_to_mode[dataset_name]
            benchmarker.train()
            total_train_loss = 0.0
            train_loader = data.data_loaders[dataset_name]["train"]

            val_split_keys = [
                key
                for key in data.data_loaders[dataset_name].keys()
                if "validation" in key
            ]
            val_loader = None
            if val_split_keys:
                val_loader_key = val_split_keys[0]
                val_loader = data.data_loaders[dataset_name][val_loader_key]

            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                optimizer.zero_grad()

                with torch.autocast(device_type=device, dtype=torch.float16):
                    train_loss = benchmarker(batch, mode)

                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_train_loss += train_loss.item()
                global_step += 1

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"{name} on {dataset_name} Train Loss: {avg_train_loss:.4f}")

            if val_loader is not None:
                benchmarker.eval()
                total_val_loss = 0.0

                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        for batch in val_loader:
                            batch = tuple(t.to(device) for t in batch)
                            val_loss = benchmarker(batch, mode)  # Pass mode here
                            total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / (len(val_loader) + 1e-6)
                print(f"{name} on {dataset_name} Val Loss: {avg_val_loss:.4f}")

    print("Benchmarking completed.")
