from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE


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
        # Mean Squared Error loss for regression tasks
        return F.mse_loss(preds, labels)

    def classification_scores(
        self, preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 2
    ) -> torch.Tensor:
        if num_classes == 2:
            # Binary classification using BCEWithLogitsLoss
            loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        else:
            # Multi-class classification using CrossEntropyLoss
            loss = F.cross_entropy(preds, labels.long())
        return loss

    def contrastive_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Contrastive loss computation
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


def glue_benchmark(model, experiment, name):
    print(model, experiment, name)
