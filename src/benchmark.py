from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    f1_score,
    precision_score,
    matthews_corrcoef
)

from src.tan import MLP


class GLUEWrapper(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        num_outputs: int = 0,
        num_sentences: int = 2,
    ):
        super(GLUEWrapper, self).__init__()
        self.model = model
        self.num_outputs = num_outputs
        self.num_sentences = num_sentences

        if self.num_outputs > 0:
            self.classifier = MLP(self.model.d_model * num_sentences, num_outputs)
            self.temperature = nn.Parameter(torch.tensor([0.07]))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        if self.num_sentences == 1:
            embedding = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
            labels = batch[-1].long()
            return self.classification_loss_one_sentence(embedding, labels)

        anchors = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        positives = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        labels = batch[-1].long()

        if self.num_outputs == 0:
            return self.correlation_loss(anchors, positives, labels)
        
        return self.classification_loss(anchors, positives, labels)

    def forward_one_sentence(self, batch: torch.Tensor) -> torch.Tensor:
        embedding = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        labels = batch[-1].long()
        return self.classification_loss(anchors, positives, labels)

    def get_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        if self.num_outputs == 0:
            return self.correlation_predictions(batch)
        return self.classification_predictions(batch)

    def classification_loss(
            self, 
            sentence1: torch.Tensor, 
            sentence2: torch.Tensor, 
            ground_truth: torch.Tensor
        ) -> torch.Tensor:

        combined_rep = torch.cat([sentence1, sentence2], dim=-1)
        logits = self.classifier(combined_rep) 
        return F.cross_entropy(logits, ground_truth)

    def classification_loss_one_sentence(
            self, 
            sentence: torch.Tensor,
            ground_truth: torch.Tensor
        ) -> torch.Tensor:
        logits = self.classifier(sentence) 
        return F.cross_entropy(logits, ground_truth)

    def classification_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        outputs1 = self.model(input_ids=batch[0], attention_mask=batch[1])
        outputs2 = self.model(input_ids=batch[2], attention_mask=batch[3])
        sentence1 = outputs1[1]
        sentence2 = outputs2[1]
        combined_rep = torch.cat([sentence1, sentence2], dim=-1)
        logits = self.classifier(combined_rep)
        preds = torch.argmax(logits, dim=-1)
        return preds

    def correlation_loss(
            self, 
            sentence1: torch.Tensor, 
            sentence2: torch.Tensor, 
            ground_truth: torch.Tensor
        ) -> torch.Tensor:

        similarities = cosine_sim(sentence1, sentence2).diagonal()
        predicted_correlation = pearson_r(similarities, ground_truth)
        return 1 - predicted_correlation

    def correlation_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        sentence1 = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        sentence2 = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        return cosine_sim(sentence1, sentence2).diagonal()
    

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dot_product = torch.einsum("ij,kj->ik", a, b)
    norm_a = torch.sqrt(torch.einsum("ij,ij->i", a, a))
    norm_b = torch.sqrt(torch.einsum("ij,ij->i", b, b))
    norm_product = torch.einsum("i,j->ij", norm_a, norm_b)
    return dot_product / norm_product


def pearson_r(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2)) * torch.sqrt(torch.sum(y_centered ** 2))
    return numerator / (denominator + eps)


def evaluate(
    model: nn.Module,
    data_loader: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    dataset_name: str,
    run_name: str,
    test: bool = False,
) -> Optional[Dict[str, float]]:

    model.to(device)
    model.eval()

    if test:
        all_predictions = []
        weight_path = f"weights/{run_name}.pt"
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in data_loader[dataset_name]["test"]:
                    batch = [x.to(device) for x in batch]
                    preds = model.get_predictions(batch).cpu().numpy()
                    all_predictions.append(preds)

        all_predictions = np.concatenate(all_predictions, axis=0)
        file_path = f"results/{run_name}.npy"
        np.save(file_path, all_predictions)
        print(f"Test predictions saved to {file_path}")
        return

    all_preds = []
    all_labels = []

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in data_loader[dataset_name]["validation"]:

                batch = [x.to(device) for x in batch]
                preds = model.get_predictions(batch)
                labels = batch[4].long()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if dataset_name == "stsb":
        return spearman_evaluate(all_preds, all_labels)

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    mcc = matthews_corrcoef(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "mcc": mcc
    }

    return metrics


def spearman_evaluate(
    similarities: np.array,
    labels: np.array,
) -> Optional[Dict[str, float]]:

    eval_pearson_cosine, _ = pearsonr(similarities, labels)
    eval_spearman_cosine, _ = spearmanr(similarities, labels)

    return {
        "pearsonr": eval_pearson_cosine, 
        "spearmanr": eval_spearman_cosine
    }