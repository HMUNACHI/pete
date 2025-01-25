from typing import List
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from typing import Dict, List, Optional
from src.tan import MLP


class STSBWrapper(nn.Module):
    def __init__(self, model):
        super(STSBWrapper, self).__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        sentence1 = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        sentence2 = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        correlation = batch[4]
        return self.loss(sentence1, sentence2, correlation)

    def get_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        sentence1 = self.model(input_ids=batch[0], attention_mask=batch[1])[1]
        sentence2 = self.model(input_ids=batch[2], attention_mask=batch[3])[1]
        return cosine_sim(sentence1, sentence2).diagonal()

    def loss(self, a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarities = cosine_sim(a, b).diagonal()
        predicted_correlation = pearson_r(similarities, labels)
        return 1 - predicted_correlation


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
    model: torch.nn.Module,
    data_loader: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    test: bool = False,
) -> List[float]:
    model.to(device)
    model.eval()
    vectors_one = []
    vectors_two = []
    labels = []

    if test:
        # Model wrapper will be passed in this case
        all_predictions = []
        weight_path = f"weights/TAN_{model.model.num_attention_heads}_{model.model.d_model}.pt"
        state_dict = torch.load(weight_path, map_location=torch.device("cuda"))
        model.load_state_dict(state_dict)
        with torch.no_grad():

            with torch.autocast(device_type="cuda", dtype=torch.float16):

                for batch in data_loader["stsb"]["test"]:
                    batch = [pair.to(device) for pair in batch]
                    predictions = model.get_predictions(batch).cpu().float().numpy()
                    all_predictions.append(predictions)

        file_path = f"results/tan_{model.model.num_attention_heads}_{model.model.d_model}.npy"
        np.save(file_path, all_predictions)
        print(f"Test predictions saved to {file_path}")
        return 

    with torch.no_grad():

        with torch.autocast(device_type="cuda", dtype=torch.float16):

            for batch in data_loader["stsb"]["validation"]:
                embeddings1 = model(
                    input_ids=batch[0].to(device), attention_mask=batch[1].to(device)
                )[1]
                embeddings2 = model(
                    input_ids=batch[2].to(device), attention_mask=batch[3].to(device)
                )[1]
                vectors_one.append(embeddings1.cpu().float())
                vectors_two.append(embeddings2.cpu().float())
                labels.extend(batch[4].cpu().numpy())

    vectors_one = torch.cat(vectors_one)
    vectors_two = torch.cat(vectors_two)
    labels = np.array(labels)
    similarities = cosine_sim(vectors_one, vectors_two).numpy().diagonal()
    eval_pearson_cosine, _ = pearsonr(similarities, labels)
    eval_spearman_cosine, _ = spearmanr(similarities, labels)
    return [eval_pearson_cosine, eval_spearman_cosine]