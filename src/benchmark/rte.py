from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.tan import MLP  # Or define your own MLP / classifier head

class RTEWrapper(nn.Module):
    """
    A wrapper for RTE (Recognizing Textual Entailment).
    Expects a backbone model that can output representations of each sentence.
    Then uses an MLP as a classifier on top of these representations to
    predict entailment vs. not-entailment (binary classification).
    """
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 768,
        num_labels: int = 2,
    ):
        super(RTEWrapper, self).__init__()
        self.model = model
        # Example classifier: MLP that takes [embed1; embed2] -> 2-label output
        self.classifier = MLP(
            hidden_size * 2, 
            hidden_dims=[hidden_size], 
            output_dim=num_labels
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns the loss.
        batch structure is assumed to be:
            batch[0]: input_ids for sentence1
            batch[1]: attention_mask for sentence1
            batch[2]: input_ids for sentence2
            batch[3]: attention_mask for sentence2
            batch[4]: labels (0 or 1 for RTE)
        """
        # Get model outputs (e.g., pooled CLS embeddings)
        # If you're using a standard Hugging Face transformer, 
        # model(...) might return: (last_hidden_state, pooler_output) or something similar
        # so indexing [1] would be the pooled output.
        # Adjust as needed for your model's actual return signature.
        outputs1 = self.model(input_ids=batch[0], attention_mask=batch[1])
        outputs2 = self.model(input_ids=batch[2], attention_mask=batch[3])
        
        # Example: using outputs[1] as the "pooled" or "CLS" embedding
        sentence1 = outputs1[1]  # shape: (batch_size, hidden_size)
        sentence2 = outputs2[1]  # shape: (batch_size, hidden_size)
        
        # Concatenate or otherwise combine the two representations
        combined_rep = torch.cat([sentence1, sentence2], dim=-1)  # shape: (batch_size, 2 * hidden_size)
        
        # Classifier head
        logits = self.classifier(combined_rep)  # shape: (batch_size, num_labels)
        
        labels = batch[4]
        loss = F.cross_entropy(logits, labels)  # cross-entropy for 2-label classification
        return loss

    def get_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Return predicted labels (not the loss).
        """
        with torch.no_grad():
            outputs1 = self.model(input_ids=batch[0], attention_mask=batch[1])
            outputs2 = self.model(input_ids=batch[2], attention_mask=batch[3])
            sentence1 = outputs1[1]
            sentence2 = outputs2[1]
            combined_rep = torch.cat([sentence1, sentence2], dim=-1)
            logits = self.classifier(combined_rep)
            
            # Predicted class (0 or 1)
            preds = torch.argmax(logits, dim=-1)
        return preds

def evaluate(
    model: nn.Module,
    data_loader: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    test: bool = False,
) -> Optional[float]:
    """
    Evaluate function analogous to the STSB one.
    - If `test` is False, run validation and return accuracy.
    - If `test` is True, you could load a saved checkpoint and store predictions.
    """
    model.to(device)
    model.eval()

    if test:
        # Example: load some checkpoint and save predictions for a test set
        # Adjust to your environment, naming, etc.
        all_predictions = []
        weight_path = "weights/RTE_checkpoint.pt"
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

        with torch.no_grad():
            # Mixed-precision example if your hardware supports it
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in data_loader["rte"]["test"]:
                    batch = [x.to(device) for x in batch]
                    preds = model.get_predictions(batch).cpu().numpy()
                    all_predictions.append(preds)

        all_predictions = np.concatenate(all_predictions, axis=0)
        file_path = "results/rte_predictions.npy"
        np.save(file_path, all_predictions)
        print(f"Test predictions saved to {file_path}")
        return

    # Validation / development set evaluation
    total = 0
    correct = 0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in data_loader["rte"]["validation"]:
                batch = [x.to(device) for x in batch]
                # Forward pass returns the loss, but here we only need predictions to compute accuracy
                preds = model.get_predictions(batch)  # shape: (batch_size,)
                labels = batch[4]                     # shape: (batch_size,)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
