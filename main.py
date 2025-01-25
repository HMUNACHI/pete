import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertTokenizer

torch.manual_seed(0)
np.random.seed(0)

from src.data import GlueDatasetLoader
# from src.embedder import Embedder
from src.benchmark.stsb import STSBWrapper as Embedder
from src.tan import TAN
from src.trainer import train
from src.transformer import Transformer
from src.utils import timer


class Experiment:
    def __init__(
        self,
        args,
        vocab_size: int = 30552,
        d_model: int = 128,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 1,
        dropout_prob: float = 0.2,
        max_seq_len: int = 128,
        num_epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        train_datasets: list = ["snli", "mnli"],
        validation_datasets: list = ["stsb"],
        train_baseline=False,
    ):

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.train_baseline = train_baseline

        benchmark_dataset = [
            "mrpc", "stsb", "ax", "cola", "mnli", "rte", 
            "qqp", "qnli", "sst2", "wnli", "paws", "snli"
        ]

        if args.config == "atomic":
            self.num_attention_heads = 1
            self.num_hidden_layers = 1
            self.d_model = 64
        elif args.config == "nano":
            self.num_attention_heads = 2
            self.num_hidden_layers = 2
            self.d_model = 128
        elif args.config == "micro":
            self.num_attention_heads = 2
            self.num_hidden_layers = 2
            self.d_model = 256
        elif args.config == "milli":
            self.num_attention_heads = 1
            self.num_hidden_layers = 1
            self.d_model = 512
        else:
            raise ValueError(f"No configuration called '{args.config}'.")

        self.data = GlueDatasetLoader(
            tokenizer=self.tokenizer,
            max_length=self.max_seq_len,
            batch_size=self.batch_size,
            dataset_names=self.train_datasets + self.validation_datasets,
        )

        tan = TAN(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_seq_len=self.max_seq_len,
        )

        if args.pretrained:
            weight_path = os.path.join("pretrained_weights", "tan_" + args.config + ".pt")
            state_dict = torch.load(weight_path, map_location=torch.device("cuda"))
            tan.load_state_dict(state_dict)

        self.tan_embedder = Embedder(tan)
        self.tan_optimizer = AdamW(self.tan_embedder.parameters(), lr=self.learning_rate)
        print(f"\nNum of params TAN: {sum(p.numel() for p in tan.parameters() if p.requires_grad)}")

        if self.train_baseline:
            transformer = Transformer(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                max_position_embeddings=self.max_seq_len,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                intermediate_size=512,
            )
            self.transformer_embedder = Embedder(transformer)
            self.transformer_optimizer = AdamW(
                self.transformer_embedder.parameters(), lr=self.learning_rate
            )
            print(
                f"Num of params in Transformer: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}"
            )


def run(experiment):
    if experiment.train_baseline:
        print("\nTraining Transformer\n")
        name = f"transformer_{experiment.num_hidden_layers}_{experiment.d_model}"
        with timer(f"Transformer training ({name})"):
            transformer_embedder = train(
                experiment.transformer_embedder,
                experiment.transformer_optimizer,
                experiment,
                name,
            )

    print("\nTraining TAN\n")
    name = f"TAN_{experiment.num_hidden_layers}_{experiment.d_model}"
    with timer(f"TAN training ({name})"):
        tan_embedder = train(
            experiment.tan_embedder, experiment.tan_optimizer, experiment, name
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train models with various parameters."
    )

    parser.add_argument(
        "--num-hidden-layers",
        nargs="+",
        type=int,
        default=[1],
        help="List of num_hidden_layers to try.",
    )
    parser.add_argument(
        "--d-model",
        nargs="+",
        type=int,
        default=[128],
        help="List of d_model dimensions to try.",
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="Number of warmup steps."
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=["snli", "mnli"],
        help="List of training datasets.",
    )
    parser.add_argument(
        "--config",
        default="atomic",
        help="one of the following configs; atomic, nano, micro, milli",
    )
    parser.add_argument(
        "--validation-datasets",
        nargs="+",
        default=["stsb"],
        help="List of validation datasets.",
    )
    parser.add_argument(
        "--train-baseline",
        action="store_true",
        help="Whether to train the baseline Transformer.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to load pretrained configuration",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="if to benchmark on glue",
    )
    parser.add_argument(
        "--dropout-prob", type=float, default=0.2, help="Dropout probability."
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Maximum sequence length."
    )
    parser.add_argument(
        "--vocab-size", type=int, default=30552, help="Vocabulary size."
    )

    args = parser.parse_args()

    for n in args.num_hidden_layers:
        for dim in args.d_model:
            experiment = Experiment(
                args,
                num_hidden_layers=n,
                num_attention_heads=n,
                d_model=dim,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                train_datasets=args.train_datasets,
                validation_datasets=args.validation_datasets,
                train_baseline=args.train_baseline,
                dropout_prob=args.dropout_prob,
                max_seq_len=args.max_seq_len,
                vocab_size=args.vocab_size,
            )

            run(experiment)
            # os.system("tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
