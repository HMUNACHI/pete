import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
np.random.seed(0)


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Layer Normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class MLP(nn.Module):
    """
    Implements a Multi-Layer Perceptron with GELU activation.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super(MLP, self).__init__()
        self.dense_expansion = nn.Linear(d_model, intermediate_size)
        self.dense_contraction = nn.Linear(intermediate_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_expansion(x)
        x = F.gelu(x)
        x = self.dense_contraction(x)
        return x


class RotaryPositionEncoding(nn.Module):
    """
    Implements Rotary Position Encoding for attention mechanism.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_position_embeddings}"
            )

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        return cos, sin

    def apply_rotary_pos_emb(self, x, cos, sin):
        x_rot = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_rot.unbind(-1)
        x = torch.stack([-x2, x1], dim=-1)
        x = x.reshape(*x.shape[:-2], -1)

        return (x * cos) + (x.roll(shifts=1, dims=-1) * sin)


class Layer(nn.Module):
    """
    Implements a single transformer layer with self-attention and feed-forward network.
    """

    def __init__(
        self,
        d_model,
        num_attention_heads,
        intermediate_size,
        attention_probs_dropout_prob,
        max_position_embeddings,
    ):
        super(Layer, self).__init__()

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(d_model / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln1 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, d_model),
        )
        self.ln2 = RMSNorm(d_model)
        self.rope = RotaryPositionEncoding(
            self.attention_head_size, max_position_embeddings
        )

    def split_heads(self, tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        dot_product = torch.matmul(q, k.transpose(-1, -2))
        scaled_dot_product = dot_product / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask == 1
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scaled_dot_product = torch.where(
                attention_mask, scaled_dot_product, torch.tensor(float("-inf"), device=q.device)
            )

        attention_weights = nn.functional.softmax(scaled_dot_product, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, v)

    def forward(self, x, attention_mask):
        residual = x

        q, k, v = self.query(x), self.key(x), self.value(x)

        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        # Apply RoPE to queries and keys
        cos, sin = self.rope(q, seq_len=x.shape[1])
        q = self.rope.apply_rotary_pos_emb(q, cos, sin)
        k = self.rope.apply_rotary_pos_emb(k, cos, sin)

        attended_outputs = self.attn(q, k, v, attention_mask)
        attended_outputs = self.merge_heads(
            attended_outputs, self.num_attention_heads, self.attention_head_size
        )
        attended_outputs = self.attn_out(attended_outputs)
        attended_outputs = self.dropout(attended_outputs)

        x = self.ln1(attended_outputs + residual)

        residual = x
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.ln2(x + residual)

        return x


class Transformer(nn.Module):
    """
    Implements a Transformer model with embedding layer, multiple transformer layers, and a pooler.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
    ):
        super(Transformer, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.ln = RMSNorm(d_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.layers = nn.ModuleList(
            [
                Layer(
                    d_model,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    max_position_embeddings,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.pooler = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(d_model, d_model)),
                    ("activation", nn.Tanh()),
                ]
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ):
        position_ids = torch.arange(
            input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.embeddings(input_ids)
        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        return (x, self.pooler(x.mean(axis=1)))
