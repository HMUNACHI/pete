import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

import chebyshev_extension

torch.manual_seed(0)
np.random.seed(0)


class ChebyshevBlock(nn.Module):
    """Uses fused CUDA kernel for normalization and Chebyshev expansion."""

    def __init__(self, max_seq_len: int, d_model: int):
        super(ChebyshevBlock, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not input_ids.is_cuda:
            raise ValueError("Input tensor must be on CUDA")

        # Call the fused CUDA kernel
        input_ids = input_ids.float()
        embeddings = chebyshev_extension.forward(
            input_ids, self.max_seq_len, self.d_model
        )
        return embeddings[0]


class RotaryPositionEncoding(nn.Module):
    """
    Implements Rotary Position Encoding for attention mechanism.
    """

    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
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


class DecomposedLinear(nn.Module):
    """
    Implements a decomposed linear layer with an intermediate activation.
    """

    def __init__(self, in_features, out_features):
        super(DecomposedLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features // 4)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(in_features // 4, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


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


class AttentionBlock(nn.Module):
    """
    Implements a multi-head attention block with feed-forward network.
    """

    def __init__(
        self,
        d_model,
        num_attention_heads,
        max_seq_len,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(AttentionBlock, self).__init__()

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert self.d_model % self.num_attention_heads == 0
        self.attention_head_size = int(d_model / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = DecomposedLinear(d_model, self.all_head_size)
        self.key = DecomposedLinear(d_model, self.all_head_size)
        self.value = DecomposedLinear(d_model, self.all_head_size)
        
        self.low_rank_factor = 4
        self.key_low_rank = nn.Linear(d_model, d_model // self.low_rank_factor)
        self.value_low_rank = nn.Linear(d_model, d_model // self.low_rank_factor)
        self.recovery = nn.Linear(d_model // self.low_rank_factor , d_model) 
        

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.attn_out = DecomposedLinear(d_model, d_model)
        self.norm1 = RMSNorm(d_model)
        self.bottleneck_mlp = DecomposedLinear(d_model, d_model)
        self.norm2 = RMSNorm(d_model)
        self.rope = RotaryPositionEncoding(self.attention_head_size, max_seq_len)

    def split_heads(self, tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        k_low_rank = self.key_low_rank(k)
        v_low_rank = self.key_low_rank(v)
        dot_product = torch.matmul(q, k_low_rank.transpose(-1, -2))
        scaled_dot_product = dot_product / math.sqrt(self.attention_head_size // self.low_rank_factor)

        if attention_mask is not None:
            attention_mask = attention_mask == 1
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scaled_dot_product = torch.where(
                attention_mask, scaled_dot_product, torch.tensor(float("-inf"))
            )

        attention_weights = nn.functional.softmax(scaled_dot_product, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attended_outputs = torch.matmul(attention_weights, v_low_rank)
        return self.recovery(attended_outputs)

    def forward(self, x, attention_mask):
        residual = x

        q, k, v = self.query(x), self.key(x), self.value(x)
        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        cos, sin = self.rope(q, seq_len=x.shape[1])
        q = self.rope.apply_rotary_pos_emb(q, cos, sin)
        k = self.rope.apply_rotary_pos_emb(k, cos, sin)

        attended_outputs = self.attn(q, k, v, attention_mask)
        attended_outputs = self.merge_heads(
            attended_outputs, self.num_attention_heads, self.attention_head_size
        )
        attended_outputs = self.attn_out(attended_outputs)
        attended_outputs = self.dropout(attended_outputs)

        x = self.norm1(attended_outputs + residual)

        residual = x
        x = self.bottleneck_mlp(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)

        return x


class TAN(nn.Module):
    """
    Main TAN class that combines all components.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_hidden_layers,
        num_attention_heads,
        max_seq_len,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
    ):
        super(TAN, self).__init__()

        self.pre_norm = RMSNorm(d_model)
        self.pre_mlp = DecomposedLinear(d_model, d_model)
        self.expansion = ChebyshevBlock(vocab_size, d_model)
        self.mlp = DecomposedLinear(d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    d_model,
                    num_attention_heads,
                    max_seq_len,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.pooler = nn.Sequential(
            OrderedDict(
                [
                    ("dense", DecomposedLinear(d_model, d_model)),
                    ("activation", nn.Tanh()),
                ]
            )
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        x = self.pre_mlp(self.pre_norm(input_ids)) 
        x_polynomials = self.expansion(x) 
        x = self.norm(self.mlp(x_polynomials) + x_polynomials)
        x = self.dropout(x)

        for AttentionBlock in self.layers:
            x = AttentionBlock(x, attention_mask)

        return (x, self.pooler(x.mean(axis=1)))
