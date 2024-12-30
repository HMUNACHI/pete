# Tiny Attention Networks (TANs)

This repository implements Tiny Attention Networks (TANs), a lightweight and efficient architecture for text embedding that combines Chebyshev polynomial expansion with attention mechanisms.

## How to use
- `git clone <link>`
- `pip install -r requirements.txt`
- tmux 
- `python main.py --num_hidden_layers_list 128 256 <blah blah>`
- `ctrl + b`, then `d`
- to log back in `tmux attach`

## Architecture Overview

TANs consist of several key components:

### 1. Chebyshev Expansion Layer
- Replaces traditional token embeddings with Chebyshev polynomial expansion
- Uses a fused CUDA kernel for efficient computation
- Projects input tokens into a d_model dimensional space using polynomial basis functions
- Implemented in `ChebyshevBlock` class

### 2. Attention Blocks
Each attention block contains:
- Multi-head attention with decomposed linear projections
- Rotary positional embeddings (RoPE) for position-aware attention
- RMSNorm for stable training
- Residual connections
- Bottleneck MLP with SiLU activation

### 3. Decomposed Linear Layers
- Reduces parameter count by factoring linear transformations
- Projects through a bottleneck dimension (in_features/4)
- Uses SiLU activation between projections
- Applied in attention key/query/value projections and FFN

### 4. Pooling & Normalization
- Mean pooling over sequence dimension
- Final projection and tanh activation for sentence embeddings
- RMSNorm used throughout instead of LayerNorm

## Training

The model is trained using:
- Contrastive learning with InfoNCE loss
- Automatic mixed precision (AMP) for efficient training
- Linear learning rate warmup
- Gradient scaling for stable mixed precision training

### Loss Function
- Normalized temperature-scaled cross entropy
- Learnable temperature parameter
- Bidirectional contrastive loss between anchor and positive pairs

### Evaluation
- Evaluated on STS benchmark using:
  - Pearson correlation
  - Spearman correlation
- Cosine similarity used for sentence similarity scoring

