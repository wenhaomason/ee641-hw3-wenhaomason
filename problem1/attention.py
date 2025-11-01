"""
Attention mechanisms for sequence-to-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor [batch, ..., seq_len_q, d_k]
        K: Key tensor [batch, ..., seq_len_k, d_k]
        V: Value tensor [batch, ..., seq_len_v, d_k]
        mask: Optional mask [batch, ..., seq_len_q, seq_len_k]
              Values: 1 for positions to attend, 0 for positions to mask

    Returns:
        output: Attention output [batch, ..., seq_len_q, d_k]
        attention_weights: Attention weights [batch, ..., seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)

    # TODO: Compute attention scores
    # TODO: Scale scores
    # TODO: Apply mask if provided (use masked_fill to set masked positions to -inf)
    # TODO: Apply softmax
    # TODO: Apply attention to values

    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits d_model into num_heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Initialize linear projections for Q, K, V
        # TODO: Initialize output projection

    def split_heads(self, x):
        """
        Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        # TODO: Reshape and transpose to split heads

        raise NotImplementedError

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor with shape [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, d_k = x.size()

        # TODO: Transpose and reshape to combine heads

        raise NotImplementedError

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # TODO: Linear projections
        # TODO: Split heads
        # TODO: Apply scaled dot-product attention
        # TODO: Combine heads
        # TODO: Apply output projection

        raise NotImplementedError


def create_causal_mask(seq_len, device=None):
    """
    Create causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] lower triangular matrix
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)