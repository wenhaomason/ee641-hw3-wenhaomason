"""
Sequence-to-sequence transformer model for addition task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from attention import MultiHeadAttention, create_causal_mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # TODO: Create positional encoding matrix
        # TODO: Register as buffer (not a parameter)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        # TODO: Add positional encoding to x
        raise NotImplementedError


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # TODO: Initialize two linear layers and dropout

    def forward(self, x):
        """
        Apply feed-forward network.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # TODO: Implement feed-forward with ReLU activation
        raise NotImplementedError


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Consists of multi-head self-attention and position-wise feed-forward,
    with residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # TODO: Initialize components
        # - self-attention
        # - feed-forward
        # - layer normalizations
        # - dropout layers

    def forward(self, x, mask=None):
        """
        Forward pass of encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # TODO: Self-attention with residual
        # TODO: Layer norm
        # TODO: Feed-forward with residual
        # TODO: Layer norm

        raise NotImplementedError


class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Consists of masked self-attention, encoder-decoder attention,
    and position-wise feed-forward, with residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # TODO: Initialize components
        # - masked self-attention
        # - encoder-decoder cross-attention
        # - feed-forward
        # - layer normalizations
        # - dropout layers

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of decoder layer.

        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention (causal)

        Returns:
            Output tensor [batch, tgt_len, d_model]
        """
        # TODO: Masked self-attention with residual
        # TODO: Layer norm
        # TODO: Cross-attention with residual
        # TODO: Layer norm
        # TODO: Feed-forward with residual
        # TODO: Layer norm

        raise NotImplementedError


class Seq2SeqTransformer(nn.Module):
    """
    Full sequence-to-sequence transformer model.
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1,
        max_len=100
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # TODO: Initialize encoder layers
        # TODO: Initialize decoder layers

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """
        Encode source sequence.

        Args:
            src: Source tokens [batch, src_len]
            src_mask: Source padding mask

        Returns:
            Encoder output [batch, src_len, d_model]
        """
        # Embed and scale
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # TODO: Pass through encoder layers

        raise NotImplementedError

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence given encoder output.

        Args:
            tgt: Target tokens [batch, tgt_len]
            encoder_output: Encoder output [batch, src_len, d_model]
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            Decoder output [batch, tgt_len, d_model]
        """
        # Embed and scale
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # TODO: Pass through decoder layers

        raise NotImplementedError

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Full forward pass.

        Args:
            src: Source tokens [batch, src_len]
            tgt: Target tokens [batch, tgt_len]
            src_mask: Source padding mask
            tgt_mask: Target causal mask

        Returns:
            Output logits [batch, tgt_len, vocab_size]
        """
        # TODO: Encode source
        # TODO: Decode target
        # TODO: Project to vocabulary

        raise NotImplementedError

    def generate(self, src, max_len=20, start_token=0):
        """
        Generate output sequence using greedy decoding.

        Args:
            src: Source tokens [batch, src_len]
            max_len: Maximum generation length
            start_token: Start-of-sequence token

        Returns:
            Generated tokens [batch, max_len]
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        with torch.no_grad():
            # Encode source once
            encoder_output = self.encode(src)

            # Initialize target with start token
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

            for _ in range(max_len - 1):
                # Create causal mask for current length
                tgt_mask = create_causal_mask(tgt.size(1), device=device)

                # Decode current sequence
                decoder_output = self.decode(tgt, encoder_output, tgt_mask=tgt_mask)

                # Get next token predictions
                logits = self.output_projection(decoder_output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)

        return tgt