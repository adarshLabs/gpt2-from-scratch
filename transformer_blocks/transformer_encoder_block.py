import torch
import torch.nn as nn
from attention.multi_head_attention import MultiHeadAttention
from transformer_blocks.feed_forward_network import FeedForwardNetwork
from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        expansion_factor=4,
        dropout=0.1,
        qk_positional_encoding=None,
        use_rotary_positional_encoding=True,
    ):
        super().__init__()

        # Per-head dimension D = E / H.
        head_dim = embed_dim // num_heads
        if qk_positional_encoding is None and use_rotary_positional_encoding:
            qk_positional_encoding = RotaryPositionalEmbedding(head_dim)

        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout, qk_positional_encoding
        )
        self.ffn = FeedForwardNetwork(embed_dim, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # x shape: (B, S, E)
        # attention_mask shape: broadcastable to (B, H, S, S).
        residual = x
        x = self.norm1(x)
        # attention_output shape: (B, S, E)
        attention_output, _, _ = self.attention(query=x, mask=attention_mask)
        x = residual + self.dropout(attention_output)
        x = self.norm2(x)
        residual = x
        # ffn_output shape: (B, S, E)
        ffn_output = self.ffn(residual)
        x = residual + self.dropout(ffn_output)
        # Output shape: (B, S, E)
        return x
