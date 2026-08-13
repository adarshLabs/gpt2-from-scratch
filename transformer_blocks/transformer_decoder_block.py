import torch
import torch.nn as nn

from attention.multi_head_attention import MultiHeadAttention
from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
from transformer_blocks.feed_forward_network import FeedForwardNetwork


class TransformerDecoderBlock(nn.Module):
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

        self.self_attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout,
            qk_positional_encoding,
        )
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, expansion_factor, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attention_mask, cross_attention_mask):
        # x shape: (B, S_tgt, E)
        # encoder_output shape: (B, S_src, E)
        # self_attention_mask shape: broadcastable to (B, H, S_tgt, S_tgt).
        # cross_attention_mask shape: broadcastable to (B, H, S_tgt, S_src).

        # Self Attention
        residual = x
        x = self.norm1(x)
        # attention_output shape: (B, S_tgt, E)
        attention_output, _, _ = self.self_attention(query=x, mask=self_attention_mask)
        x = residual + self.dropout(attention_output)

        # Cross Attention
        residual = x
        x = self.norm2(x)
        cross_attention_output, _, _ = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=cross_attention_mask,
        )
        # cross_attention_output shape: (B, S_tgt, E)
        x = residual + self.dropout(cross_attention_output)

        # Feed Forward Network
        residual = x
        x = self.norm3(x)
        # ffn_output shape: (B, S_tgt, E)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        # Output shape: (B, S_tgt, E)
        return x
