import torch.nn as nn
from attention.multi_head_attention import MultiHeadAttention
from positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
from transformer_blocks.feed_forward_network import FeedForwardNetwork


class GPTDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, expansion_factor=4, dropout=0.1):

        super().__init__()
        head_dim = embed_dim // num_heads
        rope = RotaryPositionalEmbedding(head_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, qk_positional_encoding=rope
        )
        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim, expansion_factor=expansion_factor, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, position_offset=0):
        residual = x
        x = self.norm1(x)

        attn_out, _, new_kv = self.self_attention(
            x, mask=mask, past_kv=past_kv, position_offset=position_offset
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x, new_kv
