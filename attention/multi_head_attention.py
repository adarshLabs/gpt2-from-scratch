import torch
import torch.nn as nn

try:
    from .scaled_dot_product_attention import ScaledDotProductAttention
except ImportError:
    from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, qk_positional_encoding=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # Per-head dimension D = E / H.
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.qk_positional_encoding = qk_positional_encoding

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout)

    def split_head(self, x):
        # Input x shape: (B, S, E)
        B, S, E = x.shape
        # Reshaped x shape: (B, S, H, D)
        x = x.view(B, S, self.num_heads, self.head_dim)
        # Output x shape: (B, H, S, D)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        # Input x shape: (B, H, S, D)
        B, H, S, D = x.shape

        # Output x shape: (B, S, E)
        x = x.transpose(1, 2)
        x = x.contiguous().view(B, S, self.embed_dim)
        return x

    def forward(self, query, key=None, value=None, mask=None, past_kv=None, position_offset=0):
        # query shape: (B, S_q, E)
        # key/value shapes: (B, S_k, E); default to query for self-attention.
        # mask shape: broadcastable to (B, H, S_q, S_k).

        if key is None:
            key = query
        if value is None:
            value = key

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Split projection shapes: (B, H, S_q, D), (B, H, S_k, D), (B, H, S_k, D)
        Q_split = self.split_head(Q)
        K_split = self.split_head(K)
        V_split = self.split_head(V)

        if self.qk_positional_encoding is not None:
            Q_split, K_split = self.qk_positional_encoding.apply_rotary(
                Q_split, K_split, offset=position_offset
            )

        if past_kv is None:
            K_attn, V_attn = K_split, V_split
            new_kv = (K_split, V_split)
        else:
            # past_kv is a preallocated (buffer_k, buffer_v, write_pos) ring
            # buffer: new tokens are written in place at write_pos % capacity
            # instead of growing a tensor via concatenation every step, so a
            # step's cost stays O(new tokens) rather than O(cached tokens).
            # Callers only ever write a full chunk starting at position 0
            # (prefill) or exactly one token at the tail (decode), so a
            # write never needs to split across the wrap boundary and a
            # plain slice assignment suffices -- no index tensor needed.
            buffer_k, buffer_v, write_pos = past_kv
            capacity = buffer_k.shape[2]
            S_new = K_split.shape[2]
            pos = write_pos % capacity
            buffer_k[:, :, pos : pos + S_new, :] = K_split
            buffer_v[:, :, pos : pos + S_new, :] = V_split

            # Once the buffer wraps, entries are no longer in chronological
            # order; that's only safe because callers pass mask=None here
            # (unmasked decode attends to every cached token regardless of
            # order). A mask-bearing call must supply a fresh, unwrapped
            # buffer (write_pos == 0), as the prefill step does.
            new_write_pos = write_pos + S_new
            valid_len = min(new_write_pos, capacity)
            K_attn = buffer_k[:, :, :valid_len, :]
            V_attn = buffer_v[:, :, :valid_len, :]
            new_kv = (buffer_k, buffer_v, new_write_pos)

        # attn_output shape: (B, H, S_q, D), attn_weights shape: (B, H, S_q, S_k)
        attn_output, attn_weights = self.attention(Q_split, K_attn, V_attn, mask)

        concatenated_output = self.combine_heads(attn_output)
        output = self.out_proj(concatenated_output)

        # output shape: (B, S_q, E)
        return output, attn_weights, new_kv


def main():
    # Demo input shape: (B, S, E)
    x = torch.randn(2, 16, 128)

    mha = MultiHeadAttention(embed_dim=128, num_heads=8)

    output, attn = mha(x)

    assert output.shape == (2, 16, 128)
    assert attn.shape == (2, 8, 16, 16)


if __name__ == "__main__":
    main()