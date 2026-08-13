from pathlib import Path
import sys

import torch
import torch.nn as nn

try:
    from attention.masking import causal_mask, combined_mask, padding_mask
    from positional_encoding.learned_positional_embedding import (
        LearnedPositionalEmbedding,
    )
    from transformer_blocks.transformer_decoder_block import TransformerDecoderBlock
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from attention.masking import causal_mask, combined_mask, padding_mask
    from positional_encoding.learned_positional_embedding import (
        LearnedPositionalEmbedding,
    )
    from transformer_blocks.transformer_decoder_block import TransformerDecoderBlock


def test_transformer_decoder_pipeline_forward_and_backward():
    torch.manual_seed(42)

    batch_size = 2
    target_seq_len = 5
    source_seq_len = 6
    vocab_size = 32
    embed_dim = 16
    num_heads = 4
    padding_token = 0

    target_ids = torch.tensor(
        [
            [5, 7, 9, 0, 0],
            [3, 4, 8, 12, 16],
        ]
    )
    # target_ids shape: (batch_size, target_seq_len)
    source_ids = torch.tensor(
        [
            [2, 6, 10, 14, 0, 0],
            [1, 5, 9, 13, 17, 21],
        ]
    )
    # source_ids shape: (batch_size, source_seq_len)

    target_embedding = nn.Embedding(vocab_size, embed_dim)
    source_embedding = nn.Embedding(vocab_size, embed_dim)
    target_position = LearnedPositionalEmbedding(
        embed_dim=embed_dim, max_seq_len=target_seq_len
    )
    source_position = LearnedPositionalEmbedding(
        embed_dim=embed_dim, max_seq_len=source_seq_len
    )
    decoder_block = TransformerDecoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        expansion_factor=2,
        dropout=0.0,
    )

    target_x = target_position(target_embedding(target_ids))
    encoder_output = source_position(source_embedding(source_ids))
    # target_x shape: (batch_size, target_seq_len, embed_dim)
    # encoder_output shape: (batch_size, source_seq_len, embed_dim)

    self_attention_mask = combined_mask(
        causal_mask(target_seq_len, device=target_ids.device),
        padding_mask(target_ids, padding_token=padding_token),
    )
    cross_attention_mask = padding_mask(source_ids, padding_token=padding_token)
    # self_attention_mask shape: (batch_size, 1, target_seq_len, target_seq_len)
    # cross_attention_mask shape: (batch_size, 1, 1, source_seq_len)

    _, self_attention_weights, _ = decoder_block.self_attention(
        query=decoder_block.norm1(target_x),
        mask=self_attention_mask,
    )
    # self_attention_weights shape: (batch_size, num_heads, target_seq_len, target_seq_len)
    assert self_attention_weights.shape == (
        batch_size,
        num_heads,
        target_seq_len,
        target_seq_len,
    )
    assert torch.all(self_attention_weights[:, :, :, :].triu(1) == 0)
    assert torch.all(self_attention_weights[0, :, :, 3:] == 0)

    _, cross_attention_weights, _ = decoder_block.cross_attention(
        query=decoder_block.norm2(target_x),
        key=encoder_output,
        value=encoder_output,
        mask=cross_attention_mask,
    )
    # cross_attention_weights shape: (batch_size, num_heads, target_seq_len, source_seq_len)
    assert cross_attention_weights.shape == (
        batch_size,
        num_heads,
        target_seq_len,
        source_seq_len,
    )
    assert torch.all(cross_attention_weights[0, :, :, 4:] == 0)

    output = decoder_block(
        target_x,
        encoder_output,
        self_attention_mask,
        cross_attention_mask,
    )
    # output shape: (batch_size, target_seq_len, embed_dim)

    assert output.shape == (batch_size, target_seq_len, embed_dim)
    assert torch.isfinite(output).all()

    loss = output.sum()
    loss.backward()

    assert target_embedding.weight.grad is not None
    assert source_embedding.weight.grad is not None
    assert torch.isfinite(target_embedding.weight.grad).all()
    assert torch.isfinite(source_embedding.weight.grad).all()

    for name, parameter in decoder_block.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"Invalid gradient for {name}"


def main():
    test_transformer_decoder_pipeline_forward_and_backward()
    print("Transformer decoder pipeline forward/backward test passed!")


if __name__ == "__main__":
    main()
