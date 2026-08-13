from pathlib import Path
import sys


import torch

import torch.nn as nn

try:
    from ..multi_head_attention import MultiHeadAttention
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from attention.multi_head_attention import MultiHeadAttention


def copy_weights(custom_mha, torch_mha):
    # PyTorch stores Q/K/V weights together with shape: (3E, E).
    qkv_weights = torch_mha.in_proj_weight
    qkv_bias = torch_mha.in_proj_bias

    # Each projected weight shape: (E, E); each bias shape: (E).
    q_weight, k_weight, v_weight = qkv_weights.chunk(3, dim=0)
    q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
    custom_mha.q_proj.weight.data.copy_(q_weight)
    custom_mha.k_proj.weight.data.copy_(k_weight)
    custom_mha.v_proj.weight.data.copy_(v_weight)

    custom_mha.q_proj.bias.data.copy_(q_bias)
    custom_mha.k_proj.bias.data.copy_(k_bias)
    custom_mha.v_proj.bias.data.copy_(v_bias)

    custom_mha.out_proj.weight.data.copy_(torch_mha.out_proj.weight.data)
    custom_mha.out_proj.bias.data.copy_(torch_mha.out_proj.bias.data)


def test_against_pytorch_mha():
    B = 2
    S = 16
    E = 128
    H = 8
    # Test input shape: (B, S, E)
    x = torch.randn(B, S, E)
    custom_mha = MultiHeadAttention(embed_dim=128, num_heads=8, dropout=0.0)
    torch_mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)

    copy_weights(custom_mha, torch_mha)

    custom_mha.eval()
    torch_mha.eval()

    with torch.no_grad():
        # Output shapes: (B, S, E); attention shapes: (B, H, S, S)
        torch_output, torch_attn = torch_mha(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        custom_output, custom_attn, _ = custom_mha(x)

    assert torch.allclose(torch_output, custom_output, atol=1e-5)
    assert torch.allclose(torch_attn, custom_attn, atol=1e-5)

    print("Custom MHA matches PyTorch MHA!")


def main():
    test_against_pytorch_mha()


if __name__ == "__main__":
    main()
