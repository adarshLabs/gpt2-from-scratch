import torch
import torch.nn as nn

from lora.lora_layer import LoraLayer


def test_lora_layer_matches_wrapped_linear_at_init():
    torch.manual_seed(0)

    linear = nn.Linear(16, 8)
    x = torch.randn(4, 16)

    expected = linear(x)

    lora_layer = LoraLayer(linear, r=4, alpha=8)
    actual = lora_layer(x)

    assert torch.equal(actual, expected)
