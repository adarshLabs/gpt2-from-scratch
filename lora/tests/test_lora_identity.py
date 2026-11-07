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


def test_lora_scale_is_alpha_over_r():
    linear = nn.Linear(16, 8)
    layer = LoraLayer(linear, r=8, alpha=16)
    assert layer.scale == 2.0


def test_replace_uses_given_r_and_alpha():
    from lora.lora_layer import replace_linear_with_lora

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)

    model = nn.Sequential(Block())
    replace_linear_with_lora(model, ["q_proj"], r=4, alpha=8)
    layer = model[0].q_proj
    assert layer.A.shape[0] == 4 and layer.scale == 2.0
