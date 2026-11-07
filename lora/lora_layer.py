import torch, torch.nn as nn

class LoraLayer(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        self.A = nn.Parameter(torch.randn(r, linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(linear.out_features, r, device=linear.weight.device, dtype=linear.weight.dtype))
        self.scale = alpha / r

    def forward(self, x):
        original = self.linear(x)
        lora = (x @ self.A.T) @ self.B.T
        return original  + self.scale * lora


def replace_linear_with_lora(model, target_substrings, r, alpha):

    replaced = []
    for name, module in model.named_modules():

        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            parts = name.split('.')
            parent = model

            for part in parts[:-1]:
                parent = getattr(parent, part)

            setattr(parent, parts[-1], LoraLayer(module, r=r, alpha=alpha))
            replaced.append(name)

    return replaced

    