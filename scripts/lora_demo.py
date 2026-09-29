import argparse

import torch

from transformer_models.gpt2 import GPT2, GPT2Config
from lora.lora_layer import replace_linear_with_lora
from scripts.train import get_device

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="checkpoint/tiny_stories/best.pt")
args = parser.parse_args()

config = GPT2Config()

device = get_device()
ckpt_path = args.checkpoint
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
config.__dict__.update(ckpt['config'])

model = GPT2(config).to(device)
model.load_state_dict(ckpt['model'])


model.eval()


x = torch.tensor([[1,2, 3, 4, 5]], device=device)

with torch.no_grad():   
    output_before, _, _ = model(x)

for param in model.parameters():
    param.requires_grad=False

replaced = replace_linear_with_lora(model, target_substrings=["q_proj", "v_proj"], r = 8, alpha=16)

with torch.no_grad():
    output_after, _, _ = model(x)

max_diff = (output_before - output_after).abs().max().item()

total_parameters = sum(p.numel() for p in model.parameters())

trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad==True)

trainable_pct = 100* trainable_parameters/total_parameters

print(f"Replaced Layers: {replaced}")
print(f"Max_Diff: {max_diff}")
print(f"Trainable Parameters: {trainable_parameters}")
print(f"Total Parameters: {total_parameters}")
print(f"Trainable Parameters % : {trainable_pct}")