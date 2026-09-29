import argparse
import os
import time

import torch

from transformer_models.gpt2 import GPT2, GPT2Config
from lora.lora_layer import replace_linear_with_lora
from scripts.train import load_data, validation, train, get_device
import tiktoken
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    # Cross-dataset by design: adapt a TinyStories-pretrained base to Tiny Shakespeare.
    parser.add_argument("--checkpoint", type=str, default="checkpoint/tiny_stories/best.pt")
    parser.add_argument("--max_steps",     type=int,   default=500)    
    parser.add_argument("--max_lr",        type=float, default=5e-4)   
    parser.add_argument("--min_lr",        type=float, default=5e-5)   
    parser.add_argument("--warmup_steps",  type=int,   default=50)    
    parser.add_argument("--save_every",    type=int,   default=100)   
    parser.add_argument("--log_every",     type=int,   default=25)    
    parser.add_argument("--dataset",          default="tiny_shakespeare")
    parser.add_argument("--batch_size",       type=int, default=8)
    parser.add_argument("--block_size",       type=int, default=256)
    parser.add_argument("--num_val_batches",  type=int, default=50)
    parser.add_argument("--max_norm",         type=float, default=1.0)
    parser.add_argument("--resume",           type=str, default=None)   
    parser.add_argument("--lora_r",     type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_k",          type=int,   default=10)
    parser.add_argument("--top_p",          type=float, default=None)
    return parser.parse_args()


def main():

    args = parse_args()
    config = GPT2Config()

    device = get_device()
    print(f"Using device: {device}")
    ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config.__dict__.update(ckpt['config'])

    model = GPT2(config).to(device)
    model.load_state_dict(ckpt['model'])

    path = Path("data/processed")
    train_data, val_data, meta = load_data(path / args.dataset)

    for param in model.parameters():
        param.requires_grad=False
    replaced = replace_linear_with_lora(model, target_substrings=["q_proj", "v_proj"], r = args.lora_r, alpha=args.lora_alpha)

    adapter_bytes = sum(p.numel() for name, p in model.named_parameters() if name.endswith((".A", ".B"))) * 4
    base_checkpoint_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    print(f"Adapter size: {adapter_bytes / (1024 ** 2):.2f} MiB vs base checkpoint: {base_checkpoint_mb:.1f} MB")

    seed = "The"
    tokenizer = tiktoken.get_encoding(meta["tokenizer"])
    query = torch.tensor(tokenizer.encode(seed), dtype=torch.long, device=device)
    if query.dim() == 1:
        query = query.unsqueeze(0)

    prior_val_loss = validation(model, val_data=val_data, device=device, args=args)
    torch.manual_seed(0)
    out1 = model.generate(
        query,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=True,
    )  # with kv cache
    result1 = tokenizer.decode(out1[0].tolist(), errors='replace')

    train_start = time.time()
    train(model, train_data, val_data, tokenizer=tokenizer, device=device, args=args, config=config)
    print(f"Training wall-clock: {(time.time() - train_start) / 60:.1f} min")

    post_val_loss = validation(model, val_data=val_data, device=device, args=args)
    torch.manual_seed(0)
    out2 = model.generate(
        query,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=True,
    )  # with kv cache
    result2 = tokenizer.decode(out2[0].tolist(), errors='replace')

    print(f"Replaced OPS: {replaced}")

    print(f"\n\n---Generated Text 1---\n{result1}")
    print(f"Prior FineTuning Val Loss: {prior_val_loss}")

    print(f"\n\n---Generated Text 2---\n{result2}")
    print(f"Post FineTuning Val Loss: {post_val_loss}")



if __name__=="__main__":
    main()