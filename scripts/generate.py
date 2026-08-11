import torch
import tiktoken
import time
import pickle
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_models.gpt2 import GPT2, GPT2Config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tiny_shakespeare")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def sync():

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    ckpt_path = Path("checkpoint") / args.dataset / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: {ckpt_path.resolve()}")

    config = GPT2Config()
    config.__dict__.update(ckpt["config"])
    model = GPT2(config).to(device)
    model.load_state_dict(ckpt["model"])

    meta_path = Path("data/processed") / args.dataset / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    tokenizer = tiktoken.get_encoding(meta["tokenizer"])

    # print("\n=== Model Configuration ===")
    # for key, value in ckpt["config"].items():
    #     print(f"{key:20}: {value}")

    #total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters : {trainable_params:,}")

    

    model.eval()
    with torch.no_grad():
        seed = args.prompt
        query = torch.tensor(tokenizer.encode(seed), dtype=torch.long, device=device)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        torch.manual_seed(0)
        sync()
        t0 = time.perf_counter()
        out1 = model.generate(
            query,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=False,
        )  # without kv cache
        sync()
        t1 = time.perf_counter()

        torch.manual_seed(0)
        sync()
        t2 = time.perf_counter()
        out2 = model.generate(
            query,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=True,
        )  # with kv cache
        sync()
        t3 = time.perf_counter()

        result1 = tokenizer.decode(out1[0].tolist(), errors='replace')
        result2 = tokenizer.decode(out2[0].tolist(), errors='replace')
        print(f"Without cache : {t1-t0:.3f}s")
        print(f"With cache: {t3-t2:.3f}s")
        print(f"Speedup: {(t1-t0)/(t3-t2):.2f}")

        print(f"\n\n---Generated Text 1---\n{result1}")
        print(f"\n\n---Generated Text 2---\n{result2}")




if __name__ == "__main__":
    main()
