"""Split validation loss into structure tokens and content tokens, before vs after LoRA.

Structure = newline tokens plus the tokens of short speaker-label lines ending in ':'.
Usage: PYTHONPATH=. python scripts/split_loss.py checkpoint/tiny_shakespeare/final.pt
"""
import torch, numpy as np, tiktoken, pickle, sys
import torch.nn.functional as F
from pathlib import Path
from transformer_models.gpt2 import GPT2, GPT2Config
from lora.lora_layer import replace_linear_with_lora
dev = "cuda" if torch.cuda.is_available() else "cpu"
base = torch.load("checkpoint/tiny_stories/best.pt", map_location=dev, weights_only=False)
lcp = sys.argv[1]
lor = torch.load(lcp, map_location=dev, weights_only=False)
cfg = GPT2Config(); cfg.__dict__.update(base["config"])
val = np.memmap("data/processed/tiny_shakespeare/val.bin", dtype=np.uint16, mode="r")
meta = pickle.load(open("data/processed/tiny_shakespeare/meta.pkl", "rb"))
enc = tiktoken.get_encoding(meta["tokenizer"])
B = 256
n = (len(val) - 1) // B
X = torch.tensor(np.stack([val[i*B:i*B+B] for i in range(n)]).astype(np.int64))
Y = torch.tensor(np.stack([val[i*B+1:i*B+B+1] for i in range(n)]).astype(np.int64))
toks = Y.flatten().tolist()
strs = [enc.decode_single_token_bytes(t).decode("utf-8", "replace") for t in toks]
struct = np.zeros(len(toks), bool)
line = []
for i, s in enumerate(strs):
    if "\n" in s:
        struct[i] = True
        txt = "".join(strs[j] for j in line).strip()
        if txt.endswith(":") and len(line) <= 6:
            for j in line: struct[j] = True
        line = []
    else:
        line.append(i)
def losses(model):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, n, 16):
            x = X[i:i+16].to(dev); y = Y[i:i+16].to(dev)
            o = model(x, y)
            lg = o[0]
            out.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), y.reshape(-1), reduction="none").cpu())
    return torch.cat(out).numpy()
m0 = GPT2(cfg).to(dev); m0.load_state_dict(base["model"])
l0 = losses(m0)
m1 = GPT2(cfg).to(dev)
for p in m1.parameters(): p.requires_grad = False
replace_linear_with_lora(m1, ["q_proj", "v_proj"], r=8, alpha=16)
m1.load_state_dict(lor["model"])
l1 = losses(m1)
tot0, tot1 = l0.sum(), l1.sum()
print("tokens", len(l0), "overall loss", round(l0.mean(), 3), "->", round(l1.mean(), 3))
for name, mask in (("structure", struct), ("content", ~struct)):
    c = int(mask.sum())
    a, b = l0[mask].mean(), l1[mask].mean()
    share = (l0[mask].sum() - l1[mask].sum()) / (tot0 - tot1)
    print(name, "tokens", c, "(", round(100*c/len(l0), 1), "% )", "loss", round(a, 3), "->", round(b, 3), "drop", round(a-b, 3), "share of total drop", round(100*share, 1), "%")
