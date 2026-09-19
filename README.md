# GPT-2 From Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)

An educational PyTorch implementation of a GPT-style, decoder-only transformer. The repository keeps the important mechanics in plain sight: custom attention, masking, positional encodings, pre-norm decoder blocks, training, checkpointing, and autoregressive sampling with optional KV caching.

It is built for reading, experimenting, and modifying. Alongside the language-model path, the project includes standalone encoder and encoder-decoder transformer reference modules for studying the broader architecture.

> **Scope.** This is a GPT-2-inspired teaching implementation, not a drop-in reproduction of OpenAI's pretrained GPT-2 models. In particular, the language model uses rotary positional embeddings (RoPE) rather than GPT-2's learned absolute position embeddings, and this repository does not ship pretrained weights.

## Highlights

| Area | Included |
| --- | --- |
| Attention | From-scratch scaled dot-product attention, multi-head self-attention, and cross-attention |
| Masking | Causal, padding, and combined attention masks |
| Position information | Rotary, learned, and sinusoidal positional encodings |
| Decoder-only LM | Pre-norm GPT decoder blocks, GELU MLPs, tied input/output embeddings, and a final language-model head |
| Generation | Temperature, top-k, and top-p sampling; optional KV caching for incremental decoding |
| Training | GPT-2 BPE tokenization, memory-mapped token files, AdamW, cosine learning-rate decay, warmup, gradient clipping, TensorBoard, and checkpoints |
| Learning modules | Independent encoder and encoder-decoder transformer components with focused tests |
| LoRA | Rank-8, alpha-16 low-rank adapters on the `q_proj`/`v_proj` projections; 294,912 of 123,998,545 parameters trainable (0.24%), with the wrapped model numerically identical to the base checkpoint before any training |

## Architecture

The primary model lives in [`transformer_models/gpt2.py`](transformer_models/gpt2.py):

```text
token IDs
  -> token embedding
  -> dropout
  -> [LayerNorm -> masked multi-head attention + RoPE -> residual
      LayerNorm -> GELU feed-forward network       -> residual] x N
  -> final LayerNorm
  -> tied language-model head
  -> next-token logits
```

The `GPT2` model ties the token embedding and output projection weights. Its decoder blocks use a causal mask during full-context evaluation and can retain per-layer key/value tensors during generation to avoid recomputing the entire prompt at every step.

## Repository Layout

```text
attention/                         # attention primitives and masks
  masking.py
  scaled_dot_product_attention.py
  multi_head_attention.py

positional_encoding/               # RoPE, learned, and sinusoidal variants
transformer_blocks/                # feed-forward, GPT, encoder, and decoder blocks
transformer_models/                # GPT-style model and encoder reference model
scripts/
  preprocess.py                    # creates tokenized train/validation files
  train.py                         # trains and checkpoints the GPT-style model
  generate.py                      # samples from a checkpoint and compares KV caching
tokenizer/character_tokenizer.py   # minimal character-tokenizer reference
data/raw/tiny_shakespeare.txt      # bundled sample corpus
tests/                             # generation and CLI coverage
```

## Requirements

- Python 3.8 or newer
- PyTorch 1.13 or newer
- A CUDA GPU, Apple Silicon (MPS), or CPU. The training and generation scripts choose CUDA first, then MPS, then CPU.

All Python dependencies are listed in [`requirements.txt`](requirements.txt).

## Installation

```bash
git clone https://github.com/adarshLabs/gpt2-from-scratch.git
cd gpt2-from-scratch

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run the commands below from the repository root. The scripts use repository-relative paths for datasets, checkpoints, and TensorBoard logs.

## Quick Start

### 1. Preprocess Tiny Shakespeare

The bundled Tiny Shakespeare corpus is split 90/10, tokenized with the GPT-2 BPE tokenizer from `tiktoken`, and written as `uint16` memory-mapped arrays.

```bash
python scripts/preprocess.py --dataset tiny_shakespeare
```

This creates:

```text
data/processed/tiny_shakespeare/
  train.bin
  val.bin
  meta.pkl
```

### 2. Run a small training smoke test

This deliberately small configuration is useful for checking the end-to-end path before committing to a longer run:

```bash
python scripts/train.py \
  --dataset tiny_shakespeare \
  --n_layers 2 \
  --n_heads 2 \
  --d_model 128 \
  --block_size 64 \
  --batch_size 4 \
  --max_steps 25 \
  --num_val_batches 5 \
  --log_every 5 \
  --save_every 25
```

For the repository's standard Tiny Shakespeare configuration, use the defaults:

```bash
python scripts/train.py --dataset tiny_shakespeare --max_steps 2000
```

The default model has 6 layers, 6 attention heads, a model width of 384, and a context window of 256 tokens. A GPU or MPS device is strongly recommended for the longer run.

### 3. Generate text

`final.pt` is written at the end of training, so it is the most reliable checkpoint name for a first run:

```bash
python scripts/generate.py \
  --dataset tiny_shakespeare \
  --checkpoint final.pt \
  --prompt "ROMEO:" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.9
```

The generation script samples once without a cache and once with KV caching, then prints both outputs and their relative timings. It uses the same random seed for both paths.

## Datasets

| Dataset | Source | Notes |
| --- | --- | --- |
| `tiny_shakespeare` | Bundled in `data/raw/` | Good for quick experiments and verifying the workflow. |
| `tiny_stories` | Hugging Face `roneneldan/TinyStories` | Downloaded through `datasets` on first preprocessing run; substantially more demanding. |

To prepare TinyStories:

```bash
python scripts/preprocess.py --dataset tiny_stories
```

For a larger corpus, increase model capacity and tune the batch size for available memory. This is a starting point rather than a benchmark recipe:

```bash
python scripts/train.py \
  --dataset tiny_stories \
  --n_layers 12 \
  --n_heads 12 \
  --d_model 768 \
  --expansion_factor 4 \
  --batch_size 24 \
  --block_size 256 \
  --max_steps 10000 \
  --warmup_steps 300 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --max_norm 1.0 \
  --log_every 100 \
  --save_every 1000 \
  --dropout 0.0
```

## Training Configuration

The training script exposes the architecture and optimization controls directly.

| Group | Arguments | Defaults |
| --- | --- | --- |
| Model | `--n_layers`, `--n_heads`, `--d_model`, `--expansion_factor`, `--dropout` | `6`, `6`, `384`, `4`, `0.1` |
| Batching | `--batch_size`, `--block_size`, `--num_val_batches` | `8`, `256`, `50` |
| Schedule | `--max_steps`, `--warmup_steps`, `--max_lr`, `--min_lr` | `2000`, `100`, `3e-4`, `3e-5` |
| Stability | `--max_norm` | `1.0` |
| Logging | `--log_every`, `--save_every` | `100`, `500` |

Training uses AdamW with weight decay, linear warmup followed by cosine decay, next-token cross-entropy, gradient clipping, and periodic validation.

### Checkpoints and TensorBoard

For each dataset, training writes:

```text
checkpoint/<dataset>/
  latest.pt       # written at every save interval
  best.pt         # written when validation loss improves
  final.pt        # written after training completes

runs/<dataset>/   # TensorBoard event files
```

Resume from the latest periodic checkpoint:

```bash
python scripts/train.py \
  --dataset tiny_shakespeare \
  --resume checkpoint/tiny_shakespeare/latest.pt
```

Inspect the recorded loss, perplexity, learning rate, and gradient norm with TensorBoard:

```bash
tensorboard --logdir runs
```

## Sampling Controls

[`scripts/generate.py`](scripts/generate.py) accepts the following useful options:

| Argument | Purpose |
| --- | --- |
| `--prompt` | Text used to seed generation. |
| `--max_new_tokens` | Number of tokens to sample after the prompt. |
| `--temperature` | Scales logits before sampling; lower values are more conservative. |
| `--top_k` | Restricts sampling to the k highest-logit tokens. |
| `--top_p` | Applies nucleus sampling over the smallest probability mass that reaches p. |
| `--checkpoint` | Checkpoint filename under `checkpoint/<dataset>/`. |

The model keeps only the most recent `block_size` tokens as context. With `use_cache=True` internally, generation reuses attention keys and values while preserving that sliding window: each key is rotated once with RoPE at its true absolute position and cached, so evicting old entries from the window never requires re-rotating anything that's still cached. Output matches the non-cached path exactly for the first `block_size` tokens, and diverges only slightly afterward, since the two paths define the sliding window differently once older tokens fall out of it (recomputed with a shared window boundary vs. cached with each token's own).

## Learning Guide

For the decoder-only language-model path, a productive reading order is:

1. [`scripts/preprocess.py`](scripts/preprocess.py) - see how text becomes GPT-2 BPE token IDs.
2. [`transformer_models/gpt2.py`](transformer_models/gpt2.py) - follow model construction, masking, loss calculation, and sampling.
3. [`transformer_blocks/gpt_decoder_block.py`](transformer_blocks/gpt_decoder_block.py) - inspect one pre-norm transformer block.
4. [`attention/multi_head_attention.py`](attention/multi_head_attention.py) and [`attention/scaled_dot_product_attention.py`](attention/scaled_dot_product_attention.py) - trace Q/K/V projection and attention scores.
5. [`positional_encoding/rotary_positional_embedding.py`](positional_encoding/rotary_positional_embedding.py) - see how RoPE rotates queries and keys.
6. [`scripts/train.py`](scripts/train.py) and [`scripts/generate.py`](scripts/generate.py) - connect the model to the full training and inference workflows.

The broader reference path adds [`transformer_models/transformer_encoder.py`](transformer_models/transformer_encoder.py), [`transformer_blocks/transformer_encoder_block.py`](transformer_blocks/transformer_encoder_block.py), and [`transformer_blocks/transformer_decoder_block.py`](transformer_blocks/transformer_decoder_block.py), plus learned and sinusoidal positional encodings.

## Tests

Run the test suite from the repository root:

```bash
PYTHONPATH=. python -m pytest
```

The suite covers padding masks, custom multi-head attention parity against PyTorch, transformer-block shape and pipeline checks, generation beyond the context window, and training CLI parsing.

## Project Scope and Limitations

- The project is optimized for clarity and experimentation, not for reproducing pretrained GPT-2 quality or production inference performance.
- Training is single-process and does not currently include mixed precision, distributed training, or dataset sharding.
- The preprocessing format is `uint16`, which suits the bundled GPT-2 tokenizer vocabulary. A tokenizer with more than 65,535 token IDs would require a wider storage dtype.
- The training path uses fixed-length contiguous token windows; the padding-mask support is primarily included for the reusable transformer modules.

## Contributing

Contributions are welcome. Please keep changes focused, add or update tests where appropriate, and include the relevant test command and result in your pull request.

## License

This project is released under the [MIT License](LICENSE). Copyright (c) 2026 Adarsh Tiwari.
