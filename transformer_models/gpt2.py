import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention.masking import causal_mask, combined_mask, padding_mask
from transformer_blocks.gpt_decoder_block import GPTDecoderBlock


@dataclass
class GPT2Config:
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    vocab_size: int = 50257
    block_size: int = 256
    expansion_factor: int = 4
    dropout: float = 0.1
    padding_token: int = -1


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                GPTDecoderBlock(
                    embed_dim=config.d_model,
                    num_heads=config.n_heads,
                    expansion_factor=config.expansion_factor,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def _build_mask(self, input_ids):
        _, seq_len = input_ids.shape
        device = input_ids.device

        causal = causal_mask(seq_len, device=device)
        padding = padding_mask(input_ids, padding_token=self.config.padding_token)
        return combined_mask(causal, padding)

    def forward(
        self,
        input_ids,
        target=None,
        past_key_values=None,
        use_cache=False,
        position_offset=0,
    ):
        embedding = self.token_embedding(input_ids)
        x = self.drop(embedding)

        # A single new token attending to its (already causal-valid) cache
        # never needs masking; only a multi-token forward -- prefill, or a
        # full-context pass with no cache -- does.
        if input_ids.shape[1] > 1:
            mask = self._build_mask(input_ids)
        else:
            mask = None

        new_kv_values = []
        for i, block in enumerate(self.blocks):
            layer_past_kv = None if past_key_values is None else past_key_values[i]
            x, new_key = block(
                x, mask=mask, past_kv=layer_past_kv, position_offset=position_offset
            )

            if use_cache:
                new_kv_values.append(new_key)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        # print(input_ids)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        if use_cache:
            return logits, loss, new_kv_values
        return logits, loss, None

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=None,
        use_cache=False,
        top_p=None,
    ):
        self.eval()
        context = input_ids[:, -self.config.block_size :]
        position = 0

        if use_cache:
            B = input_ids.shape[0]
            head_dim = self.config.d_model // self.config.n_heads
            buffer_shape = (B, self.config.n_heads, self.config.block_size, head_dim)
            dtype = self.token_embedding.weight.dtype
            past_key_values = [
                (
                    torch.zeros(buffer_shape, device=input_ids.device, dtype=dtype),
                    torch.zeros(buffer_shape, device=input_ids.device, dtype=dtype),
                    0,
                )
                for _ in range(self.config.n_layers)
            ]
        else:
            past_key_values = None

        for _ in range(max_new_tokens):
            if use_cache:
                step_input = context
                position_offset = position
            else:
                step_input = input_ids[:, -self.config.block_size :]
                position_offset = 0

            logits, _, past_key_values = self(
                step_input,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_offset=position_offset,
            )
            position += step_input.shape[1]
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, self.config.vocab_size))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_logits[cumulative_probs - sorted_probs > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            context = next_token if use_cache else input_ids[:, -self.config.block_size :]

        return input_ids


def main():
    config = GPT2Config()
    model = GPT2(config)

    total = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {total}")

    batch_size, seq_len = 2, 32

    input_ids = torch.randint(1, 50257, (batch_size, seq_len))
    input_ids[0, -3:] = 0
    targets = torch.randint(0, 50257, (batch_size, seq_len))

    logits, loss, _ = model(input_ids, targets)

    print(
        f"Logits : {logits.shape}    expected ({batch_size}, {seq_len}, {config.vocab_size})"
    )
    print(f"Loss   : {loss.item():.4f}  expected ~{math.log(config.vocab_size):.2f}")

    seed = torch.randint(1, 50257, (1, 5))
    out = model.generate(
        seed, max_new_tokens=20, temperature=0.8, top_k=50, top_p=0.9, use_cache=True
    )

    print(f"Seed: {seed.shape}, Out: {out.shape}")
    print("All Checks Passed")


if __name__ == "__main__":
    main()
