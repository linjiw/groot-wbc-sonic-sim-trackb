"""Tiny decoder-only transformer (~3M params, matching the paper's maze model)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from maze_env import VOCAB_SIZE, PAD


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 6, n_heads: int = 8,
                 max_len: int = 512):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.max_len = max_len

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) -> logits (B, L, V), causal."""
        B, L = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        mask = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool),
                          diagonal=1)
        pad_mask = ids == PAD
        x = self.blocks(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, prompts: torch.Tensor, prompt_lens: torch.Tensor,
                 max_new: int, eos: int, temperature: float = 1.0) -> torch.Tensor:
        """Batched sampling. prompts: (B, Lp) right-padded. Returns (B, max_new)
        response tokens right-padded with PAD after EOS."""
        B = prompts.shape[0]
        device = prompts.device
        ids = torch.full((B, prompts.shape[1] + max_new), PAD, dtype=torch.long,
                         device=device)
        ids[:, :prompts.shape[1]] = prompts
        cur = prompt_lens.clone()          # next write position per row
        alive = torch.ones(B, dtype=torch.bool, device=device)
        for _ in range(max_new):
            L = int(cur.max())
            logits = self.forward(ids[:, :L])
            step_logits = logits[torch.arange(B, device=device), cur - 1]
            probs = F.softmax(step_logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(1)
            nxt = torch.where(alive, nxt, torch.full_like(nxt, PAD))
            ids[torch.arange(B, device=device), cur] = nxt
            cur = cur + alive.long()
            alive = alive & (nxt != eos)
            if not alive.any():
                break
        # slice out responses
        resp = torch.full((B, max_new), PAD, dtype=torch.long, device=device)
        for b in range(B):
            s = int(prompt_lens[b])
            e = int(cur[b])
            resp[b, :e - s] = ids[b, s:e]
        return resp
