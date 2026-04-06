"""PyTorch AR draft model for codebook-0 prediction.

Same architecture as local/draft_mlx.py but in PyTorch for A100 training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim

    def forward(self, x, offset: int = 0):
        T = x.shape[1]
        t = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class DraftAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x, mask=None, offset: int = 0):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, T, H, D) each

        q = self.rope(q.transpose(1, 2), offset=offset).transpose(1, 2)
        k = self.rope(k.transpose(1, 2), offset=offset).transpose(1, 2)

        # (B, T, H, D) -> (B, H, T, D) for SDPA
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class DraftMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        intermediate = hidden_size * 4
        self.gate = nn.Linear(hidden_size, intermediate, bias=False)
        self.up = nn.Linear(hidden_size, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DraftBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.attn = DraftAttention(hidden_size, num_heads, max_seq_len)
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.mlp = DraftMLP(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, mask=None, offset: int = 0):
        x = x + self.dropout(self.attn(self.attn_norm(x), mask=mask, offset=offset))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class DraftModel(nn.Module):
    """Small AR transformer predicting codebook-0 tokens.

    Conditioned on teacher hidden states (projected to draft hidden size),
    then autoregressively predicts the next cb0 token.
    """

    def __init__(
        self,
        vocab_size: int = 1025,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        cond_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.cond_proj = nn.Linear(cond_dim, hidden_size, bias=False)
        self.layers = nn.ModuleList([
            DraftBlock(hidden_size, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, tokens, cond_embeds=None):
        """
        Args:
            tokens: (B, T) codebook-0 token IDs
            cond_embeds: (B, C_len, cond_dim) conditioning from teacher
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embed(tokens)

        if cond_embeds is not None:
            cond = self.cond_proj(cond_embeds)
            x = torch.cat([cond, x], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        if cond_embeds is not None:
            x = x[:, cond_embeds.shape[1]:, :]

        return self.head(x)

    @torch.no_grad()
    def generate_ar(self, cond_embeds, num_tokens, temperature=0.0):
        """Autoregressive generation."""
        self.eval()
        cond = self.cond_proj(cond_embeds)
        generated = []
        current = torch.zeros(1, 1, dtype=torch.long, device=cond.device)

        for _ in range(num_tokens):
            tok_emb = self.embed(current)
            x = torch.cat([cond, tok_emb], dim=1)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.head(x[:, -1:, :])

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs.squeeze(1), 1)
            else:
                next_tok = logits.argmax(dim=-1)

            generated.append(next_tok[0, 0].item())
            current = torch.cat([current, next_tok], dim=1)

        return torch.tensor(generated)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
