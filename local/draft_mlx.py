"""MLX draft model — small AR transformer for codebook-0 prediction.

Runs on Apple Silicon for local validation. Same architecture as the
PyTorch version in src/draft_model.py, just in MLX.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DraftMLXConfig:
    vocab_size: int = 1025
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 512
    cond_dim: int = 1024  # teacher hidden size


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self._inv_freq = inv_freq

    def __call__(self, x, offset: int = 0):
        T = x.shape[1]
        t = mx.arange(offset, offset + T, dtype=mx.float32)
        freqs = mx.outer(t, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, :, None, :]
        sin = mx.sin(emb)[None, :, None, :]
        # Rotate pairs
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin


class DraftAttention(nn.Module):
    def __init__(self, config: DraftMLXConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RoPE(self.head_dim, config.max_seq_len)

    def __call__(self, x, mask=None, offset: int = 0):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # (B, T, H, D) -> (B, H, T, D) for RoPE and sdpa
        q = self.rope(q.transpose(0, 2, 1, 3), offset=offset)  # (B, H, T, D)
        k = self.rope(k.transpose(0, 2, 1, 3), offset=offset)  # (B, H, T, D)
        v = v.transpose(0, 2, 1, 3)  # (B, H, T, D)

        # Causal mask: (1, 1, T, T) broadcasts to (B, H, T, T)
        if mask is None:
            mask = mx.triu(mx.full((T, T), -1e9, dtype=q.dtype), k=1)
            mask = mask.reshape(1, 1, T, T)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0 / math.sqrt(self.head_dim))
        # (B, H, T, D) -> (B, T, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out(out)


class DraftMLP(nn.Module):
    def __init__(self, config: DraftMLXConfig):
        super().__init__()
        intermediate = config.hidden_size * 4
        self.gate = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down = nn.Linear(intermediate, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class DraftBlock(nn.Module):
    def __init__(self, config: DraftMLXConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size)
        self.attn = DraftAttention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size)
        self.mlp = DraftMLP(config)

    def __call__(self, x, mask=None, offset: int = 0):
        x = x + self.attn(self.attn_norm(x), mask=mask, offset=offset)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DraftModelMLX(nn.Module):
    """Small AR transformer that predicts codebook-0 tokens.

    Input: conditioning embeddings (from teacher) + previous cb0 tokens
    Output: next-token logits over vocab_size=1025
    """

    def __init__(self, config: DraftMLXConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cond_proj = nn.Linear(config.cond_dim, config.hidden_size, bias=False)
        self.layers = [DraftBlock(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, tokens, cond_embeds=None, mask=None):
        """
        Args:
            tokens: (B, T) codebook-0 token IDs
            cond_embeds: (B, C_len, teacher_hidden) conditioning from teacher
            mask: causal attention mask
        Returns:
            logits: (B, T, vocab_size) — next-token predictions
        """
        x = self.embed(tokens)  # (B, T, H)

        if cond_embeds is not None:
            cond = self.cond_proj(cond_embeds)  # (B, C_len, H)
            x = mx.concatenate([cond, x], axis=1)  # (B, C_len + T, H)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        # Only return logits for token positions (not conditioning)
        if cond_embeds is not None:
            x = x[:, cond_embeds.shape[1]:, :]

        return self.head(x)

    def generate_ar(self, cond_embeds, num_tokens, temperature=0.0, start_token=None):
        """Autoregressive generation for inference.

        Args:
            cond_embeds: (1, C_len, teacher_hidden) conditioning
            num_tokens: how many cb0 tokens to generate
            temperature: 0 = greedy, >0 = sampling
            start_token: (1,) or (1,1) optional seed token
        Returns:
            tokens: (num_tokens,) generated token IDs
        """
        cond = self.cond_proj(cond_embeds)  # (1, C_len, H)
        C_len = cond.shape[1]

        generated = []
        if start_token is not None:
            if start_token.ndim == 1:
                start_token = start_token.reshape(1, 1)
            current = start_token
        else:
            current = mx.zeros((1, 1), dtype=mx.int32)

        for step in range(num_tokens):
            tok_emb = self.embed(current)  # (1, step+1, H)
            x = mx.concatenate([cond, tok_emb], axis=1)

            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)

            # Logits for last position only
            logits = self.head(x[:, -1:, :])  # (1, 1, V)

            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_tok = mx.random.categorical(mx.log(probs + 1e-10), axis=-1)
            else:
                next_tok = logits.argmax(axis=-1)  # (1, 1)

            generated.append(next_tok[0, 0])
            current = mx.concatenate([current, next_tok], axis=1)
            mx.eval(next_tok)

        return mx.stack(generated)
