"""MLX token-conditioned draft model v2.

Uses teacher's embedding weights (loaded from safetensors) to embed
conditioning tokens. Framework-independent conditioning.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DraftV2Config:
    vocab_size: int = 1025
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 2048
    teacher_hidden: int = 1024
    text_vocab_size: int = 151676
    audio_vocab_size: int = 1025
    num_codebooks: int = 8


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self._inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    def __call__(self, x, offset=0):
        T = x.shape[2]  # x is (B, H, T, D)
        t = mx.arange(offset, offset + T, dtype=mx.float32)
        freqs = mx.outer(t, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, None, :, :]  # (1, 1, T, D)
        sin = mx.sin(emb)[None, None, :, :]
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        return x * cos + rotated * sin


class DraftAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RoPE(self.head_dim, config.max_seq_len)

    def __call__(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # (B, T, H, D) -> (B, H, T, D)
        q = self.rope(q.transpose(0, 2, 1, 3))
        k = self.rope(k.transpose(0, 2, 1, 3))
        v = v.transpose(0, 2, 1, 3)

        # Causal mask
        if mask is None:
            mask = mx.triu(mx.full((T, T), -1e9, dtype=q.dtype), k=1)
            mask = mask.reshape(1, 1, T, T)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, mask=mask, scale=1.0 / math.sqrt(self.head_dim)
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out(out)


class DraftMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate = config.hidden_size * 4
        self.gate = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down = nn.Linear(intermediate, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class DraftBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size)
        self.attn = DraftAttention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size)
        self.mlp = DraftMLP(config)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DraftModelV2MLX(nn.Module):
    """Token-conditioned draft model for MLX inference."""

    def __init__(self, config: DraftV2Config):
        super().__init__()
        self.config = config

        # Teacher embeddings (loaded separately)
        self.text_embed = nn.Embedding(config.text_vocab_size, config.teacher_hidden)
        self.audio_embed = nn.Embedding(config.num_codebooks * config.audio_vocab_size, config.teacher_hidden)
        self.codebook_offsets = mx.arange(config.num_codebooks) * config.audio_vocab_size

        # Projection
        self.cond_proj = nn.Linear(config.teacher_hidden, config.hidden_size, bias=False)

        # CB0 embedding
        self.cb0_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer
        self.layers = [DraftBlock(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _embed_conditioning(self, cond_ids, audio_mask):
        """Embed conditioning tokens — mirrors teacher exactly."""
        C = self.config.num_codebooks
        if cond_ids.ndim == 2:
            cond_ids = mx.expand_dims(cond_ids, 0)
        B, C_dim, L = cond_ids.shape

        # Text
        text_emb = self.text_embed(cond_ids[:, 0, :])

        # Audio
        offsets = self.codebook_offsets.reshape(1, C, 1)
        audio_mask_exp = mx.expand_dims(audio_mask, axis=-2) if audio_mask.ndim == 2 else mx.expand_dims(mx.expand_dims(audio_mask, 0), 1)
        if audio_mask_exp.ndim == 2:
            audio_mask_exp = mx.expand_dims(audio_mask_exp, 1)
        shifted = ((cond_ids * audio_mask_exp.astype(mx.int32)) + offsets).astype(mx.int32)
        audio_emb = self.audio_embed(shifted).sum(axis=1)

        mask = mx.expand_dims(audio_mask if audio_mask.ndim == 2 else mx.expand_dims(audio_mask, 0), axis=-1)
        mixed = mx.where(mask, audio_emb, text_emb)

        return self.cond_proj(mixed)

    def __call__(self, cb0_tokens, cond_ids=None, audio_mask=None):
        x = self.cb0_embed(cb0_tokens)

        if cond_ids is not None:
            cond = self._embed_conditioning(cond_ids, audio_mask)
            if cond.ndim == 2:
                cond = mx.expand_dims(cond, 0)
            if x.ndim == 2:
                x = mx.expand_dims(x, 0)
            x = mx.concatenate([cond, x], axis=1)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if cond_ids is not None:
            cond_len = cond_ids.shape[-1] if cond_ids.ndim == 3 else cond_ids.shape[-1]
            x = x[:, cond_len:, :]

        return self.head(x)

    def generate_ar(self, cond_ids, audio_mask, num_tokens, temperature=0.0, start_token=None):
        """Autoregressive generation."""
        cond = self._embed_conditioning(cond_ids, audio_mask)
        if cond.ndim == 2:
            cond = mx.expand_dims(cond, 0)

        generated = []
        if start_token is not None:
            if start_token.ndim == 0:
                start_token = start_token.reshape(1)
            current = start_token.reshape(1, -1)
        else:
            current = mx.zeros((1, 1), dtype=mx.int32)

        for step in range(num_tokens):
            tok_emb = self.cb0_embed(current)
            x = mx.concatenate([cond, tok_emb], axis=1)

            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.head(x[:, -1:, :])

            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_tok = mx.random.categorical(mx.log(probs + 1e-10), axis=-1)
            else:
                next_tok = logits.argmax(axis=-1)

            generated.append(next_tok[0, 0])
            current = mx.concatenate([current, next_tok], axis=1)
            mx.eval(next_tok)

        return mx.stack(generated)
