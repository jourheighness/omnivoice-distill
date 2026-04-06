"""Token-conditioned draft model v2.

Key difference from v1: conditioned on raw input tokens, not LLM hidden states.
Uses teacher's embedding weights (frozen) so the conditioning is identical
across PyTorch and MLX — no framework numerical gap.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from draft_model import RoPE, DraftAttention, DraftMLP, DraftBlock


class DraftModelV2(nn.Module):
    """AR draft model conditioned on raw tokens, not hidden states.

    Embeds conditioning tokens using the teacher's embedding weights (frozen),
    projects to draft dimension, then autoregressively predicts cb0 tokens.
    """

    def __init__(
        self,
        vocab_size: int = 1025,       # cb0 prediction vocab
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        teacher_hidden: int = 1024,
        text_vocab_size: int = 151676,
        audio_vocab_size: int = 1025,
        num_codebooks: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.audio_vocab_size = audio_vocab_size

        # Teacher embeddings (frozen) — loaded from teacher weights
        self.text_embed = nn.Embedding(text_vocab_size, teacher_hidden)
        self.audio_embed = nn.Embedding(num_codebooks * audio_vocab_size, teacher_hidden)

        # Codebook offsets
        self.register_buffer(
            "codebook_offsets",
            torch.arange(num_codebooks) * audio_vocab_size,
        )

        # Project teacher embedding dim → draft dim
        self.cond_proj = nn.Linear(teacher_hidden, hidden_size, bias=False)

        # CB0 token embedding for AR prediction
        self.cb0_embed = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            DraftBlock(hidden_size, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying for cb0
        self.head.weight = self.cb0_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "text_embed" in name or "audio_embed" in name:
                continue  # skip teacher embeddings
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def load_teacher_embeddings(self, teacher_weights_path: str, device=None):
        """Load embedding weights from teacher safetensors."""
        from safetensors.torch import load_file
        weights = load_file(teacher_weights_path, device=str(device) if device else "cpu")

        # Text embedding
        text_key = "llm.embed_tokens.weight"
        if text_key in weights:
            self.text_embed.weight.data = weights[text_key].float()
            self.text_embed.weight.requires_grad = False
            print(f"  Loaded text embedding: {self.text_embed.weight.shape}")

        # Audio embedding
        audio_key = "audio_embeddings.weight"
        if audio_key in weights:
            self.audio_embed.weight.data = weights[audio_key].float()
            self.audio_embed.weight.requires_grad = False
            print(f"  Loaded audio embedding: {self.audio_embed.weight.shape}")

    def _embed_conditioning(self, cond_ids, audio_mask):
        """Embed conditioning tokens using teacher's embeddings.

        Mirrors teacher's _prepare_embed_inputs exactly:
        text tokens → text_embed, audio tokens → sum of codebook embeds.

        Args:
            cond_ids: (B, C, L_cond) int — raw conditioning token IDs
            audio_mask: (B, L_cond) bool — True where audio
        Returns:
            (B, L_cond, hidden_size)
        """
        B, C, L = cond_ids.shape

        # Text path: embed from codebook 0
        text_emb = self.text_embed(cond_ids[:, 0, :])  # (B, L, teacher_H)

        # Audio path: shift IDs per codebook, embed, sum
        offsets = self.codebook_offsets.reshape(1, C, 1)
        audio_mask_exp = audio_mask.unsqueeze(1)  # (B, 1, L)
        shifted = (cond_ids * audio_mask_exp.long()) + offsets
        audio_emb = self.audio_embed(shifted).sum(dim=1)  # (B, L, teacher_H)

        # Mix based on audio_mask
        mask = audio_mask.unsqueeze(-1)  # (B, L, 1)
        mixed = torch.where(mask, audio_emb, text_emb)  # (B, L, teacher_H)

        # Project to draft dimension
        return self.cond_proj(mixed)  # (B, L, hidden)

    def forward(self, cb0_tokens, cond_ids=None, audio_mask=None):
        """
        Args:
            cb0_tokens: (B, T) codebook-0 token IDs
            cond_ids: (B, C, L_cond) conditioning token IDs
            audio_mask: (B, L_cond) bool
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.cb0_embed(cb0_tokens)  # (B, T, H)

        if cond_ids is not None:
            cond = self._embed_conditioning(cond_ids, audio_mask)  # (B, L_cond, H)
            x = torch.cat([cond, x], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        if cond_ids is not None:
            x = x[:, cond_ids.shape[2]:, :]  # skip conditioning

        return self.head(x)

    @torch.no_grad()
    def generate_ar(self, cond_ids, audio_mask, num_tokens, temperature=0.0):
        """Autoregressive generation."""
        self.eval()
        cond = self._embed_conditioning(cond_ids, audio_mask)

        generated = []
        current = torch.zeros(1, 1, dtype=torch.long, device=cond.device)

        for _ in range(num_tokens):
            tok_emb = self.cb0_embed(current)
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

    def count_params(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
