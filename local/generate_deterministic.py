"""Deterministic unmasking that matches the PT _run_unmasking_loop exactly.

The key differences from generate_iterative:
- Linear timestep schedule (not shifted)
- Pure argmax token selection (no class_temperature/Gumbel)
- Confidence-based position selection with no position_temperature
- No CFG (single forward pass)
- Simple layer penalty

This produces outputs that are directly comparable to the PT training data,
making them learnable by the v2 draft model.
"""

import math
import mlx.core as mx


def generate_deterministic(
    model,
    input_ids,       # (C, L_total)
    audio_mask,      # (L_total,) bool
    target_len: int,
    num_step: int = 8,
):
    """Deterministic iterative unmasking — matches PT _run_unmasking_loop."""
    config = model.config
    C = config.num_audio_codebook
    mask_id = config.audio_mask_id
    V = config.audio_vocab_size
    L_total = input_ids.shape[1]
    L_cond = L_total - target_len

    # Initialize target with masks
    tokens = mx.full((C, target_len), mask_id, dtype=mx.int32)
    total_mask = target_len * C

    # Layer penalty
    layer_ids = mx.arange(C).reshape(1, C, 1)

    for step in range(num_step):
        # Build full sequence
        full_ids = mx.concatenate([input_ids[:, :L_cond], tokens], axis=1)
        batch_ids = mx.expand_dims(full_ids, axis=0)      # (1, C, L)
        batch_mask = mx.expand_dims(audio_mask, axis=0)    # (1, L)
        attn = mx.ones((1, 1, L_total, L_total), dtype=mx.bool_)

        # Single forward pass (no CFG)
        logits = model(batch_ids, batch_mask, attn)        # (1, C, L, V)
        logits = logits[:, :, -target_len:, :].astype(mx.float32)

        # Mask out the mask token
        logits = logits.at[:, :, :, mask_id].add(mx.array(float("-inf")))

        # Greedy prediction
        pred_tokens = logits.argmax(axis=-1)[0]            # (C, T)

        # Confidence scores (max log prob)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        scores = log_probs.max(axis=-1)[0]                 # (C, T)

        # Layer penalty
        scores = scores - layer_ids.squeeze(0) * 5.0

        # Only score masked positions
        is_masked = tokens == mask_id
        scores = mx.where(is_masked, scores, mx.array(float("-inf")))

        # Linear schedule: how many to unmask this step
        k = math.ceil(total_mask * (step + 1) / num_step) - math.ceil(total_mask * step / num_step)
        k = min(k, int(mx.sum(is_masked).item()))
        if k <= 0:
            continue

        # Top-k selection
        flat_scores = scores.reshape(-1)
        flat_pred = pred_tokens.reshape(-1)
        flat_tokens = tokens.reshape(-1)

        if k >= flat_scores.shape[0]:
            topk_mask = mx.ones(flat_scores.shape[0], dtype=mx.bool_)
        else:
            threshold = mx.sort(flat_scores)[-k]
            topk_mask = flat_scores >= threshold

        flat_tokens = mx.where(topk_mask, flat_pred, flat_tokens)
        tokens = flat_tokens.reshape(C, target_len)
        mx.eval(tokens)

    return tokens
