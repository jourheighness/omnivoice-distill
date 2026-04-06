"""Speculative decoding inference — combines draft + teacher for streaming.

This is the final inference pipeline:
1. Draft model generates cb0 tokens autoregressively (fast, streamable)
2. Teacher model fills in cb1-7 via unmasking (quality)
3. Optional: teacher verifies/corrects cb0 predictions

For RunPod evaluation and eventual deployment.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from draft_model import DraftModel


@dataclass
class SpecDecodeResult:
    """Result from speculative generation of one chunk."""
    tokens: np.ndarray           # (C, target_len) final audio tokens
    draft_cb0: np.ndarray        # (target_len,) draft's cb0 predictions
    teacher_cb0: np.ndarray      # (target_len,) teacher's cb0 (ground truth)
    acceptance_rate: float       # fraction of cb0 tokens matching teacher
    draft_ms: float              # draft generation time
    teacher_ms: float            # teacher verification time
    total_ms: float


class SpeculativeDecoder:
    """Combines a fast AR draft with the OmniVoice teacher for streaming inference.

    The draft handles streaming (predicts cb0 left-to-right at 25Hz),
    while the teacher handles quality (fills cb1-7 and optionally corrects cb0).
    """

    def __init__(self, draft_model, teacher_model, teacher_config, device="cuda"):
        self.draft = draft_model
        self.teacher = teacher_model
        self.config = teacher_config
        self.device = device

    def generate_chunk(
        self,
        cond_hidden,        # (1, L_cond, H) teacher conditioning embeddings
        target_len: int,
        teacher_input_ids,  # (C, L_total) for teacher with mask region
        teacher_audio_mask, # (L_total,)
        draft_temperature: float = 0.0,
        teacher_steps: int = 4,
        guidance_scale: float = 3.0,
    ) -> SpecDecodeResult:
        """Generate one chunk using speculative decoding.

        Flow:
        1. Draft generates target_len cb0 tokens (AR, fast)
        2. Teacher runs unmasking with cb0 hints -> gets all codebooks
        3. Compare draft cb0 vs teacher cb0 for acceptance rate
        """
        # Step 1: Draft generates cb0
        t0 = time.perf_counter()

        if isinstance(cond_hidden, np.ndarray):
            cond_hidden = torch.from_numpy(cond_hidden).unsqueeze(0).to(self.device)

        draft_cb0 = self.draft.generate_ar(
            cond_hidden, num_tokens=target_len, temperature=draft_temperature,
        )
        t_draft = time.perf_counter()

        # Step 2: Teacher verification
        # In the full pipeline, you'd pre-fill cb0 in the teacher's input
        # and run fewer unmasking steps. For now, run teacher normally
        # and compare cb0 for acceptance measurement.

        # NOTE: This is a placeholder — the actual teacher integration
        # depends on whether you're running PyTorch or MLX teacher.
        # The MLX version in local/test_speculative.py shows the full flow.

        # For RunPod with PyTorch teacher, you'd do:
        # teacher_tokens = teacher.generate(teacher_input_ids, teacher_audio_mask, ...)
        # For now, return draft tokens as-is with dummy teacher comparison

        t_teacher = time.perf_counter()

        draft_np = draft_cb0.cpu().numpy() if isinstance(draft_cb0, torch.Tensor) else np.array(draft_cb0)

        return SpecDecodeResult(
            tokens=None,  # filled by caller with full teacher output
            draft_cb0=draft_np,
            teacher_cb0=draft_np,  # placeholder until teacher integration
            acceptance_rate=1.0,   # placeholder
            draft_ms=(t_draft - t0) * 1000,
            teacher_ms=(t_teacher - t_draft) * 1000,
            total_ms=(t_teacher - t0) * 1000,
        )

    def measure_draft_speed(self, cond_hidden, target_len: int, num_runs: int = 10):
        """Benchmark draft model speed — tokens/second."""
        if isinstance(cond_hidden, np.ndarray):
            cond_hidden = torch.from_numpy(cond_hidden).unsqueeze(0).to(self.device)

        # Warmup
        self.draft.generate_ar(cond_hidden, num_tokens=10)

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.draft.generate_ar(cond_hidden, num_tokens=target_len)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        tokens_per_sec = target_len / np.mean(times)

        return {
            "avg_ms": avg_ms,
            "tokens_per_sec": tokens_per_sec,
            "frames_per_sec": tokens_per_sec,  # 1 frame = 1 cb0 token
            "realtime_factor": tokens_per_sec / 25.0,  # 25 Hz frame rate
        }
