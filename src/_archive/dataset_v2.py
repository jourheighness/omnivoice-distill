"""Dataset for token-conditioned draft model v2.

Loads cached token IDs (not hidden states) — framework independent.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherCacheDatasetV2(Dataset):
    """Loads token-based teacher cache for v2 draft model.

    Each sample contains:
        - cond_ids: (C, L_cond) conditioning token IDs
        - audio_mask: (L_cond,) bool
        - cb0_tokens: (target_len,) codebook-0 tokens
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        self.samples = []
        skipped = 0
        for entry in self.manifest:
            try:
                data = np.load(self.cache_dir / entry["file"])
                cond = data["cond_ids"].astype(np.int64)
                mask = data["audio_mask"].astype(np.bool_)
                cb0 = data["cb0_tokens"].astype(np.int64)
                # Validate
                if cond.ndim != 2 or mask.ndim != 1 or cb0.ndim != 1:
                    skipped += 1
                    continue
                if len(cb0) < 2 or cond.shape[1] != len(mask):
                    skipped += 1
                    continue
                self.samples.append({"cond_ids": cond, "audio_mask": mask, "cb0_tokens": cb0})
            except Exception:
                skipped += 1
        if skipped:
            print(f"  Skipped {skipped} invalid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cond_ids = torch.from_numpy(s["cond_ids"])        # (C, L_cond)
        audio_mask = torch.from_numpy(s["audio_mask"])    # (L_cond,)
        tokens = torch.from_numpy(s["cb0_tokens"])        # (T,)

        return {
            "cond_ids": cond_ids,
            "audio_mask": audio_mask,
            "input_tokens": tokens[:-1],
            "target_tokens": tokens[1:],
        }


def collate_v2(batch):
    """Pad batch to max length."""
    max_cond = max(s["cond_ids"].shape[1] for s in batch)
    max_tok = max(s["input_tokens"].shape[0] for s in batch)
    B = len(batch)
    C = batch[0]["cond_ids"].shape[0]

    cond_ids = torch.zeros(B, C, max_cond, dtype=torch.long)
    audio_mask = torch.zeros(B, max_cond, dtype=torch.bool)
    input_tokens = torch.zeros(B, max_tok, dtype=torch.long)
    target_tokens = torch.full((B, max_tok), -100, dtype=torch.long)

    for i, s in enumerate(batch):
        cl = s["cond_ids"].shape[1]
        tl = s["input_tokens"].shape[0]
        cond_ids[i, :, :cl] = s["cond_ids"]
        audio_mask[i, :cl] = s["audio_mask"]
        input_tokens[i, :tl] = s["input_tokens"]
        target_tokens[i, :tl] = s["target_tokens"]

    return {
        "cond_ids": cond_ids,
        "audio_mask": audio_mask,
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
    }
