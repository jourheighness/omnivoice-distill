"""PyTorch dataset for cached teacher outputs."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherCacheDataset(Dataset):
    """Loads cached teacher outputs (npz files) for draft model training.

    Each sample contains:
        - cond_hidden: (L_cond, 1024) teacher hidden states at conditioning positions
        - cb0_tokens: (target_len,) codebook-0 token sequence from teacher
    """

    def __init__(self, cache_dir: str, max_seq_len: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.max_seq_len = max_seq_len

        manifest_path = self.cache_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Pre-load all data into memory (dataset is small enough)
        self.samples = []
        for entry in self.manifest:
            data = np.load(self.cache_dir / entry["file"])
            self.samples.append({
                "cond_hidden": data["cond_hidden"].astype(np.float32),
                "cb0_tokens": data["cb0_tokens"].astype(np.int64),
                "all_tokens": data["all_tokens"].astype(np.int64),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cond = torch.from_numpy(s["cond_hidden"])      # (L_cond, H)
        tokens = torch.from_numpy(s["cb0_tokens"])      # (T,)

        # AR shift: input = tokens[:-1], target = tokens[1:]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        return {
            "cond_hidden": cond,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
        }


def collate_fn(batch):
    """Pad batch to max length with -100 as ignore index."""
    max_cond = max(s["cond_hidden"].shape[0] for s in batch)
    max_tok = max(s["input_tokens"].shape[0] for s in batch)
    B = len(batch)
    H = batch[0]["cond_hidden"].shape[1]

    cond = torch.zeros(B, max_cond, H)
    input_tokens = torch.zeros(B, max_tok, dtype=torch.long)
    target_tokens = torch.full((B, max_tok), -100, dtype=torch.long)

    for i, s in enumerate(batch):
        cl = s["cond_hidden"].shape[0]
        tl = s["input_tokens"].shape[0]
        cond[i, :cl] = s["cond_hidden"]
        input_tokens[i, :tl] = s["input_tokens"]
        target_tokens[i, :tl] = s["target_tokens"]

    return {
        "cond_hidden": cond,
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
    }
