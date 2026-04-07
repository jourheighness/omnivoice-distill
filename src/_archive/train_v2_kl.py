"""Train v2 draft with KL divergence on soft targets.

Instead of cross-entropy on hard tokens, minimizes KL divergence
between draft's output distribution and teacher's distribution.
This lets the draft learn the teacher's uncertainty, not just argmax.

Usage:
    python src/train_v2_kl.py --cache_dir ./cache_v2_soft \
        --teacher_weights model.safetensors --num_epochs 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from draft_model_v2 import DraftModelV2


class SoftTargetDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.samples = []
        skipped = 0
        for entry in self.manifest:
            try:
                d = np.load(self.cache_dir / entry["file"])
                cond = d["cond_ids"].astype(np.int64)
                mask = d["audio_mask"].astype(np.bool_)
                cb0 = d["cb0_tokens"].astype(np.int64)
                topk_idx = d["topk_indices"].astype(np.int64)
                topk_probs = d["topk_probs"].astype(np.float32)
                if cond.ndim != 2 or len(cb0) < 2:
                    skipped += 1
                    continue
                self.samples.append({
                    "cond_ids": cond, "audio_mask": mask,
                    "cb0_tokens": cb0,
                    "topk_indices": topk_idx, "topk_probs": topk_probs,
                })
            except Exception:
                skipped += 1
        if skipped:
            print(f"  Skipped {skipped} invalid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "cond_ids": torch.from_numpy(s["cond_ids"]),
            "audio_mask": torch.from_numpy(s["audio_mask"]),
            "input_tokens": torch.from_numpy(s["cb0_tokens"][:-1]),
            "target_tokens": torch.from_numpy(s["cb0_tokens"][1:]),
            "topk_indices": torch.from_numpy(s["topk_indices"][1:]),  # shifted
            "topk_probs": torch.from_numpy(s["topk_probs"][1:]),      # shifted
        }


def collate_soft(batch):
    max_cond = max(s["cond_ids"].shape[1] for s in batch)
    max_tok = max(s["input_tokens"].shape[0] for s in batch)
    B = len(batch)
    C = batch[0]["cond_ids"].shape[0]
    K = batch[0]["topk_indices"].shape[1]

    cond_ids = torch.zeros(B, C, max_cond, dtype=torch.long)
    audio_mask = torch.zeros(B, max_cond, dtype=torch.bool)
    input_tokens = torch.zeros(B, max_tok, dtype=torch.long)
    target_tokens = torch.full((B, max_tok), -100, dtype=torch.long)
    topk_indices = torch.zeros(B, max_tok, K, dtype=torch.long)
    topk_probs = torch.zeros(B, max_tok, K)
    valid_mask = torch.zeros(B, max_tok, dtype=torch.bool)

    for i, s in enumerate(batch):
        cl = s["cond_ids"].shape[1]
        tl = s["input_tokens"].shape[0]
        cond_ids[i, :, :cl] = s["cond_ids"]
        audio_mask[i, :cl] = s["audio_mask"]
        input_tokens[i, :tl] = s["input_tokens"]
        target_tokens[i, :tl] = s["target_tokens"]
        topk_indices[i, :tl] = s["topk_indices"]
        topk_probs[i, :tl] = s["topk_probs"]
        valid_mask[i, :tl] = True

    return {
        "cond_ids": cond_ids, "audio_mask": audio_mask,
        "input_tokens": input_tokens, "target_tokens": target_tokens,
        "topk_indices": topk_indices, "topk_probs": topk_probs,
        "valid_mask": valid_mask,
    }


def kl_loss_topk(draft_logits, topk_indices, topk_probs, valid_mask, temperature=2.0):
    """KL divergence using sparse top-k teacher distribution.

    draft_logits: (B, T, V)
    topk_indices: (B, T, K) — teacher's top-k token indices
    topk_probs: (B, T, K) — teacher's top-k probabilities
    valid_mask: (B, T) — which positions are valid
    """
    B, T, V = draft_logits.shape
    K = topk_indices.shape[-1]

    # Draft log-probs at temperature
    draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)  # (B, T, V)

    # Gather draft log-probs at teacher's top-k indices
    # topk_indices: (B, T, K)
    draft_topk_log_probs = torch.gather(draft_log_probs, 2, topk_indices)  # (B, T, K)

    # Teacher probs (already softmaxed, re-temperature)
    teacher_probs = topk_probs  # (B, T, K)
    # Renormalize teacher probs at temperature
    teacher_logits = torch.log(teacher_probs + 1e-10) * temperature
    teacher_probs_t = F.softmax(teacher_logits / temperature, dim=-1)

    # KL: sum_k p_teacher * (log p_teacher - log p_draft)
    kl = teacher_probs_t * (torch.log(teacher_probs_t + 1e-10) - draft_topk_log_probs)
    kl = kl.sum(dim=-1)  # (B, T)

    # Mask
    kl = kl * valid_mask.float()
    return kl.sum() / valid_mask.sum().clamp(min=1)


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0, kl_weight=0.5, temperature=2.0):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        cond_ids = batch["cond_ids"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        tokens = batch["input_tokens"].to(device)
        targets = batch["target_tokens"].to(device)
        topk_idx = batch["topk_indices"].to(device)
        topk_p = batch["topk_probs"].to(device)
        valid = batch["valid_mask"].to(device)

        logits = model(tokens, cond_ids=cond_ids, audio_mask=audio_mask)

        # CE loss on hard targets
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        # KL loss on soft targets
        kl = kl_loss_topk(logits, topk_idx, topk_p, valid, temperature=temperature)

        loss = (1 - kl_weight) * ce + kl_weight * kl * (temperature ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        preds = logits.argmax(dim=-1)
        mask = targets != -100
        correct = ((preds == targets) & mask).sum().item()
        n = mask.sum().item()

        total_loss += loss.item() * n
        total_ce += ce.item() * n
        total_kl += kl.item() * n
        total_correct += correct
        total_tokens += n

    n = max(total_tokens, 1)
    return total_loss / n, total_ce / n, total_kl / n, total_correct / n


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0

    for batch in dataloader:
        cond_ids = batch["cond_ids"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        tokens = batch["input_tokens"].to(device)
        targets = batch["target_tokens"].to(device)

        logits = model(tokens, cond_ids=cond_ids, audio_mask=audio_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        mask = targets != -100
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & mask).sum().item()
        n = mask.sum().item()
        total_loss += loss.item() * n
        total_correct += correct
        total_tokens += n

    n = max(total_tokens, 1)
    return total_loss / n, total_correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--checkpoint_dir", default="./checkpoints_v2_kl")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl_weight", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--eval_split", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = SoftTargetDataset(args.cache_dir)
    n_eval = max(1, int(len(ds) * args.eval_split))
    n_train = len(ds) - n_eval
    train_ds, eval_ds = torch.utils.data.random_split(ds, [n_train, n_eval])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_soft, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_soft, num_workers=2, pin_memory=True)

    print(f"Dataset: {n_train} train, {n_eval} eval (soft targets)")

    model = DraftModelV2(
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=0.1,
    ).to(device)
    model.load_teacher_embeddings(args.teacher_weights, device=device)

    print(f"Draft: {model.count_params():,} trainable params")
    print(f"KL weight: {args.kl_weight}, temperature: {args.temperature}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")

    print(f"\nTraining {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        t0 = time.perf_counter()
        loss, ce, kl, acc = train_epoch(model, train_loader, optimizer, scheduler, device,
                                         kl_weight=args.kl_weight, temperature=args.temperature)
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        elapsed = time.perf_counter() - t0

        print(f"  Epoch {epoch+1}/{args.num_epochs} | "
              f"loss={loss:.4f} ce={ce:.4f} kl={kl:.4f} acc={acc:.1%} | "
              f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.1%} | {elapsed:.1f}s")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({
                "model_state_dict": {k: v for k, v in model.state_dict().items()
                                     if "text_embed" not in k and "audio_embed" not in k},
                "config": {"hidden_size": args.hidden_size, "num_layers": args.num_layers, "num_heads": args.num_heads},
                "epoch": epoch, "eval_loss": eval_loss, "eval_acc": eval_acc,
            }, ckpt_dir / "best.pt")
            print(f"    -> Saved best")

    print(f"\nDone. Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
