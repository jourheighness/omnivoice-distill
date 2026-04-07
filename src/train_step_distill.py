"""SDTT-style step distillation for OmniVoice: 8→4 steps.

The student is OmniVoice itself (fine-tuned with LoRA or full).
Training: given state at step K, student's logits should match
teacher's logits at step K+2.

This teaches the model to "skip" every other step, producing
8-step quality in 4 steps.
"""

import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


class TrajectoryDataset(Dataset):
    """Load cached trajectories and create training pairs.

    Each pair: (input_state, target_logits)
    - input_state: partially-unmasked tokens at step K
    - target_logits: teacher's logits at step K+2

    For 8→4 distillation, pairs are: (step0→step2), (step2→step4), (step4→step6), (step6→step7)
    """

    def __init__(self, cache_dir, manifest_path, max_t_len=150):
        self.cache_dir = Path(cache_dir)
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.max_t_len = max_t_len

        # Pre-filter to skip too-long samples
        self.manifest = [m for m in self.manifest if m["t_len"] <= max_t_len]
        print(f"  Dataset: {len(self.manifest)} samples (max_t_len={max_t_len})")

        # Each sample produces 4 training pairs
        self.pair_indices = []  # (sample_idx, src_step, tgt_step)
        for idx in range(len(self.manifest)):
            for src, tgt in [(0, 2), (2, 4), (4, 6), (6, 7)]:
                self.pair_indices.append((idx, src, tgt))

        print(f"  Training pairs: {len(self.pair_indices)}")

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, index):
        sample_idx, src_step, tgt_step = self.pair_indices[index]
        m = self.manifest[sample_idx]

        data = torch.load(
            self.cache_dir / m["file"],
            map_location="cpu",
            weights_only=False,
        )

        t_len = data["t_len"]

        # Input: token state at src_step
        # step_states shape: (9, C, T) — states 0-8
        input_tokens = data["step_states"][src_step]  # (C, T)

        # Target: teacher logits at tgt_step
        logits_key = f"logits_step{tgt_step}"
        target_logits = data["pair_logits"][logits_key].float()  # (C, T, V)

        # Conditioning
        cond_ids = data["cond_ids"]      # (C, c_len)
        audio_mask = data["audio_mask"]  # (c_len,)

        return {
            "input_tokens": input_tokens,    # (C, T)
            "target_logits": target_logits,  # (C, T, V)
            "cond_ids": cond_ids,
            "audio_mask": audio_mask,
            "c_len": data["c_len"],
            "t_len": t_len,
        }


def collate_fn(batch):
    """Pad batch to max lengths."""
    max_t = max(b["t_len"] for b in batch)
    max_c = max(b["c_len"] for b in batch)
    B = len(batch)
    C = batch[0]["input_tokens"].shape[0]
    V = batch[0]["target_logits"].shape[-1]
    mask_id = 1024  # OmniVoice mask token

    input_tokens = torch.full((B, C, max_t), mask_id, dtype=torch.long)
    target_logits = torch.zeros(B, C, max_t, V)
    cond_ids = torch.full((B, C, max_c), mask_id, dtype=torch.long)
    audio_mask = torch.zeros(B, max_c, dtype=torch.bool)
    t_lens = torch.tensor([b["t_len"] for b in batch])
    c_lens = torch.tensor([b["c_len"] for b in batch])

    for i, b in enumerate(batch):
        t = b["t_len"]
        c = b["c_len"]
        input_tokens[i, :, :t] = b["input_tokens"][:, :t]
        target_logits[i, :, :t, :] = b["target_logits"][:, :t, :]
        cond_ids[i, :, :c] = b["cond_ids"][:, :c]
        audio_mask[i, :c] = b["audio_mask"][:c]

    return {
        "input_tokens": input_tokens,
        "target_logits": target_logits,
        "cond_ids": cond_ids,
        "audio_mask": audio_mask,
        "t_lens": t_lens,
        "c_lens": c_lens,
    }


def train_step(model, batch, optimizer, device, gen_config):
    """One training step: student predicts logits, match teacher logits via KL."""
    B = batch["input_tokens"].shape[0]
    C = batch["input_tokens"].shape[1]
    mask_id = 1024

    input_tokens = batch["input_tokens"].to(device)
    target_logits = batch["target_logits"].to(device)
    t_lens = batch["t_lens"]
    c_lens = batch["c_lens"]
    cond_ids = batch["cond_ids"].to(device)
    audio_mask_batch = batch["audio_mask"].to(device)

    max_t = input_tokens.shape[2]
    max_c = cond_ids.shape[2]

    # Build full input: conditioning + target tokens
    # For training, we run both cond + uncond (CFG) like the teacher
    full_len = max_c
    full_input_ids = torch.full(
        (B, C, full_len), mask_id, dtype=torch.long, device=device,
    )
    full_audio_mask = torch.zeros(
        (B, full_len), dtype=torch.bool, device=device,
    )
    full_attention_mask = torch.zeros(
        (B, 1, full_len, full_len), dtype=torch.bool, device=device,
    )

    for i in range(B):
        c = c_lens[i].item()
        t = t_lens[i].item()
        full_input_ids[i, :, :c] = cond_ids[i, :, :c]
        full_input_ids[i, :, c - t:c] = input_tokens[i, :, :t]
        full_audio_mask[i, :c] = audio_mask_batch[i, :c]
        full_attention_mask[i, :, :c, :c] = True

    # Forward pass through student (conditional only — no CFG during training)
    output = model(
        input_ids=full_input_ids,
        audio_mask=full_audio_mask,
        attention_mask=full_attention_mask,
    )
    student_logits = output.logits.to(torch.float32)  # (B, C, full_len, V)

    # Extract target region logits
    total_loss = 0.0
    n_positions = 0

    for i in range(B):
        c = c_lens[i].item()
        t = t_lens[i].item()

        s_logits = student_logits[i, :, c - t:c, :]  # (C, T, V)
        t_logits = target_logits[i, :, :t, :]  # (C, T, V)

        # Only compute loss on positions that were MASKED in the input
        # (positions already unmasked shouldn't be trained on)
        is_masked = input_tokens[i, :, :t] == mask_id  # (C, T)

        if is_masked.sum() == 0:
            continue

        # KL divergence: teacher is the target distribution
        # KL(teacher || student) = sum(teacher * (log_teacher - log_student))
        teacher_probs = F.softmax(t_logits, dim=-1)
        student_log_probs = F.log_softmax(s_logits, dim=-1)

        # Per-position KL
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")  # (C, T, V)
        kl = kl.sum(dim=-1)  # (C, T)

        # Mask out non-masked positions
        kl = kl * is_masked.float()

        total_loss += kl.sum()
        n_positions += is_masked.sum().item()

    if n_positions == 0:
        return 0.0

    loss = total_loss / n_positions

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()

    return loss.item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./cache_trajectories")
    parser.add_argument("--manifest", default="./cache_trajectories/manifest.json")
    parser.add_argument("--output_dir", default="./checkpoints_distill")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank. 0 = full fine-tune")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = TrajectoryDataset(args.cache_dir, args.manifest)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    print("Loading OmniVoice student model...")
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float32,
    )

    gen_config = OmniVoiceGenerationConfig(
        num_step=8, guidance_scale=3.0,
        position_temperature=5.0, class_temperature=0.0,
    )

    # Apply LoRA if requested
    if args.lora_rank > 0:
        print(f"Applying LoRA (rank={args.lora_rank})...")
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model.llm = get_peft_model(model.llm, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")
        except ImportError:
            print("  peft not installed, falling back to full fine-tune")
            args.lora_rank = 0

    if args.lora_rank == 0:
        # Full fine-tune: freeze embeddings, only train transformer
        for name, p in model.named_parameters():
            if "audio_embed" in name or "text_embed" in name or "audio_tokenizer" in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Full fine-tune: {trainable:,} trainable params")

    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            loss = train_step(model, batch, optimizer, args.device, gen_config)
            total_loss += loss
            n_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}", avg=f"{total_loss/n_batches:.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"epoch{epoch+1}.pt"
        if args.lora_rank > 0:
            model.llm.save_pretrained(str(output_dir / f"lora_epoch{epoch+1}"))
        else:
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()
                 if any(x not in k for x in ["audio_tokenizer"])},
                ckpt_path,
            )
        print(f"  Saved checkpoint: {ckpt_path}")

    print("Done!")


if __name__ == "__main__":
    main()
