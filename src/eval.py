"""Evaluate draft model — acceptance rate, speed, and quality metrics.

Usage:
    python src/eval.py \
        --cache_dir ./cache \
        --checkpoint ./checkpoints/best.pt \
        --num_tests 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import TeacherCacheDataset
from draft_model import DraftModel


@torch.no_grad()
def measure_acceptance_rate(model, dataset, device, num_tests=50):
    """Measure how well draft predicts teacher's cb0 tokens."""
    model.eval()
    results = []

    indices = np.random.choice(len(dataset), min(num_tests, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        cond = sample["cond_hidden"].unsqueeze(0).to(device)
        target = sample["target_tokens"]  # (T,) shifted targets
        full_tokens = torch.from_numpy(dataset.samples[idx]["cb0_tokens"])

        # Generate AR
        t0 = time.perf_counter()
        generated = model.generate_ar(cond, num_tokens=len(full_tokens))
        gen_ms = (time.perf_counter() - t0) * 1000

        # Compare
        matches = (generated.cpu() == full_tokens).sum().item()
        total = len(full_tokens)

        results.append({
            "acceptance_rate": matches / total,
            "gen_ms": gen_ms,
            "tokens": total,
            "tokens_per_sec": total / (gen_ms / 1000),
        })

    return results


@torch.no_grad()
def measure_speed(model, cond_hidden, target_len=75, num_runs=20, device="cuda"):
    """Benchmark pure draft generation speed."""
    model.eval()
    cond = torch.from_numpy(cond_hidden).unsqueeze(0).to(device)

    # Warmup
    for _ in range(3):
        model.generate_ar(cond, num_tokens=target_len)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate_ar(cond, num_tokens=target_len)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "avg_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "tokens_per_sec": target_len / np.mean(times),
        "realtime_factor": (target_len / np.mean(times)) / 25.0,  # 25 Hz
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate draft model")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--num_tests", type=int, default=50)
    parser.add_argument("--speed_runs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DraftModel(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']}, eval_loss={ckpt['eval_loss']:.4f})")
    print(f"Parameters: {model.count_params():,}")

    # Load dataset
    dataset = TeacherCacheDataset(args.cache_dir)
    print(f"Dataset: {len(dataset)} samples")

    # Acceptance rate
    print(f"\n--- Acceptance Rate ({args.num_tests} tests) ---")
    results = measure_acceptance_rate(model, dataset, device, args.num_tests)

    rates = [r["acceptance_rate"] for r in results]
    speeds = [r["tokens_per_sec"] for r in results]

    print(f"  Mean acceptance:  {np.mean(rates):.1%} +/- {np.std(rates):.1%}")
    print(f"  Min/Max:          {np.min(rates):.1%} / {np.max(rates):.1%}")
    print(f"  Mean speed:       {np.mean(speeds):.0f} tok/s")

    # Speed benchmark
    print(f"\n--- Speed Benchmark ({args.speed_runs} runs) ---")
    sample_cond = dataset.samples[0]["cond_hidden"]
    speed = measure_speed(model, sample_cond, target_len=75, num_runs=args.speed_runs, device=str(device))

    print(f"  Generation:       {speed['avg_ms']:.1f} +/- {speed['std_ms']:.1f} ms")
    print(f"  Throughput:       {speed['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Realtime factor:  {speed['realtime_factor']:.1f}x (1.0x = realtime at 25Hz)")

    # Streaming feasibility
    frame_ms = 1000 / 25  # 40ms per frame
    draft_ms_per_token = speed["avg_ms"] / 75
    print(f"\n--- Streaming Feasibility ---")
    print(f"  Frame interval:   {frame_ms:.0f}ms (25 Hz)")
    print(f"  Draft per token:  {draft_ms_per_token:.1f}ms")
    if draft_ms_per_token < frame_ms:
        print(f"  Status:           FEASIBLE (draft is {frame_ms/draft_ms_per_token:.1f}x faster than realtime)")
    else:
        print(f"  Status:           TOO SLOW (need {draft_ms_per_token/frame_ms:.1f}x speedup)")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    acceptance = np.mean(rates)
    if acceptance > 0.7:
        verdict = "EXCELLENT — draft is highly predictive, speculative decode will be very effective"
    elif acceptance > 0.5:
        verdict = "GOOD — decent acceptance rate, worth deploying"
    elif acceptance > 0.3:
        verdict = "MODERATE — some benefit, consider training longer or larger model"
    else:
        verdict = "LOW — draft needs more training data or larger architecture"
    print(f"  Acceptance: {acceptance:.1%} — {verdict}")

    # Save results
    results_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "acceptance_mean": float(np.mean(rates)),
            "acceptance_std": float(np.std(rates)),
            "speed_avg_ms": speed["avg_ms"],
            "speed_tokens_per_sec": speed["tokens_per_sec"],
            "realtime_factor": speed["realtime_factor"],
            "num_params": model.count_params(),
            "config": cfg,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
