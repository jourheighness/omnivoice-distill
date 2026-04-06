"""Cache MLX teacher outputs for draft training.

Uses the MLX OmniVoice teacher (same as inference) to generate
conditioning hidden states + audio tokens. This ensures the draft
model is trained on the exact same conditioning distribution it
will see at inference time on Apple Silicon.

Usage:
    python local/cache_mlx_teacher.py \
        --teacher_weights ~/bartholomew/source/voice-service/weights/omnivoice_mlx \
        --ref_audio ~/bartholomew/source/voice-service/voices/barth-v01/ref_audio.wav \
        --output_dir ./cache_mlx \
        --num_samples 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))

# Diverse text samples for training data
TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "She walked through the garden, admiring the colorful flowers.",
    "Scientists discovered a new species deep in the ocean last week.",
    "The train arrived at the station exactly on time this morning.",
    "He opened the old book and began reading the first chapter aloud.",
    "The weather forecast predicts rain throughout the entire weekend.",
    "They decided to take a different route through the mountains.",
    "The concert was absolutely incredible and the crowd loved it.",
    "Please remember to lock the door when you leave the building.",
    "The children played happily in the park until sunset arrived.",
    "I would like to order a large coffee with extra cream please.",
    "The new software update includes several important security fixes.",
    "Her presentation at the conference received a standing ovation.",
    "The bridge was built over two hundred years ago by skilled craftsmen.",
    "We need to finish this project before the deadline next Friday.",
    "The library has thousands of books covering every subject imaginable.",
    "He learned to play the piano when he was just five years old.",
    "The restaurant on the corner serves the best pasta in the city.",
    "They flew across the Atlantic Ocean in less than eight hours.",
    "The museum exhibit showcases artifacts from ancient civilizations.",
    "Can you help me carry these heavy boxes up to the third floor?",
    "The meeting has been rescheduled to next Wednesday at two o'clock.",
    "She finished the marathon in under four hours, setting a personal best.",
    "The old castle on the hill has been converted into a luxury hotel.",
    "Please make sure to backup your files before the system update.",
    "The documentary about ocean life won several international awards.",
    "He drives to work every day, but today he decided to take the bus.",
    "The company announced plans to expand into three new markets this year.",
    "She speaks fluently in English, French, and Mandarin Chinese.",
    "The sunset painted the sky in brilliant shades of orange and purple.",
    "We should consider all options before making a final decision.",
    "The hospital opened a new wing dedicated to pediatric care.",
    "His grandmother taught him how to bake bread from scratch.",
    "The flight was delayed by two hours due to severe thunderstorms.",
    "Technology continues to transform the way we communicate daily.",
    "The garden produces fresh vegetables throughout the summer months.",
    "She received a scholarship to study engineering at the university.",
    "The team worked late into the night to meet the project deadline.",
    "Artificial intelligence is rapidly changing many industries worldwide.",
    "The hiking trail leads through dense forests and open meadows.",
    "Please submit your application before the end of the month.",
    "The symphony orchestra performed beautifully at the gala event.",
    "He spent three years traveling around the world after graduating.",
    "The new shopping center will open its doors to the public tomorrow.",
    "Regular exercise and a balanced diet are essential for good health.",
    "The volcano erupted unexpectedly, sending ash clouds into the sky.",
    "She designed an award winning mobile application for healthcare.",
    "The winter storm brought heavy snow and freezing temperatures.",
    "We celebrated their anniversary with a surprise dinner party.",
    "The research paper was published in a prestigious scientific journal.",
    "Hello there! How are you doing today? I hope everything is going well.",
    "Good morning everyone. Today we will discuss the quarterly results.",
    "I've been thinking about this problem for a long time now.",
    "Let me explain how the new system works step by step.",
    "The most important thing to remember is to stay focused.",
    "Once upon a time, in a land far away, there lived a wise old king.",
    "Breaking news: a major discovery has been made in quantum physics.",
    "Thank you so much for your help. I really appreciate everything.",
    "The algorithm processes millions of data points in real time.",
    "Would you mind passing me that book on the top shelf please?",
    "It was the best of times, it was the worst of times.",
    "The architecture of this building is truly remarkable and unique.",
    "Every morning, she begins her day with a cup of green tea.",
    "The stock market experienced significant volatility this quarter.",
    "We are pleased to announce the launch of our newest product line.",
    "The detective carefully examined the evidence at the crime scene.",
    "Spring brings warmer temperatures and longer days of sunshine.",
    "His speech inspired thousands of people to take action immediately.",
    "The recipe calls for two cups of flour and one cup of sugar.",
    "International cooperation is essential for addressing climate change.",
    "She painted a beautiful landscape of the coastal cliffs at dawn.",
    "The software engineer debugged the complex issue within an hour.",
    "Tomorrow's weather will be partly cloudy with a chance of showers.",
    "The ancient manuscript was carefully preserved in a climate controlled vault.",
    "Running a marathon requires months of dedicated training and preparation.",
    "The professor explained the complex theorem with remarkable clarity.",
    "They adopted a golden retriever puppy from the local animal shelter.",
    "The spacecraft successfully landed on the surface of Mars yesterday.",
    "A good night's sleep is crucial for mental and physical wellbeing.",
    "The jazz festival attracts musicians and fans from around the globe.",
    "She completed her doctoral thesis on renewable energy systems.",
    "The coastal town is known for its stunning beaches and fresh seafood.",
    "Please review the contract carefully before signing on the dotted line.",
    "The photographer captured the perfect moment during the wildlife safari.",
    "Education is the most powerful weapon for changing the world.",
    "The startup raised ten million dollars in their latest funding round.",
    "He volunteers at the community center every Saturday afternoon.",
    "The northern lights created a spectacular display across the arctic sky.",
    "Machine learning models require large amounts of training data.",
    "The bakery down the street makes the most delicious croissants.",
    "Climate scientists warn about rising sea levels in coastal regions.",
    "She won the chess tournament by defeating the reigning champion.",
    "The new highway will significantly reduce commute times for residents.",
    "Fresh herbs from the garden add wonderful flavor to any dish.",
    "The astronomer discovered a previously unknown exoplanet last night.",
    "We look forward to welcoming you at our annual conference in June.",
    "The ancient ruins tell the story of a once thriving civilization.",
    "Practicing mindfulness meditation can help reduce stress and anxiety.",
    "The electric vehicle market continues to grow rapidly each year.",
    "Her novel became an instant bestseller upon its release last month.",
]


def encode_ref_audio(ref_audio_path, ref_text):
    """Encode reference audio via PyTorch (one-time cost)."""
    from omnivoice import OmniVoice as OmniVoicePT

    print("  Loading PyTorch model for ref audio encoding...")
    pt_model = OmniVoicePT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)

    voice_prompt = pt_model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        preprocess_prompt=True,
    )

    ref_audio_tokens = voice_prompt.ref_audio_tokens.cpu().numpy()
    actual_ref_text = voice_prompt.ref_text

    del pt_model
    if hasattr(torch, 'mps'):
        torch.mps.empty_cache()

    return ref_audio_tokens, actual_ref_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", default="This is a reference audio sample for voice cloning.")
    parser.add_argument("--output_dir", default="./cache_mlx")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_step", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Encode ref audio (PyTorch, one-time)
    print("Step 1: Encoding reference audio...")
    ref_tokens_np, ref_text = encode_ref_audio(args.ref_audio, args.ref_text)
    ref_audio_tokens = mx.array(ref_tokens_np, dtype=mx.int32)
    print(f"  Ref tokens shape: {ref_audio_tokens.shape}")
    print(f"  Ref text: '{ref_text}'")

    # Step 2: Load MLX teacher
    print("\nStep 2: Loading MLX teacher...")
    from omnivoice_mlx.generate import (
        OmniVoiceMLXConfig, OmniVoiceMLXModel, generate_iterative,
    )

    config = OmniVoiceMLXConfig()
    teacher = OmniVoiceMLXModel(config)
    wp = Path(args.teacher_weights)
    teacher.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
    mx.eval(teacher.parameters())
    print("  Teacher loaded.")

    # Step 3: Setup tokenizer and style tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")

    C = config.num_audio_codebook
    mask_id = config.audio_mask_id

    style_text = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tokenizer(style_text, return_tensors="np").input_ids[0]
    style_tokens = mx.broadcast_to(
        mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids))
    )

    # Step 4: Generate and cache
    print(f"\nStep 3: Generating {args.num_samples} samples...")
    manifest = []
    t_start = time.perf_counter()

    for i in range(args.num_samples):
        text = TEXTS[i % len(TEXTS)]

        # Add some variation by combining texts
        if i >= len(TEXTS):
            # Combine two random texts for longer sequences
            t2 = TEXTS[(i * 7) % len(TEXTS)]
            text = text + " " + t2

        # Build text tokens
        text_str = f"<|text_start|>{ref_text} {text}<|text_end|>"
        text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
        text_tokens = mx.broadcast_to(
            mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids))
        )

        # Estimate target length
        chars_per_frame = max(1, len(ref_text) / ref_audio_tokens.shape[1])
        target_len = max(25, min(150, int(len(text) / chars_per_frame)))

        # Build full input
        target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)
        input_ids = mx.concatenate([style_tokens, text_tokens, ref_audio_tokens, target_masks], axis=1)
        L_total = input_ids.shape[1]
        L_cond = L_total - target_len

        audio_mask = mx.concatenate([
            mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
            mx.ones(ref_audio_tokens.shape[1] + target_len, dtype=mx.bool_),
        ])

        # Generate tokens via iterative unmasking
        mx.random.seed(i)
        tokens = generate_iterative(
            teacher, input_ids, audio_mask, target_len,
            num_step=args.num_step, guidance_scale=3.0,
        )
        mx.eval(tokens)

        # Extract conditioning hidden states
        ids_batch = mx.expand_dims(input_ids, 0)
        mask_batch = mx.expand_dims(audio_mask, 0)
        attn = mx.ones((1, 1, L_total, L_total), dtype=mx.bool_)

        embeds = teacher._prepare_embed_inputs(ids_batch, mask_batch)
        hidden = teacher.llm(embeds, mask=attn)
        cond_hidden = hidden[:, :L_cond, :]
        mx.eval(cond_hidden)

        # Save
        fname = f"sample_{i:04d}.npz"
        np.savez_compressed(
            output_dir / fname,
            cond_hidden=np.array(cond_hidden[0]),
            cb0_tokens=np.array(tokens[0]),
            all_tokens=np.array(tokens),
        )
        manifest.append({
            "file": fname,
            "target_len": target_len,
            "cond_len": L_cond,
            "text": text[:100],
        })

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (args.num_samples - i - 1) / rate
            print(f"  [{i+1}/{args.num_samples}] {rate:.1f} samples/s | ETA: {eta:.0f}s")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = time.perf_counter() - t_start
    print(f"\nCached {len(manifest)} samples to {output_dir}/ in {total:.0f}s")


if __name__ == "__main__":
    main()
