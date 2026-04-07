"""Test v2 draft with a known training speaker — proof it works."""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-service"))

import mlx.core as mx
from draft_mlx_v2 import DraftV2Config, DraftModelV2MLX
from omnivoice_mlx.generate import OmniVoiceMLXConfig, OmniVoiceMLXModel, generate_iterative
from omnivoice_mlx.vocoder import AudioTokenizerDecoder

# Load draft
draft = DraftModelV2MLX(DraftV2Config(hidden_size=512, num_layers=6, num_heads=8))
draft.load_weights("../checkpoints/draft_v2_mlx.safetensors")
mx.eval(draft.parameters())

# Load teacher + vocoder
config = OmniVoiceMLXConfig()
teacher = OmniVoiceMLXModel(config)
vocoder = AudioTokenizerDecoder(
    num_quantizers=config.num_quantizers, hidden_size=config.vq_hidden_size,
    codebook_dim=config.codebook_dim, codebook_size=config.codebook_size,
    semantic_hidden_size=config.semantic_hidden_size, dac_input_dim=config.dac_input_dim,
    dac_hidden_dim=config.dac_hidden_dim, dac_upsampling_ratios=config.dac_upsampling_ratios,
)
wp = Path("/Users/johannescarlsten/bartholomew/source/voice-service/weights/omnivoice_mlx")
teacher.load_weights(list(mx.load(str(wp / "model.safetensors")).items()))
vocoder.load_weights(list(mx.load(str(wp / "vocoder.safetensors")).items()))
mx.eval(teacher.parameters(), vocoder.parameters())
print("Models loaded.")

# Use LibriTTS speaker from training data
from omnivoice import OmniVoice as PT
pt = PT.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)
manifest = json.load(open("../data/libritts_local/manifest.json"))
ref = manifest[0]
audio_path = str(Path("..") / ref["audio_path"])
print(f"Speaker {ref['speaker_id']}: {ref['text'][:60]}")

prompt = pt.create_voice_clone_prompt(ref_audio=audio_path, ref_text=ref["text"][:50], preprocess_prompt=True)
ref_tokens = mx.array(prompt.ref_audio_tokens.cpu().numpy(), dtype=mx.int32)
ref_text = prompt.ref_text
del pt

# Build input
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("k2-fsa/OmniVoice")
C, mask_id = 8, 1024

style_str = "<|denoise|><|lang_start|>English<|lang_end|><|instruct_start|>None<|instruct_end|>"
style_ids = tokenizer(style_str, return_tensors="np").input_ids[0]
style_tokens = mx.broadcast_to(mx.array(style_ids, dtype=mx.int32).reshape(1, -1), (C, len(style_ids)))

synth_text = "The weather today is absolutely beautiful and I feel wonderful."
text_str = f"<|text_start|>{ref_text} {synth_text}<|text_end|>"
text_ids = tokenizer(text_str, return_tensors="np").input_ids[0]
text_tokens = mx.broadcast_to(mx.array(text_ids, dtype=mx.int32).reshape(1, -1), (C, len(text_ids)))

target_len = 60
target_masks = mx.full((C, target_len), mask_id, dtype=mx.int32)
input_ids = mx.concatenate([style_tokens, text_tokens, ref_tokens, target_masks], axis=1)
L_cond = input_ids.shape[1] - target_len
audio_mask = mx.concatenate([
    mx.zeros(style_tokens.shape[1] + text_tokens.shape[1], dtype=mx.bool_),
    mx.ones(ref_tokens.shape[1] + target_len, dtype=mx.bool_),
])
print(f"Cond len: {L_cond}, target: {target_len}")

# Baseline generation
mx.random.seed(42)
t0 = time.perf_counter()
teacher_tokens = generate_iterative(teacher, input_ids, audio_mask, target_len, num_step=8, guidance_scale=3.0)
mx.eval(teacher_tokens)
base_ms = (time.perf_counter() - t0) * 1000
teacher_cb0 = np.array(teacher_tokens[0])
print(f"Teacher cb0: {teacher_cb0[:15]}")

# Draft teacher-forced
cond_ids = mx.expand_dims(input_ids[:, :L_cond], 0)
cond_mask = mx.expand_dims(audio_mask[:L_cond], 0)
tokens_in = mx.expand_dims(mx.array(teacher_cb0[:-1].astype(np.int32)), 0)
logits = draft(tokens_in, cond_ids=cond_ids, audio_mask=cond_mask)
mx.eval(logits)
preds = np.array(logits.argmax(axis=-1)[0])
match = int((preds == teacher_cb0[1:]).sum())
total = len(teacher_cb0) - 1

# Decode audio
import soundfile as sf
out = Path("../test_output/known_speaker")
out.mkdir(parents=True, exist_ok=True)
audio = np.array(vocoder(mx.expand_dims(teacher_tokens, 0))[0, :, 0])
sf.write(str(out / "output.wav"), audio, 24000)

print(f"\nRESULTS (known LibriTTS speaker)")
print(f"  Acceptance:  {match/total:.1%} ({match}/{total})")
print(f"  Baseline:    {base_ms:.0f}ms")
print(f"  Audio:       {out}/output.wav ({len(audio)/24000:.1f}s)")
