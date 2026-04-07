"""Inspect OmniVoice generate API to find how to get tokens."""

import torch
import numpy as np
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
prompt = model.create_voice_clone_prompt(ref_audio="barth_ref.wav", ref_text="Reference.", preprocess_prompt=True)

# Check _generate_iterative signature
import inspect
sig = inspect.signature(model._generate_iterative)
print(f"_generate_iterative params: {list(sig.parameters.keys())}")

# Check GenerationTask and GenerationConfig
from omnivoice.models.omnivoice import GenerationTask, OmniVoiceGenerationConfig
print(f"\nGenerationTask fields: {[f for f in dir(GenerationTask) if not f.startswith('_')]}")
print(f"GenerationConfig fields: {[f for f in dir(OmniVoiceGenerationConfig) if not f.startswith('_')]}")

# Try generate and see what we get back
result = model.generate(text="Hello there.", voice_clone_prompt=prompt)
print(f"\ngenerate() returns: {type(result)}")
if hasattr(result, '__len__'):
    print(f"  length: {len(result)}")
if isinstance(result, dict):
    print(f"  keys: {result.keys()}")
if isinstance(result, (list, tuple)):
    for i, r in enumerate(result[:3]):
        print(f"  [{i}]: type={type(r)} shape={r.shape if hasattr(r, 'shape') else 'N/A'}")
if hasattr(result, 'shape'):
    print(f"  shape: {result.shape}")
if hasattr(result, 'audio'):
    print(f"  .audio: {result.audio.shape if hasattr(result.audio, 'shape') else type(result.audio)}")
if hasattr(result, 'tokens'):
    print(f"  .tokens: {result.tokens}")
if hasattr(result, 'audio_tokens'):
    print(f"  .audio_tokens: {result.audio_tokens}")
