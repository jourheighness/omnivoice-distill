"""Encode a voice reference on CUDA and save tokens for inference."""

import sys
import torch
import numpy as np
from omnivoice import OmniVoice

audio_path = sys.argv[1] if len(sys.argv) > 1 else "barth_ref.wav"
output_path = sys.argv[2] if len(sys.argv) > 2 else "barth_cuda_tokens.npz"

print(f"Encoding {audio_path}...")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
prompt = model.create_voice_clone_prompt(ref_audio=audio_path, ref_text="Reference audio.", preprocess_prompt=True)
np.savez(output_path, ref_audio_tokens=prompt.ref_audio_tokens.cpu().numpy(), ref_text=str(prompt.ref_text))
print(f"Saved: {prompt.ref_audio_tokens.shape} to {output_path}")
