# OmniVoice Speculative Decode Distillation — Learnings

## What We Tried (2026-04-06/07)

### The Goal
Stream OmniVoice (a masked/unmasking TTS model) by training a small AR draft model to predict codebook-0 tokens, verified by the teacher for quality.

### Architecture Evolution

**v1 — Hidden state conditioning**
- Draft conditioned on teacher's LLM hidden states
- 91% accuracy on PyTorch, 0% on MLX
- Root cause: LLM hidden states differ numerically between PyTorch (CUDA) and MLX (Metal) due to accumulated floating-point differences through 28 transformer layers
- Noise augmentation and fine-tuning couldn't bridge the gap

**v2 — Token conditioning (framework-independent)**
- Draft uses frozen teacher embedding weights, conditions on raw token IDs
- Embeddings are just lookups (no computation) → identical across frameworks
- 97.8% accuracy on cached data, works identically on PyTorch and MLX
- But still 0% on real speech — see below

### Why It Fails on Real Speech

**The simplified fallback was the wrong teacher.**
Our `_run_unmasking_loop` is a hand-written approximation:
- Linear timestep schedule (uniform token count per step)
- No CFG (single forward pass, no classifier-free guidance)
- No Gumbel noise (pure argmax)
- Produces LOW entropy output (2-5 unique cb0 tokens per utterance)

The real `_generate_iterative` uses:
- Shifted timestep schedule (t_shift=0.1, front-loaded)
- CFG with guidance_scale=3.0 (two forward passes per step)
- Gumbel noise for position selection (position_temperature=5.0)
- Produces HIGH entropy output (30-60 unique cb0 tokens per utterance)

**Results by generation method:**

| Generation | CB0 Entropy | Draft Accuracy | Notes |
|---|---|---|---|
| `_run_unmasking_loop` (simplified) | Low (2-5 unique) | 97% | Easy to memorize, wrong algorithm |
| `generate_iterative` pos_temp=0 (deterministic MLX) | Medium (11 unique) | 77% | Better but still not the real thing |
| `_generate_iterative` (real OmniVoice) | High (30-60 unique) | 5% | Correct algorithm, too stochastic |
| Real + KL soft targets | N/A | 95.9% on cache, 4% on live | Soft targets from single-pass don't match 8-step generation |

### The Fundamental Mismatch

Speculative decoding assumes an **autoregressive** teacher where:
1. Teacher generates token-by-token
2. Draft predicts the next token
3. Teacher verifies: "would I have generated this?"

OmniVoice's teacher is **parallel iterative unmasking**:
1. Start with all masks
2. Each step: predict ALL positions, unmask the most confident K
3. Repeat 8 times with different K per step
4. Final tokens depend on the ORDER of unmasking (stochastic)

An AR draft can predict what a single forward pass would argmax to. But it can't predict the outcome of an 8-step iterative refinement where each step's context depends on previous steps' stochastic selections.

### What We Proved Works
- v2 token-conditioned architecture: framework-independent conditioning ✓
- Training pipeline: cache → RunPod 5090 → train → convert to MLX ✓
- Infrastructure: Tailscale funnel for file transfer, SSH to RunPod ✓
- 97% accuracy IS achievable when training and inference use the same simplified generation

### Cost
- RunPod 5090: ~$5-8 total across all sessions
- Total compute time: ~2 hours of GPU

---

## Promising Next Approaches

### 1. Chunk-Parallel Draft (Most Natural Fit)

Instead of predicting one token at a time (AR), predict a CHUNK of N tokens in parallel — mimicking how the teacher actually generates.

**Architecture:**
```
Input: [conditioning tokens]
Draft predicts: N masked positions simultaneously (like a mini-unmasking step)
Teacher verifies: run one real unmasking step, compare
```

This matches the teacher's actual behavior. The draft learns to do what the teacher does in one step, not to predict the outcome of 8 steps.

**Why it should work:**
- Single-step prediction IS learnable (we proved 97% for single-pass argmax)
- The teacher's per-step logits are cached correctly (no multi-step mismatch)
- Verification is natural: teacher runs 1 step, compares with draft's 1 step
- Speedup: draft does 1 step cheap, replaces K of the teacher's 8 steps

**Implementation:**
1. Cache teacher's per-step logits (not just final tokens) — save the logits at each of the 8 unmasking steps
2. Train draft to predict the teacher's step-K logits given the partially-unmasked sequence from step K-1
3. At inference: draft predicts step 1-4, teacher does steps 5-8 (or teacher verifies draft's steps)

### 2. Consistency Distillation (From Diffusion Literature)

OmniVoice's iterative unmasking is essentially discrete diffusion. Consistency models (Song et al. 2023) distill multi-step diffusion into 1-2 steps.

**Apply to discrete tokens:**
- Train a model that maps (fully masked sequence + conditioning) → final tokens in 1 step
- This IS what the simplified `_run_unmasking_loop` does (single-pass argmax)
- But train it on the REAL teacher's final outputs (not single-pass predictions)
- Use consistency loss: model(x_t) should match model(x_t') for any two noise levels on the same trajectory

**Why it might work:**
- ConsistencyTTS and CoMoSpeech have shown this works for continuous diffusion TTS
- Reduces 8 steps to 1-2 steps (4-8x speedup)
- No AR → no error propagation
- Naturally parallel (like the teacher)

### 3. Progressive Distillation

Train the draft to match the teacher with fewer steps:
1. Teacher: 8 steps → Student: 4 steps (learn to skip every other step)
2. Student-4: 4 steps → Student-2: 2 steps
3. Student-2: 2 steps → Student-1: 1 step

Each halving is a small distribution shift that's learnable. The jump from 8→1 in one go is too large (what we tried), but 8→4→2→1 should work.

### 4. EAGLE-3 Style (Hidden State Prediction)

Instead of predicting tokens, predict the teacher's HIDDEN STATES at the next step. Then use those hidden states directly with the teacher's audio heads.

**Why it avoids our problem:**
- Hidden states are continuous → smoother distribution to learn
- Audio heads (frozen, shared) convert hidden states to tokens
- The draft only needs to predict the LLM's behavior, not the full generation pipeline

**Caveat:** This is what v1 tried (hidden state conditioning). The framework gap killed it. Would need to deploy on same framework, or use the token-conditioning trick from v2 as input while predicting hidden states as output.

### 5. Lookahead Decoding (No Draft Model)

Recent work (2024-2025) on lookahead decoding generates tokens by exploiting n-gram patterns in the teacher's own generation. No separate draft model needed.

For OmniVoice: analyze the teacher's typical unmasking patterns, identify that certain positions are always unmasked in the same order with the same tokens, and pre-compute those.

---

## Recommendation

**Start with Approach 1 (Chunk-Parallel Draft)** because:
- Most natural mapping to the teacher's actual algorithm
- The per-step logits ARE predictable (we proved single-pass works)
- No AR → no error propagation
- The caching is straightforward: save logits at each unmasking step
- Can reuse the v2 token-conditioning architecture

**Key change from current approach:**
- Current: predict final cb0 tokens one at a time (impossible for stochastic 8-step process)
- New: predict what ONE step of unmasking produces given the current partially-unmasked state

This is a much simpler prediction task that we've already shown is learnable.

---

## Infrastructure Notes

- RunPod 5090 via Tailscale: `root@100.75.219.41` (SSH works from macbook — Claude Code can drive it directly)
- Mac Mini: `johannescarlsten@mac-mini` (Tailscale)
- Tailscale funnel on Mac Mini for file serving (port 9999)
- LibriTTS-R train-clean-360 extracted on RunPod: 116k files, 904 speakers
- Manifest with 855 speakers: `data/libritts_big/manifest.json`
- GitHub: `jourheighness/omnivoice-distill`
- OmniVoice MLX weights on Mac Mini: `~/bartholomew/source/voice-service/weights/omnivoice_mlx/`

### Running Everything via RunPod

The PT→MLX framework gap means training AND inference should ideally happen on the same framework. For development/iteration, run everything on RunPod (PyTorch/CUDA):

- **Caching:** `python3 src/cache_v2_real_api.py` — uses real `_generate_iterative` via GenerationTask API
- **Training:** `python3 src/train_v2_kl.py` — KL distillation with soft targets
- **Testing:** `python3 src/demo_pt.py` — end-to-end demo with real speech + audio output
- **Voice encoding:** `python3 src/encode_voice.py barth_ref.wav` — encode any voice for inference
- **Speaker manifest:** `python3 src/build_manifest.py --num_samples 5000` — index 904 speakers

All scripts run on RunPod via SSH (`root@100.75.219.41`). Claude Code can SSH in directly and run everything without manual copy-paste.

Install Tailscale on new pods:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscaled --tun=userspace-networking &
sleep 2
tailscale up --authkey=YOUR_KEY
```

Only move to MLX once the approach is validated end-to-end on PyTorch. The v2 token-conditioning architecture makes the final PT→MLX conversion straightforward (just convert weights, embeddings are identical).
