"""Sweep CFG crossover point and strength in late steps.

Two knobs: when CFG kicks in, and how strong it is.
"""

import math
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r: return 0.0 if not h else 1.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(r)][len(h)] / len(r)


def generate_cfg_scheduled(model, task, num_step, cfg_schedule, pos_temp=5.0, seed=42):
    from omnivoice.models.omnivoice import _get_time_steps
    torch.manual_seed(seed)
    C = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id

    inp = model._prepare_inference_inputs(
        task.texts[0], task.target_lens[0], task.ref_texts[0],
        task.ref_audio_tokens[0], task.langs[0], task.instructs[0], True)
    inp_ids = inp["input_ids"].squeeze(0)
    inp_amask = inp["audio_mask"].squeeze(0)
    c_len = inp_ids.size(1)
    t_len = task.target_lens[0]

    b2_ids = torch.full((2, C, c_len), mask_id, dtype=torch.long, device=model.device)
    b2_amask = torch.zeros((2, c_len), dtype=torch.bool, device=model.device)
    b2_attn = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=model.device)
    b2_ids[0, :, :c_len] = inp_ids
    b2_amask[0, :c_len] = inp_amask
    b2_attn[0, :, :c_len, :c_len] = True
    u_len = t_len
    b2_ids[1, :, :u_len] = inp_ids[:, -u_len:]
    b2_amask[1, :u_len] = inp_amask[-u_len:]
    b2_attn[1, :, :u_len, :u_len] = True
    if c_len > u_len:
        pd = torch.arange(u_len, c_len, device=model.device)
        b2_attn[1, :, pd, pd] = True

    b1_ids = b2_ids[0:1].clone()
    b1_amask = b2_amask[0:1].clone()
    b1_attn = b2_attn[0:1].clone()

    tokens = torch.full((C, t_len), mask_id, dtype=torch.long, device=model.device)
    layer_ids = torch.arange(C, device=model.device).view(-1, 1)

    timesteps = _get_time_steps(0.0, 1.0, num_step + 1, 0.1).tolist()
    total_mask = t_len * C
    rem = total_mask
    schedule = []
    for step in range(num_step):
        num = rem if step == num_step - 1 else min(
            math.ceil(total_mask * (timesteps[step+1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)

    fwd_passes = 0

    for step in range(num_step):
        cfg = cfg_schedule[step] if step < len(cfg_schedule) else 0.0

        if cfg > 0:
            b2_ids[0, :, c_len-t_len:c_len] = tokens
            b2_ids[1, :, :t_len] = tokens
            with torch.no_grad():
                logits = model(input_ids=b2_ids, audio_mask=b2_amask,
                               attention_mask=b2_attn).logits.to(torch.float32)
            fwd_passes += 2
            c_log = F.log_softmax(logits[0:1, :, c_len-t_len:c_len, :], dim=-1)
            u_log = F.log_softmax(logits[1:2, :, :t_len, :], dim=-1)
            log_probs = torch.log_softmax(c_log + cfg * (c_log - u_log), dim=-1)
        else:
            b1_ids[0, :, c_len-t_len:c_len] = tokens
            with torch.no_grad():
                logits = model(input_ids=b1_ids, audio_mask=b1_amask,
                               attention_mask=b1_attn).logits.to(torch.float32)
            fwd_passes += 1
            log_probs = F.log_softmax(logits[0:1, :, c_len-t_len:c_len, :], dim=-1)

        log_probs[..., mask_id] = float("-inf")
        pred_tokens = log_probs.argmax(dim=-1)[0]
        scores = log_probs.max(dim=-1)[0][0]
        scores = scores - (layer_ids * 5.0)
        if pos_temp > 0:
            g = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
            scores = scores / pos_temp + g
        is_masked = tokens == mask_id
        scores[~is_masked] = float("-inf")

        k = schedule[step]
        if k > 0:
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat = tokens.flatten()
            flat[topk_idx] = pred_tokens.flatten()[topk_idx]
            tokens = flat.view(C, t_len)

    return tokens, fwd_passes


def main():
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import GenerationTask
    from faster_whisper import WhisperModel

    print("Loading...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    model.eval()
    whisper = WhisperModel("base.en", device="cuda", compute_type="float16")

    segs, _ = whisper.transcribe("barth_ref.wav", language="en")
    actual_ref = " ".join(s.text.strip() for s in segs)
    prompt = model.create_voice_clone_prompt(
        ref_audio="barth_ref.wav", ref_text=actual_ref, preprocess_prompt=True)

    output_dir = Path("./sweep_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        "I can not believe you would do something like that. After everything we have been through together, you just threw it all away.",
        "Welcome to the annual science conference. Today we will explore the fascinating world of quantum computing and its implications for artificial intelligence.",
        "Once upon a time, in a land far far away, there lived a brave knight who feared nothing. But one dark night, everything changed.",
        "The committee has reviewed your proposal and we are pleased to inform you that your project has been approved for funding.",
    ]

    # Knob 1: crossover point (when CFG kicks in), at cfg=3
    print("=== CROSSOVER SWEEP (cfg=3 in active steps) ===")
    for start in range(9):  # 0=all CFG, 8=no CFG
        sched = [0.0]*start + [3.0]*(8-start)
        name = f"start{start}_cfg3"
        n_cfg = 8 - start
        wers = []
        fwd = 0
        for ti, text in enumerate(texts):
            rl = prompt.ref_audio_tokens.shape[1]
            cpf = max(1, len(prompt.ref_text) / rl)
            tl = max(25, min(200, int(len(text) / cpf)))
            task = GenerationTask(
                batch_size=1, texts=[text], target_lens=[tl], langs=["English"],
                instructs=["None"], ref_audio_tokens=[prompt.ref_audio_tokens],
                ref_texts=[prompt.ref_text], ref_rms=[getattr(prompt, "ref_rms", 0.1)])
            toks, fp = generate_cfg_scheduled(model, task, 8, sched)
            audio = model.audio_tokenizer.decode(toks.unsqueeze(0)).audio_values.squeeze().cpu().float()
            path = output_dir / f"{name}_{ti}.wav"
            torchaudio.save(str(path), audio.unsqueeze(0), 24000)
            segs2, _ = whisper.transcribe(str(path), language="en")
            transcript = " ".join(s.text.strip() for s in segs2)
            w = wer(text, transcript)
            wers.append(w)
            fwd = fp
        print(f"  CFG steps {start}-7 ({n_cfg} steps): fwd={fwd:2d} WER={np.mean(wers)*100:3.0f}%  sched={sched}")

    # Knob 2: CFG strength in last 2 steps
    print("\n=== CFG STRENGTH SWEEP (last 2 steps) ===")
    for cfg_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        sched = [0,0,0,0,0,0, cfg_val, cfg_val]
        name = f"last2_cfg{cfg_val}"
        wers = []
        fwd = 0
        for ti, text in enumerate(texts):
            rl = prompt.ref_audio_tokens.shape[1]
            cpf = max(1, len(prompt.ref_text) / rl)
            tl = max(25, min(200, int(len(text) / cpf)))
            task = GenerationTask(
                batch_size=1, texts=[text], target_lens=[tl], langs=["English"],
                instructs=["None"], ref_audio_tokens=[prompt.ref_audio_tokens],
                ref_texts=[prompt.ref_text], ref_rms=[getattr(prompt, "ref_rms", 0.1)])
            toks, fp = generate_cfg_scheduled(model, task, 8, sched)
            audio = model.audio_tokenizer.decode(toks.unsqueeze(0)).audio_values.squeeze().cpu().float()
            path = output_dir / f"{name}_{ti}.wav"
            torchaudio.save(str(path), audio.unsqueeze(0), 24000)
            segs2, _ = whisper.transcribe(str(path), language="en")
            transcript = " ".join(s.text.strip() for s in segs2)
            w = wer(text, transcript)
            wers.append(w)
            fwd = fp
        print(f"  last2 cfg={cfg_val:<3}: fwd={fwd:2d} WER={np.mean(wers)*100:3.0f}%")

    # Knob 2b: CFG strength in last 3 steps
    print("\n=== CFG STRENGTH SWEEP (last 3 steps) ===")
    for cfg_val in [1.0, 1.5, 2.0, 3.0]:
        sched = [0,0,0,0,0, cfg_val, cfg_val, cfg_val]
        name = f"last3_cfg{cfg_val}"
        wers = []
        fwd = 0
        for ti, text in enumerate(texts):
            rl = prompt.ref_audio_tokens.shape[1]
            cpf = max(1, len(prompt.ref_text) / rl)
            tl = max(25, min(200, int(len(text) / cpf)))
            task = GenerationTask(
                batch_size=1, texts=[text], target_lens=[tl], langs=["English"],
                instructs=["None"], ref_audio_tokens=[prompt.ref_audio_tokens],
                ref_texts=[prompt.ref_text], ref_rms=[getattr(prompt, "ref_rms", 0.1)])
            toks, fp = generate_cfg_scheduled(model, task, 8, sched)
            audio = model.audio_tokenizer.decode(toks.unsqueeze(0)).audio_values.squeeze().cpu().float()
            path = output_dir / f"{name}_{ti}.wav"
            torchaudio.save(str(path), audio.unsqueeze(0), 24000)
            segs2, _ = whisper.transcribe(str(path), language="en")
            transcript = " ".join(s.text.strip() for s in segs2)
            w = wer(text, transcript)
            wers.append(w)
            fwd = fp
        print(f"  last3 cfg={cfg_val:<3}: fwd={fwd:2d} WER={np.mean(wers)*100:3.0f}%")

    print("\nDone! Audio samples in ./sweep_samples/")


if __name__ == "__main__":
    main()
