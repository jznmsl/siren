#!/usr/bin/env python
"""
Full Tensor Build: 2000 items × 12 prompts × Qwen3-8B (no-think)
==================================================================
Builds the complete score tensor for Stage 2 offline analysis.
Estimated time: ~30 minutes on RTX 4090.

Saves checkpoint after every prompt (so you don't lose progress if interrupted).

Usage:
  python build_tensor.py                # full 2000 items
  python build_tensor.py --items 500    # smaller test run

Output:
  results/score_tensor.npy       — shape [N_items, 12], values 0/1
  results/token_costs.npy        — shape [N_items, 12, 2], (prompt_tok, completion_tok)
  results/tensor_metadata.json   — item IDs, prompt names, model info, timing
"""
import json
import time
import random
import re
import sys
import os
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ============================================================
#  Config
# ============================================================
N_ITEMS = 2000
for arg in sys.argv[1:]:
    if arg.startswith("--items"):
        N_ITEMS = int(sys.argv[sys.argv.index(arg) + 1])

MODEL = "Qwen/Qwen3-8B"
SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
#  Step 1: Sample items
# ============================================================
print(f"{'='*65}")
print(f"  FULL TENSOR BUILD")
print(f"  Model: {MODEL}")
print(f"  Items: {N_ITEMS}")
print(f"  Prompts: 12")
print(f"{'='*65}\n")

print("Loading MMLU-Pro...", flush=True)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
print(f"Total items in dataset: {len(ds)}")

random.seed(SEED)
indices = random.sample(range(len(ds)), N_ITEMS)
items = [ds[i] for i in indices]
print(f"Sampled {len(items)} items (seed={SEED})\n")

# Save item IDs for reproducibility
item_ids = [item.get("question_id", i) for i, item in enumerate(items)]

# ============================================================
#  Step 2: Prompt templates (same 12 as pilot)
# ============================================================
def format_options(options):
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))

TEMPLATES = {
    "P01_detailed_direct": {
        "name": "Detailed Direct",
        "system": "You are a helpful assistant.",
        "user": (
            "Answer the following multiple-choice question. "
            "Output ONLY the letter of the correct answer "
            "(A, B, C, D, E, F, G, H, I, or J). "
            "Do not include any explanation.\n\n"
            "Question: {question}\n\n{options}\n\nAnswer:"
        ),
    },
    "P02_brief_direct": {
        "name": "Brief Direct",
        "system": "You are a helpful assistant.",
        "user": (
            "Choose the correct answer. Reply with one letter only.\n\n"
            "{question}\n\n{options}\n\nAnswer:"
        ),
    },
    "P03_minimal": {
        "name": "Minimal",
        "system": "",
        "user": "Q: {question}\n{options}\nA:",
    },
    "P04_analyze": {
        "name": "Analyze Then Answer",
        "system": "You are a helpful assistant.",
        "user": (
            "Read the question carefully. Consider each option. "
            "Then output the letter of the best answer.\n\n"
            "Question: {question}\n\n{options}\n\nThe best answer is:"
        ),
    },
    "P05_eliminate": {
        "name": "Eliminate Wrong",
        "system": "You are a helpful assistant.",
        "user": (
            "Read the question. Eliminate clearly wrong options first, "
            "then select the best remaining option. "
            "Output one letter only.\n\n"
            "Question: {question}\n\n{options}\n\nAnswer:"
        ),
    },
    "P06_confidence": {
        "name": "Confidence-based",
        "system": "You are a helpful assistant.",
        "user": (
            "Answer the question below. If you are confident, just give "
            "the letter. If unsure, briefly reason then give the letter.\n\n"
            "Question: {question}\n\n{options}\n\nAnswer:"
        ),
    },
    "P07_expert": {
        "name": "Expert Professor",
        "system": "You are an expert professor with deep knowledge across all academic disciplines.",
        "user": (
            "Answer the following question with confidence. "
            "Output ONLY the letter of the correct answer.\n\n"
            "Question: {question}\n\n{options}\n\nThe correct answer is:"
        ),
    },
    "P08_student": {
        "name": "Careful Student",
        "system": "You are a careful student taking an important exam.",
        "user": (
            "Think carefully and select the best answer. "
            "Output one letter only.\n\n"
            "Question: {question}\n\n{options}\n\nMy answer is:"
        ),
    },
    "P09_no_role": {
        "name": "No Role",
        "system": "",
        "user": (
            "What is the correct answer to this question? "
            "Output one letter.\n\n"
            "Question: {question}\n\n{options}\n\nAnswer:"
        ),
    },
    "P10_answer_first": {
        "name": "Answer First",
        "system": "You are a helpful assistant.",
        "user": (
            "State your answer letter first.\n\n"
            "Question: {question}\n\n{options}\n\nAnswer letter:"
        ),
    },
    "P11_structured": {
        "name": "Structured Output",
        "system": "You are a helpful assistant.",
        "user": (
            "Answer the question below. Format your response as:\n"
            "Answer: [single letter A-J]\n\n"
            "Question: {question}\n\n{options}"
        ),
    },
    "P12_json_style": {
        "name": "JSON-style",
        "system": "You are a helpful assistant that responds concisely.",
        "user": (
            "Respond with exactly one character (A-J) representing "
            "the correct option for this question.\n\n"
            "Question: {question}\n\n{options}\n\nAnswer:"
        ),
    },
}

prompt_keys = list(TEMPLATES.keys())
N_PROMPTS = len(prompt_keys)

# ============================================================
#  Step 3: Answer extraction
# ============================================================
VALID = set("ABCDEFGHIJ")

def extract_answer(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if text and text[0] in VALID:
        return text[0]
    m = re.search(r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([A-J])', text)
    if m:
        return m.group(1)
    matches = re.findall(r'\b([A-J])\b', text)
    if matches:
        return matches[-1]
    for ch in reversed(text):
        if ch in VALID:
            return ch
    return "?"

def get_true_answer(item):
    ans = item.get("answer", "")
    if isinstance(ans, int):
        return "ABCDEFGHIJ"[ans]
    ans = str(ans).strip().upper()
    if ans in VALID:
        return ans
    idx = item.get("answer_index", None)
    if idx is not None and isinstance(idx, int):
        return "ABCDEFGHIJ"[idx]
    return ans

# ============================================================
#  Step 4: Check for existing checkpoints (resume support)
# ============================================================
checkpoint_path = os.path.join(RESULTS_DIR, "tensor_checkpoint.npz")
score_tensor = np.full((N_ITEMS, N_PROMPTS), -1, dtype=np.int8)  # -1 = not done
token_costs = np.zeros((N_ITEMS, N_PROMPTS, 2), dtype=np.int32)  # (prompt_tok, compl_tok)
timing = {}
start_prompt_idx = 0

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint: {checkpoint_path}")
    ckpt = np.load(checkpoint_path)
    old_scores = ckpt["scores"]
    old_tokens = ckpt["tokens"]
    # Copy over completed prompts
    for k in range(N_PROMPTS):
        if np.all(old_scores[:, k] >= 0):
            score_tensor[:, k] = old_scores[:, k]
            token_costs[:, k] = old_tokens[:, k]
            start_prompt_idx = k + 1
    print(f"Resuming from prompt index {start_prompt_idx} ({prompt_keys[start_prompt_idx] if start_prompt_idx < N_PROMPTS else 'DONE'})\n")

if start_prompt_idx >= N_PROMPTS:
    print("All prompts already completed! Skipping to summary.")
else:
    # ============================================================
    #  Step 5: Load model
    # ============================================================
    print(f"Loading {MODEL} (no-think mode)...", flush=True)
    t0 = time.time()
    llm = LLM(model=MODEL, gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=16)
    model_load_time = time.time() - t0
    print(f"Model loaded in {model_load_time:.1f}s\n")

    sampling = SamplingParams(temperature=0, max_tokens=64)

    # ============================================================
    #  Step 6: Run inference prompt by prompt
    # ============================================================
    total_t0 = time.time()

    for k in range(start_prompt_idx, N_PROMPTS):
        tmpl_key = prompt_keys[k]
        tmpl = TEMPLATES[tmpl_key]

        # Build conversations
        conversations = []
        for item in items:
            opts = format_options(item["options"])
            user_msg = tmpl["user"].format(question=item["question"], options=opts)
            msgs = []
            if tmpl["system"]:
                msgs.append({"role": "system", "content": tmpl["system"]})
            msgs.append({"role": "user", "content": user_msg + "\n/no_think"})
            conversations.append(msgs)

        print(f"[{k+1}/{N_PROMPTS}] Running {tmpl_key} ({tmpl['name']}) "
              f"on {N_ITEMS} items...", flush=True)
        t1 = time.time()
        outputs = llm.chat(conversations, sampling_params=sampling)
        dt = time.time() - t1

        # Score and record tokens
        correct = 0
        failures = 0
        for i, output in enumerate(outputs):
            raw = output.outputs[0].text
            predicted = extract_answer(raw)
            true_ans = get_true_answer(items[i])

            n_prompt_tok = len(output.prompt_token_ids)
            n_compl_tok = len(output.outputs[0].token_ids)

            score_tensor[i, k] = 1 if predicted == true_ans else 0
            token_costs[i, k, 0] = n_prompt_tok
            token_costs[i, k, 1] = n_compl_tok

            if predicted == true_ans:
                correct += 1
            if predicted == "?":
                failures += 1

        acc = correct / N_ITEMS
        fail_rate = failures / N_ITEMS
        avg_tok = token_costs[:, k].sum(axis=1).mean()
        timing[tmpl_key] = dt

        print(f"    Acc: {acc:.1%}  |  Fail: {fail_rate:.1%}  |  "
              f"Time: {dt:.1f}s  |  Avg tokens: {avg_tok:.0f}/item")

        # Save checkpoint
        np.savez(checkpoint_path, scores=score_tensor, tokens=token_costs)
        print(f"    Checkpoint saved.\n")

    total_time = time.time() - total_t0
    print(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f} min)\n")

# ============================================================
#  Step 7: Save final outputs
# ============================================================
# Score tensor
tensor_path = os.path.join(RESULTS_DIR, "score_tensor.npy")
np.save(tensor_path, score_tensor)

# Token costs
costs_path = os.path.join(RESULTS_DIR, "token_costs.npy")
np.save(costs_path, token_costs)

# Metadata
meta = {
    "model": MODEL,
    "n_items": N_ITEMS,
    "n_prompts": N_PROMPTS,
    "prompt_keys": prompt_keys,
    "prompt_names": [TEMPLATES[k]["name"] for k in prompt_keys],
    "item_ids": item_ids[:20],  # first 20 for reference
    "seed": SEED,
    "no_think": True,
    "timing": timing,
    "tensor_shape": list(score_tensor.shape),
}
meta_path = os.path.join(RESULTS_DIR, "tensor_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# ============================================================
#  Step 8: Summary
# ============================================================
print("=" * 70)
print("  TENSOR BUILD COMPLETE")
print("=" * 70)
print(f"  Model:  {MODEL}")
print(f"  Items:  {N_ITEMS}")
print(f"  Prompts: {N_PROMPTS}")
print(f"\n  Per-prompt accuracy:")
print(f"  {'Prompt':<25s}  {'Acc':>6s}  {'Tok/item':>8s}")
print(f"  {'─'*45}")

accs = []
for k, key in enumerate(prompt_keys):
    acc = score_tensor[:, k].mean()
    avg_tok = token_costs[:, k].sum(axis=1).mean()
    accs.append(acc)
    print(f"  {key:<25s}  {acc:>5.1%}  {avg_tok:>8.0f}")

print(f"  {'─'*45}")
print(f"  Best:  {prompt_keys[np.argmax(accs)]} ({max(accs):.1%})")
print(f"  Worst: {prompt_keys[np.argmin(accs)]} ({min(accs):.1%})")
print(f"  Spread: {max(accs)-min(accs):.1%}")
print(f"  Total tokens: {token_costs.sum():,}")

print(f"\n  Files saved:")
print(f"    {tensor_path}  — score tensor {score_tensor.shape}")
print(f"    {costs_path}  — token costs {token_costs.shape}")
print(f"    {meta_path}  — metadata")

# Clean up checkpoint
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print(f"    (checkpoint removed)")

print(f"\n  Next step: run  python analyze_tensor.py  for Stage 2 analysis.")
print(f"  Done!\n")
