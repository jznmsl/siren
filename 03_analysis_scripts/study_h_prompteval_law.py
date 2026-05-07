"""
PromptEval vs M7 — Faithful Point-Estimate Comparison (MMLU-Pro Law)
====================================================================
Same Algorithm 1 (Rasch IRT + pIRT) and Algorithm 2 (two-way balanced
sampling) as study_h_prompteval_gsm8k_v2.py, byte-identical to
Polo et al. NeurIPS 2024.  No CI is invented for PromptEval.

Metrics reported per cell (NO CI, NO coverage, NO width):
  - PE point estimate   S_hat_best
  - Bias vs theta_star  S_hat_best - theta_true_a   (procedure-level truth)
  - Directional call    sign(S_hat_best - theta_true_b) vs true sign
For comparison: M7 point estimate, M7 bias, M7 directional.

Scope: 11 models x 4 budgets x 2 tuners = 88 cells per method.

Required input layout (relative to working directory):
  results/multimodel/<MODEL>_law/                       # custom random search
    tensor_a_B500000.npy ... tensor_a_B6500000.npy
    study_e_results.json
  results/dspy_randsearch_multimodel/<MODEL>_law/       # DSPy
    tensor_a_B500000.npy ... tensor_a_B6500000.npy
    study_e_dspy_randsearch_results.json

Usage:
    cd ~/auto_research/llm_evaluation
    python study_h_prompteval_law.py

Outputs:
    results/study_h_prompteval_law_results.json     (per-cell results)
    Aggregate summary printed to stdout.

Runtime: ~1 minute on CPU (88 cells x 7 fractions, vectorized GD).
No GPU, no vLLM, no LLM calls.
"""
import json, os
import numpy as np
from pathlib import Path
from scipy.special import expit

# ============== Config ==============
RESULTS_BASE = Path("results")

NAME_MAP = {
    '01-ai_Yi-1.5-6B-Chat_law': 'Yi-1.5-6B',
    'Qwen_Qwen2-7B-Instruct_law': 'Qwen2-7B',
    'Qwen_Qwen2.5-3B-Instruct_law': 'Qwen2.5-3B',
    'Qwen_Qwen2.5-7B-Instruct_law': 'Qwen2.5-7B',
    'Qwen_Qwen3-8B_law': 'Qwen3-8B',
    'THUDM_glm-4-9b-chat_law': 'GLM-4-9B',
    'internlm_internlm2_5-7b-chat_law': 'InternLM2.5',
    'meta-llama_Llama-3.1-8B-Instruct_law': 'Llama-3.1',
    'microsoft_Phi-3.5-mini-instruct_law': 'Phi-3.5',
    'mistralai_Mistral-7B-Instruct-v0.1_law': 'Mistral-v0.1',
    'mistralai_Mistral-7B-Instruct-v0.3_law': 'Mistral-v0.3',
}
TUNERS = {
    'random_search': {
        'dir':  'multimodel',
        'study_e_filename': 'study_e_results.json',
    },
    'dspy': {
        'dir':  'dspy_randsearch_multimodel',
        'study_e_filename': 'study_e_dspy_randsearch_results.json',
    },
}
BUDGETS = [500_000, 1_500_000, 3_000_000, 6_500_000]
PE_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
SEED = 42
RIDGE = 0.01
RASCH_N_ITER = 200


# ============== Algorithm 2 (two-way balanced sampling) ==============
def two_way_balanced_sample(I, J, budget_cells, rng):
    if budget_cells >= I * J:
        return np.ones((I, J), dtype=bool)
    mask = np.zeros((I, J), dtype=bool)
    col_counts = np.zeros(J, dtype=np.int64)
    remaining = budget_cells
    while remaining > 0:
        prompt_order = rng.permutation(I)
        if remaining < I:
            prompt_order = prompt_order[:remaining]
        for i_hat in prompt_order:
            unobs = np.where(~mask[i_hat])[0]
            if len(unobs) == 0:
                continue
            min_cc = col_counts[unobs].min()
            cands = unobs[col_counts[unobs] == min_cc]
            j_hat = rng.choice(cands) if len(cands) > 1 else cands[0]
            mask[i_hat, j_hat] = True
            col_counts[j_hat] += 1
            remaining -= 1
    return mask


# ============== Algorithm 1 (Rasch IRT fit) ==============
def fit_rasch(Y_full, mask, n_iter=200, lr=0.5):
    I, J = Y_full.shape
    Y = Y_full.astype(np.float32)
    mask_f = mask.astype(np.float32)
    row_n = np.maximum(mask_f.sum(axis=1), 1).astype(np.float32)
    row_mean = np.clip((Y * mask_f).sum(axis=1) / row_n, 0.05, 0.95)
    theta = np.log(row_mean / (1 - row_mean)).astype(np.float32)
    col_n = np.maximum(mask_f.sum(axis=0), 1).astype(np.float32)
    col_mean = np.clip((Y * mask_f).sum(axis=0) / col_n, 0.05, 0.95)
    beta = -np.log(col_mean / (1 - col_mean)).astype(np.float32)
    for _ in range(n_iter):
        prob = expit(theta[:, None] - beta[None, :])
        resid = (Y - prob) * mask_f
        theta = theta + lr * (resid.sum(axis=1) - RIDGE * theta) / row_n
        beta  = beta  + lr * (-resid.sum(axis=0) - RIDGE * beta) / col_n
    return theta, beta


def prompteval_point(Y, budget_cells, seed):
    """Returns (best_idx, S_hat_best, n_observed). Y is [I=K prompts, J=M items]."""
    I, J = Y.shape
    rng = np.random.default_rng(seed)
    mask = two_way_balanced_sample(I, J, budget_cells, rng)
    theta, beta = fit_rasch(Y, mask, n_iter=RASCH_N_ITER)
    S_hat = np.zeros(I)
    for i in range(I):
        obs_j   = np.where(mask[i])[0]
        unobs_j = np.where(~mask[i])[0]
        lam = len(obs_j) / J
        obs_mean = Y[i, obs_j].mean() if len(obs_j) > 0 else 0.0
        imp = expit(theta[i] - beta[unobs_j]).mean() if len(unobs_j) > 0 else 0.0
        S_hat[i] = lam * obs_mean + (1 - lam) * imp
    best_idx = int(np.argmax(S_hat))
    return best_idx, float(S_hat[best_idx]), int(mask.sum())


# ============== Main ==============
def directional_call(point, theta_b):
    """Returns 'A>B' or 'A<B' based on point estimate vs theta_b."""
    return 'A>B' if point > theta_b else 'A<B'

def true_dir(theta_a, theta_b):
    return 'A>B' if theta_a > theta_b else 'A<B'


all_results = {}  # tuner -> model -> budget -> {...}

for tuner_name, tuner_cfg in TUNERS.items():
    all_results[tuner_name] = {}
    print(f"\n{'='*90}\nTuner: {tuner_name}\n{'='*90}")
    for model_full, model_short in NAME_MAP.items():
        d = RESULTS_BASE / tuner_cfg['dir'] / model_full
        se_path = d / tuner_cfg['study_e_filename']
        if not se_path.exists():
            print(f"  [skip] {model_short}: missing {se_path.name}")
            continue
        with open(se_path) as f:
            study = json.load(f)
        all_results[tuner_name][model_short] = {}
        for B in BUDGETS:
            tpath = d / f'tensor_a_B{B}.npy'
            if not tpath.exists():
                continue
            tensor = np.load(tpath).astype(np.float32)  # [M, K]
            M, K = tensor.shape
            Y = tensor.T                                 # [I=K, J=M]
            bkey = str(B)
            if bkey not in study['results']:
                continue
            r_b = study['results'][bkey]
            theta_a = r_b['ground_truth']['theta_true_a']
            theta_b = r_b['ground_truth']['theta_true_b']
            m7_pt   = r_b['m7']['system_a']['theta']
            m1_pt   = r_b['comparison']['m1']
            t_dir   = true_dir(theta_a, theta_b)

            # M7 metrics
            m7_bias  = m7_pt - theta_a
            m7_dcall = directional_call(m7_pt, theta_b)
            m7_dir_correct = (m7_dcall == t_dir)

            # PE metrics across fractions
            pe_at_frac = {}
            for frac in PE_FRACTIONS:
                cells = max(K, int(np.ceil(frac * M * K)))
                cells = min(cells, M * K)
                bi, pe_pt, n_obs = prompteval_point(Y, cells, seed=SEED)
                pe_bias  = pe_pt - theta_a
                pe_dcall = directional_call(pe_pt, theta_b)
                pe_dir_correct = (pe_dcall == t_dir)
                pe_at_frac[f"{frac:.2f}"] = {
                    'fraction': frac, 'cells': cells,
                    'best_idx': bi,
                    'point': round(pe_pt, 6),
                    'bias_vs_theta_star': round(pe_bias, 6),
                    'directional_call': pe_dcall,
                    'directional_correct': pe_dir_correct,
                    'agrees_with_m1_pt': abs(pe_pt - m1_pt) < 1e-3,
                }

            all_results[tuner_name][model_short][bkey] = {
                'M': M, 'K': K,
                'theta_true_a': theta_a,
                'theta_true_b': theta_b,
                'true_dir': t_dir,
                'm7': {
                    'point': round(m7_pt, 6),
                    'bias_vs_theta_star': round(m7_bias, 6),
                    'directional_call': m7_dcall,
                    'directional_correct': m7_dir_correct,
                },
                'm1_point': m1_pt,
                'prompteval': pe_at_frac,
            }
        print(f"  [done] {model_short}: {len(all_results[tuner_name][model_short])} budgets")

# Save
out_path = RESULTS_BASE / 'study_h_prompteval_law_results.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved {out_path}")

# ====================== Aggregate ======================
import statistics

print("\n" + "=" * 90)
print("AGGREGATE — PromptEval vs M7 on MMLU-Pro Law (88 cells per method)")
print("=" * 90)

for tuner in ['random_search', 'dspy']:
    print(f"\n--- {tuner.upper()} ---")
    cells = []
    for model in all_results[tuner]:
        for B in all_results[tuner][model]:
            cells.append(all_results[tuner][model][B])
    n = len(cells)
    print(f"  N = {n} cells (= 11 models * 4 budgets)")

    # M7 stats
    m7_biases = [c['m7']['bias_vs_theta_star'] for c in cells]
    m7_dir_ok = sum(c['m7']['directional_correct'] for c in cells)
    print(f"\n  M7:  bias mean={np.mean(m7_biases)*100:+.2f}pp  std={np.std(m7_biases)*100:.2f}pp  "
          f"max|b|={max(abs(b) for b in m7_biases)*100:.2f}pp")
    print(f"       directional correct: {m7_dir_ok}/{n}")

    # PE stats per fraction
    print(f"\n  PE @ each fraction:")
    print(f"    {'frac':>5s}   {'mean bias':>14s}   {'std':>7s}   {'max |bias|':>10s}   {'dir correct':>11s}")
    for frac in PE_FRACTIONS:
        fkey = f"{frac:.2f}"
        biases = [c['prompteval'][fkey]['bias_vs_theta_star'] for c in cells]
        dir_ok = sum(c['prompteval'][fkey]['directional_correct'] for c in cells)
        print(f"    {frac:>5.2f}   {np.mean(biases)*100:>+10.2f}pp  {np.std(biases)*100:>5.2f}pp  "
              f"{max(abs(b) for b in biases)*100:>8.2f}pp     {dir_ok}/{n}")

print("\n" + "=" * 90)
print("Sanity check: PE @ frac=1.00 should equal M1 in point estimate")
print("=" * 90)
for tuner in ['random_search', 'dspy']:
    n_match = 0; n_tot = 0
    for model in all_results[tuner]:
        for B in all_results[tuner][model]:
            c = all_results[tuner][model][B]
            n_tot += 1
            if c['prompteval']['1.00']['agrees_with_m1_pt']:
                n_match += 1
    print(f"  {tuner}: PE@1.00 ~ M1 in {n_match}/{n_tot} cells")
