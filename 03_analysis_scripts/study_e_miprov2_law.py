"""
Study E — Real MIPROv2 Version (Law subject, multi-model)
============================================================
This version uses ACTUAL dspy.teleprompt.MIPROv2 for Phase 1 search
(not random search). Budget-aware num_trials controls MIPROv2 search depth.

Usage:
    python study_e_miprov2_law.py \
        --model Qwen/Qwen3-8B \
        --port 8000 \
        --results-dir results/miprov2_multimodel/Qwen_Qwen3-8B_law

Pipeline:
    For each budget B:
        1. MIPROv2 with num_trials = map(B)
           - Phase 1a: bootstrap demos (inside MIPROv2)
           - Phase 1b: propose instructions (inside MIPROv2)
           - Phase 1c: Bayesian search with Optuna TPE (inside MIPROv2)
        2. Extract top-K=10 candidates from best_program.candidate_programs
        3. Phase 2: for each top-K program, evaluate on full eval set (1101 items)
           → build tensor [M x K]
        4. Phase 2.5: run M1-M7 methods + ground truth
        5. Save results in same format as random search

Requirements: vLLM server already running on --port. `pip install optuna`.
"""

import argparse
import json
import re
import time
import os
import sys
import traceback
from pathlib import Path
import numpy as np

# ----- CLI -----
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--results-dir", required=True)
parser.add_argument("--subject", default="law",
                    help="MMLU-Pro subject")
parser.add_argument("--budgets", default="500000,1500000,3000000,6500000")
# Budget -> num_trials mapping for MIPROv2, based on token-budget equivalence.
# Empirical estimate: ~75K tokens per trial (minibatch 50 × ~1.5K tokens/item).
# Subtracting ~150K overhead (bootstrap + propose + final eval), trials ≈ (B - 150K) / 75K.
#   B=500K   → ~5 trials (we use 7, adding margin for light-equivalent)
#   B=1.5M   → ~18 trials (use 20)
#   B=3M     → ~38 trials (use 40, close to DSPy heavy preset)
#   B=6.5M   → ~85 trials (heavy+)
parser.add_argument("--trials-per-budget", default="7,20,40,85",
                    help="Comma-separated num_trials per budget (token-aligned)")
parser.add_argument("--K", type=int, default=10, help="Top-K candidates for tensor")
parser.add_argument("--R", type=int, default=10, help="Repetitions for rep-split")
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--n-boot", type=int, default=2000)
parser.add_argument("--n-train", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-bootstrapped-demos", type=int, default=2)
parser.add_argument("--max-labeled-demos", type=int, default=2)
parser.add_argument("--num-candidates", type=int, default=10,
                    help="Number of instruction candidates MIPROv2 proposes")
parser.add_argument("--minibatch-size", type=int, default=50)
args = parser.parse_args()

BUDGETS = [int(b) for b in args.budgets.split(",")]
TRIALS = [int(t) for t in args.trials_per_budget.split(",")]
assert len(BUDGETS) == len(TRIALS), f"budgets ({len(BUDGETS)}) must match trials ({len(TRIALS)})"
RESULTS_DIR = Path(args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"STUDY E — REAL MIPROv2 | {args.model} | subject={args.subject}")
print(f"Budgets: {BUDGETS}")
print(f"Num_trials per budget: {TRIALS}")
print(f"K={args.K} R={args.R} rho={args.rho} tau={args.tau}")
print(f"bootstrap={args.max_bootstrapped_demos} labeled={args.max_labeled_demos} "
      f"num_candidates={args.num_candidates}")
print("=" * 70)


# ----- Import DSPy (after argparse so we can fail fast on CLI errors) -----
import dspy
from dspy.teleprompt import MIPROv2

# Verify optuna is installed (MIPROv2 needs it)
try:
    import optuna
    print(f"[Setup] optuna version: {optuna.__version__}")
except ImportError:
    print("ERROR: optuna required for MIPROv2. Install with:")
    print("    pip install optuna")
    sys.exit(1)

# Suppress verbose optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----- Configure DSPy LM (vLLM-served, same for task and prompt) -----
# max_tokens=1024: Even with /no_think in the signature, some models
# may still emit short reasoning. 1024 is safe ceiling. Cost per call
# is bounded by actual output, not max_tokens, so this doesn't hurt budget.
lm = dspy.LM(
    f"openai/{args.model}",
    api_base=f"http://localhost:{args.port}/v1",
    api_key="dummy",
    max_tokens=1024,
    temperature=0.0,
    cache=False,  # disable caching so evaluations are independent
)
dspy.configure(lm=lm, track_usage=True)
print(f"[Setup] DSPy configured with {args.model} @ port {args.port} "
      f"(max_tokens=1024, track_usage=True)")


# ----- Data loading -----
items = []
with open("data/mmlu_pro_test.json") as f:
    for line in f:
        item = json.loads(line.strip())
        if item.get("category") == args.subject:
            items.append(item)

if len(items) == 0:
    print(f"ERROR: no items found for subject={args.subject!r}")
    sys.exit(1)

rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(items))
trainset_raw = [items[i] for i in perm[:args.n_train]]
evalset_raw = [items[i] for i in perm[args.n_train:]]
M = len(evalset_raw)
print(f"[Data] Subject={args.subject} | Total: {len(items)} | "
      f"Train: {len(trainset_raw)} | Eval: {M}")


# ----- Convert to DSPy examples -----
def format_options(options):
    return "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))


def to_dspy_example(item):
    """Convert MMLU-Pro item to dspy.Example."""
    return dspy.Example(
        question=item["question"],
        options=format_options(item["options"]),
        answer=item["answer"],  # A, B, C, ...
    ).with_inputs("question", "options")


dspy_trainset = [to_dspy_example(it) for it in trainset_raw]
dspy_evalset = [to_dspy_example(it) for it in evalset_raw]
print(f"[DSPy] Converted to DSPy examples: "
      f"train={len(dspy_trainset)}, eval={len(dspy_evalset)}")


# ----- Define DSPy program -----
class MCQAnswer(dspy.Signature):
    """Answer a multiple-choice question by outputting the correct letter."""
    question = dspy.InputField(desc="The question to answer")
    options = dspy.InputField(desc="The answer options, one per line, labeled A through J")
    answer = dspy.OutputField(desc="The single letter (A-J) of the correct answer")


class MCQClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(MCQAnswer)

    def forward(self, question, options):
        return self.classify(question=question, options=options)


# ----- Metric -----
def extract_letter(s):
    """Extract a single A-J letter from model output."""
    if not s:
        return "?"
    s = str(s).strip()
    # strip common reasoning markers
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()
    # first char if it's a letter
    if s and s[0] in "ABCDEFGHIJ":
        rest = s[1:].strip()
        if not rest or rest[0] in ".\n,) ":
            return s[0]
    m = re.search(r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([A-J])', s)
    if m:
        return m.group(1)
    matches = re.findall(r'\b([A-J])\b', s)
    return matches[-1] if matches else "?"


def mcq_metric(example, pred, trace=None):
    """Binary correctness: 1 if predicted letter matches gold."""
    try:
        pred_letter = extract_letter(pred.answer)
        return int(pred_letter == example.answer)
    except Exception:
        return 0


# ----- Helpers for Phase 2 (tensor build) -----
def evaluate_program_on_set(program, dataset):
    """Run a (compiled) DSPy program on every example in dataset.
    Return binary [0,1] scores array.
    """
    binary = []
    for ex in dataset:
        try:
            pred = program(question=ex.question, options=ex.options)
            ok = int(extract_letter(pred.answer) == ex.answer)
        except Exception as e:
            ok = 0
        binary.append(ok)
    return np.array(binary, dtype=float)


# ----- Inference methods (M1-M7, ground truth) -----
def softmax(x, tau):
    z = x / tau - (x / tau).max()
    e = np.exp(z)
    return e / e.sum()


def run_all_methods(scores, rho, tau, R, n_boot, seed):
    """Same inference methods as random search version for direct comparison."""
    lr = np.random.default_rng(seed)
    M_, K = scores.shape
    n_dev = int(rho * M_)

    # M1/M2: naive best-of (argmax column mean)
    col_means = scores.mean(axis=0)
    m1 = float(col_means.max())
    m2 = m1
    se = np.sqrt(m1 * (1 - m1) / M_)
    m3 = [m1, [m1 - 1.96*se, m1 + 1.96*se]]

    Y4, Y5, thetas = [], [], []
    psi6 = np.zeros(M_)
    psi7 = np.zeros(M_)

    for r in range(R):
        idx = lr.permutation(M_)
        d, e = idx[:n_dev], idx[n_dev:]
        n_eval = len(e)
        S = scores[d].mean(0)
        q = softmax(S, tau)
        T = scores[e].mean(0)
        Y = q @ T
        thetas.append(Y)
        if r == 0:
            Y4.append(float(scores[e, np.argmax(S)].mean()))
        Y5.append(float(scores[e, np.argmax(S)].mean()))
        w = 1.0 / R
        er = scores[e] - T
        psi6[e] += w * (M_ / n_eval) * (er @ q)
        psi7[e] += w * (M_ / n_eval) * (er @ q)
        c = q * (T - q @ T) / tau
        psi7[d] += w * (M_ / n_dev) * ((scores[d] - S) @ c)

    theta = np.mean(thetas)
    t4 = Y4[0]
    se4 = np.sqrt(t4*(1-t4)/(M_//2))
    m4 = [t4, [t4-1.96*se4, t4+1.96*se4]]
    t5 = np.mean(Y5)
    se5 = np.std(Y5)/np.sqrt(R)
    m5 = [t5, [t5-1.96*se5, t5+1.96*se5]]

    def boot_ci(psi_vec, seed_offset):
        br = np.random.default_rng(seed + seed_offset)
        b = np.array([theta + (br.standard_normal(M_) * psi_vec).mean() for _ in range(n_boot)])
        return [float(theta), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]]

    m6 = boot_ci(psi6, 6666)
    m7 = boot_ci(psi7, 7777)

    return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6, "m7": m7}


def run_m7_single(scores, rho, tau, R, n_boot, seed):
    """M7 for a single system. Returns (theta, ci_lo, ci_hi)."""
    lr = np.random.default_rng(seed)
    M_, K = scores.shape
    n_dev = int(rho * M_)
    n_eval = M_ - n_dev
    psi = np.zeros(M_)
    thetas = []
    for r in range(R):
        idx = lr.permutation(M_)
        d, e = idx[:n_dev], idx[n_dev:]
        S = scores[d].mean(0)
        q = softmax(S, tau)
        T = scores[e].mean(0)
        Y = q @ T
        thetas.append(Y)
        w = 1.0 / R
        psi[e] += w * (M_ / n_eval) * ((scores[e] - T) @ q)
        c = q * (T - q @ T) / tau
        psi[d] += w * (M_ / n_dev) * ((scores[d] - S) @ c)
    theta = np.mean(thetas)
    boots = np.array([theta + (lr.standard_normal(M_) * psi).mean() for _ in range(n_boot)])
    return float(theta), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_ground_truth(scores, rho, tau, n_mc=100000, seed=42):
    mr = np.random.default_rng(seed)
    M_, K = scores.shape
    nd = int(rho * M_)
    res = np.empty(n_mc)
    for i in range(n_mc):
        idx = mr.permutation(M_)
        S = scores[idx[:nd]].mean(0)
        q = softmax(S, tau)
        T = scores[idx[nd:]].mean(0)
        res[i] = q @ T
    return float(res.mean()), float(res.std())


# ----- System B (baseline prompt) -----
print(f"\n[System B] Building baseline program...")
# System B = default MCQClassifier with no instruction/demo tuning
system_b_program = MCQClassifier()
print(f"[System B] Evaluating on {M} eval items...")
t0 = time.time()
col_b = evaluate_program_on_set(system_b_program, dspy_evalset)
print(f"[System B] accuracy={col_b.mean():.3f} ({time.time()-t0:.0f}s)")
np.save(RESULTS_DIR / "col_b.npy", col_b)


# ----- MAIN LOOP: per budget, run MIPROv2 -----
checkpoints = {}          # record top-K config per budget
all_results = {}

for B, num_trials in zip(BUDGETS, TRIALS):
    bk = str(B)
    print(f"\n{'='*70}\nBUDGET B={B:,}  num_trials={num_trials}\n{'='*70}")

    # --- Phase 1: Run MIPROv2 ---
    print(f"[Phase 1] Initializing MIPROv2 (auto=None, manual num_trials={num_trials})...")
    
    # Record history length before Phase 1 to isolate Phase 1 token usage
    phase1_history_start = len(lm.history)
    
    try:
        teleprompter = MIPROv2(
            metric=mcq_metric,
            auto=None,
            num_candidates=args.num_candidates,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            verbose=False,
            track_stats=True,
            seed=args.seed,
        )

        print(f"[Phase 1] Compiling (bootstrap + propose + {num_trials} trials)...")
        t0 = time.time()
        student = MCQClassifier()
        # Use trainset for both bootstrap + valset (DSPy splits internally if no valset)
        compiled = teleprompter.compile(
            student,
            trainset=dspy_trainset,
            num_trials=num_trials,
            minibatch=True,
            minibatch_size=min(args.minibatch_size, len(dspy_trainset) // 2),
            requires_permission_to_run=False,
        )
        phase1_time = time.time() - t0

        # ---- NEW: Measure actual Phase 1 token consumption ----
        phase1_new_calls = lm.history[phase1_history_start:]
        phase1_prompt_tokens = 0
        phase1_completion_tokens = 0
        phase1_total_tokens = 0
        for entry in phase1_new_calls:
            usage = entry.get("usage", {}) if isinstance(entry, dict) else {}
            if isinstance(usage, dict):
                phase1_prompt_tokens += usage.get("prompt_tokens", 0) or 0
                phase1_completion_tokens += usage.get("completion_tokens", 0) or 0
                phase1_total_tokens += usage.get("total_tokens", 0) or 0
        # Fallback: if usage wasn't populated (some backends don't return it),
        # estimate from LM history count using a conservative 1500 tokens/call.
        if phase1_total_tokens == 0 and len(phase1_new_calls) > 0:
            phase1_total_tokens = len(phase1_new_calls) * 1500
            print(f"[Phase 1] WARNING: usage not reported by LM; estimated "
                  f"{phase1_total_tokens:,} tokens from {len(phase1_new_calls)} calls")

        budget_deviation_pct = (phase1_total_tokens - B) / B * 100 if B > 0 else 0
        print(f"[Phase 1] Complete ({phase1_time:.0f}s)")
        print(f"[Phase 1] Actual Phase 1 tokens: {phase1_total_tokens:,} "
              f"(target B={B:,}, deviation: {budget_deviation_pct:+.1f}%)")
        print(f"[Phase 1] Phase 1 LLM calls: {len(phase1_new_calls)}")

    except Exception as e:
        print(f"[Phase 1] ERROR: MIPROv2 compile failed for B={B}")
        traceback.print_exc()
        all_results[bk] = {"error": f"miprov2 compile failed: {e}"}
        continue

    # --- Extract top-K candidates from MIPROv2 trial logs ---
    # Try multiple attribute names (DSPy API has varied across versions)
    candidate_list = None
    for attr in ("candidate_programs", "full_eval_candidate_programs"):
        if hasattr(compiled, attr):
            cand = getattr(compiled, attr)
            if cand:
                candidate_list = cand
                print(f"[Phase 1] Found {len(cand)} candidates via best.{attr}")
                break

    if candidate_list is None:
        # Fallback: parse trial_logs
        print(f"[Phase 1] No candidate_programs attr; falling back to trial_logs")
        trial_logs = getattr(compiled, "trial_logs", None)
        if trial_logs:
            print(f"[Phase 1]   trial_logs has {len(trial_logs)} entries")
            print(f"[Phase 1]   sample keys: "
                  f"{list(trial_logs[list(trial_logs.keys())[0]].keys()) if isinstance(trial_logs, dict) else list(trial_logs[0].keys())}")
        # If we still can't get candidates, we're stuck
        if not trial_logs:
            print(f"[Phase 1] ERROR: no candidates or trial_logs; skipping budget")
            all_results[bk] = {"error": "no candidates extracted"}
            continue
        # Convert trial_logs to candidates list of dicts with 'score' and 'program'
        candidate_list = []
        logs_iter = trial_logs.values() if isinstance(trial_logs, dict) else trial_logs
        for t in logs_iter:
            # Keys vary; try standard DSPy names
            score = t.get("score", t.get("full_eval_score", t.get("mb_score", None)))
            prog = t.get("program", t.get("candidate_program", None))
            if score is not None and prog is not None:
                candidate_list.append({"score": score, "program": prog})

    # Normalize candidate_list to list of (score, program) tuples
    normalized = []
    for c in candidate_list:
        if isinstance(c, dict):
            score = c.get("score", c.get("full_eval_score", c.get("mb_score", None)))
            prog = c.get("program", c.get("candidate_program", None))
        elif isinstance(c, tuple) and len(c) == 2:
            score, prog = c
        else:
            # assume it's a program directly
            score = getattr(c, "score", None)
            prog = c
        if score is not None and prog is not None:
            normalized.append((float(score), prog))

    if len(normalized) == 0:
        print(f"[Phase 1] ERROR: could not normalize candidates; skipping budget")
        all_results[bk] = {"error": "could not normalize candidates"}
        continue

    # Sort by score descending, take top K
    normalized.sort(key=lambda x: x[0], reverse=True)
    top_k = normalized[:args.K]
    K_actual = len(top_k)
    print(f"[Phase 1] Top-{K_actual} scores: {[round(s, 3) for s, _ in top_k]}")

    # Record checkpoint info (serializable summary)
    checkpoints[bk] = []
    for rank, (score, prog) in enumerate(top_k):
        # Try to extract instruction + num demos for logging
        instr = "?"
        n_demos = 0
        try:
            for pred in prog.predictors():
                sig = pred.signature
                instr = getattr(sig, "instructions", "?")
                n_demos = len(getattr(pred, "demos", []) or [])
                break  # only one predictor in our program
        except Exception:
            pass
        checkpoints[bk].append({
            "rank": rank,
            "id": f"miprov2_B{B}_rank{rank}",
            "score": float(score),
            "instruction_preview": str(instr)[:200] if instr else "",
            "n_demos": int(n_demos),
        })

    # --- Phase 2: Build tensor on evalset ---
    print(f"[Phase 2] Building tensor [{M} x {K_actual}] on eval set...")
    scores_a = np.zeros((M, K_actual))
    for k, (score, prog) in enumerate(top_k):
        t0 = time.time()
        col = evaluate_program_on_set(prog, dspy_evalset)
        scores_a[:, k] = col
        print(f"  {k+1}/{K_actual}: eval_acc={col.mean():.3f}  "
              f"train_score={score:.3f}  ({time.time()-t0:.0f}s)")

    np.save(RESULTS_DIR / f"tensor_a_B{B}.npy", scores_a)

    # --- Phase 2.5: M1-M7 + ground truth ---
    print(f"[Phase 2.5] Running inference methods...")
    ta, la, ha = run_m7_single(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    tb, lb, hb = run_m7_single(col_b.reshape(-1, 1), args.rho, args.tau,
                                args.R, args.n_boot, args.seed + 1)

    comp = run_all_methods(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    comp["system_b"] = float(col_b.mean())

    gt_a, gt_std = compute_ground_truth(scores_a, args.rho, args.tau)
    gt_b = float(col_b.mean())

    print(f"\n  M7: θ̃_A={ta:.4f} [{la:.3f},{ha:.3f}] | θ̃_B={tb:.4f}")
    print(f"  GT: θ*_A={gt_a:.4f} | θ*_B={gt_b:.4f}")
    print(f"  M7 covers θ*_A? {'YES' if la <= gt_a <= ha else 'NO'}")

    for name in ["m1", "m3", "m4", "m5", "m6", "m7"]:
        v = comp[name]
        if isinstance(v, list):
            est, (lo, hi) = v
            tag = "A>B" if lo > gt_b else "   "
            print(f"  {name}: {est:.4f} [{lo:.3f},{hi:.3f}] {tag}")
        else:
            print(f"  {name}: {v:.4f}")

    all_results[bk] = {
        "num_trials": num_trials,
        "phase1_time_sec": phase1_time,
        "target_budget_tokens": B,
        "actual_phase1_tokens": phase1_total_tokens,
        "actual_phase1_prompt_tokens": phase1_prompt_tokens,
        "actual_phase1_completion_tokens": phase1_completion_tokens,
        "budget_deviation_pct": budget_deviation_pct,
        "phase1_llm_calls": len(phase1_new_calls),
        "K_actual": K_actual,
        "top_k_scores": [float(s) for s, _ in top_k],
        "m7": {"system_a": {"theta": ta, "ci_lo": la, "ci_hi": ha},
               "system_b": {"theta": tb, "ci_lo": lb, "ci_hi": hb},
               "delta": ta - tb},
        "comparison": comp,
        "ground_truth": {
            "theta_true_a": gt_a,
            "theta_true_a_std": gt_std,
            "theta_true_b": gt_b,
            "true_delta": gt_a - gt_b,
            "m7_covers_a": bool(la <= gt_a <= ha),
            "m7_covers_b": bool(lb <= gt_b <= hb),
        },
    }


# ----- Save checkpoints (top-K info) and results -----
with open(RESULTS_DIR / "checkpoints.json", "w") as f:
    json.dump(checkpoints, f, indent=2, default=str)


def serialize(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


output = {
    "model": args.model,
    "tuner": "MIPROv2",
    "config": vars(args),
    "results": serialize(all_results),
}

with open(RESULTS_DIR / "study_e_miprov2_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print(f"DONE — {args.model} | MIPROv2")
print(f"Results: {RESULTS_DIR}/")
print(f"{'='*70}")
