"""
Study E — DSPy Random Search with Token Budget (Option 3a, Law)
================================================================
Uses DSPy components (BootstrapFewShot + LabeledFewShot) to generate
a candidate pool, then runs budget-controlled random search over the pool.

Pipeline:
    Phase 0: DSPy setup + data loading
    Phase 1: Pool generation (one-time, NOT counted in budget)
             50 candidates via BootstrapFewShot w/ diverse seeds
    Phase 2: System B baseline evaluation on evalset (1001 items)
    Phase 3: Per-budget loop:
        3a. Random-shuffle pool
        3b. Evaluate each candidate on full trainset (100 items)
        3c. Stop when cumulative tokens exceed budget
        3d. Top-K by trainset score
        3e. Full evaluation of top-K on evalset → tensor [M × K]
        3f. Run M1–M7 inference
    Phase 4: Save results JSON

Usage:
    python study_e_dspy_randsearch_law.py \
        --model Qwen/Qwen3-8B \
        --port 8000 \
        --subject law \
        --results-dir results/dspy_randsearch_multimodel/Qwen_Qwen3-8B_law

Requires: vLLM serving on --port (with --reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking": false}' for Qwen3)
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
parser.add_argument("--provider", default="vllm",
                    choices=["vllm", "openai", "anthropic", "gemini", "vertex_ai"],
                    help="LM backend. vllm=local server, others=cloud APIs.")
parser.add_argument("--results-dir", required=True)
parser.add_argument("--subject", default="law")
parser.add_argument("--budgets", default="500000,1500000,3000000,6500000")
parser.add_argument("--pool-size", type=int, default=50)
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--R", type=int, default=10)
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--n-boot", type=int, default=2000)
parser.add_argument("--n-train", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-demo-range", type=int, default=4,
                    help="sample max_bootstrapped_demos and max_labeled_demos from [0, this)")
args = parser.parse_args()

BUDGETS = [int(b) for b in args.budgets.split(",")]
RESULTS_DIR = Path(args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"STUDY E — DSPy RANDOM SEARCH | {args.model} | subject={args.subject}")
print(f"Budgets: {BUDGETS}")
print(f"Pool size: {args.pool_size}, K={args.K}, R={args.R}")
print("=" * 70)


# ----- DSPy import -----
import dspy
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot


# ----- Configure DSPy LM -----
def build_lm():
    """Dispatch dspy.LM constructor based on --provider."""
    common = dict(max_tokens=1024, temperature=0.0, cache=False)
    if args.provider == "vllm":
        return dspy.LM(
            f"openai/{args.model}",
            api_base=f"http://localhost:{args.port}/v1",
            api_key="dummy",
            **common,
        )
    elif args.provider == "openai":
        return dspy.LM(f"openai/{args.model}", **common)
    elif args.provider == "anthropic":
        return dspy.LM(f"anthropic/{args.model}", **common)
    elif args.provider == "gemini":
        return dspy.LM(f"gemini/{args.model}", **common)
    elif args.provider == "vertex_ai":
        import os
        project  = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "vertex_ai requires env var GOOGLE_CLOUD_PROJECT. "
                "Run: export GOOGLE_CLOUD_PROJECT=<your-project-id>"
            )
        # Disable Gemini 2.5 Flash thinking mode (mirrors API script).
        extra = {}
        if "2.5-flash" in args.model.lower() or "2.5-flash-lite" in args.model.lower():
            extra["reasoning_effort"] = "disable"
        return dspy.LM(
            f"vertex_ai/{args.model}",
            vertex_project=project,
            vertex_location=location,
            **extra,
            timeout=120,  # 2 min per API call (kills hung sockets)
            num_retries=3,  # auto-retry on timeout/5xx
            **common,
        )
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

lm = build_lm()
dspy.configure(lm=lm, track_usage=True)
print(f"[Setup] DSPy configured with {args.model} @ port {args.port}")


# ----- Data loading -----
items = []
with open("data/mmlu_pro_test.json") as f:
    for line in f:
        item = json.loads(line.strip())
        if item.get("category") == args.subject:
            items.append(item)

if len(items) == 0:
    print(f"ERROR: no items for subject={args.subject!r}")
    sys.exit(1)

rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(items))
trainset_raw = [items[i] for i in perm[:args.n_train]]
evalset_raw = [items[i] for i in perm[args.n_train:]]
M = len(evalset_raw)
print(f"[Data] {len(items)} items | Train: {len(trainset_raw)} | Eval: {M}")


# ----- DSPy program -----
class MCQAnswer(dspy.Signature):
    """Answer a multiple-choice question by outputting the correct letter."""
    question = dspy.InputField(desc="The question to answer")
    options = dspy.InputField(desc="Options A through J")
    answer = dspy.OutputField(desc="The single letter of the correct answer")


class MCQClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(MCQAnswer)

    def forward(self, question, options):
        return self.classify(question=question, options=options)


def format_options(options):
    return "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))


def to_dspy_example(item):
    return dspy.Example(
        question=item["question"],
        options=format_options(item["options"]),
        answer=item["answer"],
    ).with_inputs("question", "options")


dspy_trainset = [to_dspy_example(it) for it in trainset_raw]
dspy_evalset = [to_dspy_example(it) for it in evalset_raw]


# ----- Metric -----
def extract_letter(s):
    if not s: return "?"
    s = str(s).strip()
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()
    if s and s[0] in "ABCDEFGHIJ":
        rest = s[1:].strip()
        if not rest or rest[0] in ".\n,) ":
            return s[0]
    m = re.search(r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([A-J])', s)
    if m: return m.group(1)
    matches = re.findall(r'\b([A-J])\b', s)
    return matches[-1] if matches else "?"


def mcq_metric(example, pred, trace=None):
    try:
        return int(extract_letter(pred.answer) == example.answer)
    except Exception:
        return 0


# ----- Token counting (robust against lm.history 10K cap) -----
class TokenTracker:
    """Accumulator for token usage. Reads from lm.history since last checkpoint.
    Maintains running total outside of lm.history to survive its 10K cap."""
    def __init__(self):
        self.total = 0
        self.prompt = 0
        self.completion = 0
        self.n_calls = 0
        self._last_history_len = 0

    def update(self):
        """Call after LLM activity; accumulates tokens from new history entries."""
        current_len = len(lm.history)
        # If history was capped (evicted), current_len might be < _last_history_len
        # In that case, iterate over all history (conservative)
        if current_len < self._last_history_len:
            new_entries = lm.history  # everything still visible
        else:
            new_entries = lm.history[self._last_history_len:current_len]
        for entry in new_entries:
            if isinstance(entry, dict):
                usage = entry.get("usage", {})
                if isinstance(usage, dict):
                    self.total += usage.get("total_tokens", 0) or 0
                    self.prompt += usage.get("prompt_tokens", 0) or 0
                    self.completion += usage.get("completion_tokens", 0) or 0
            self.n_calls += 1
        self._last_history_len = current_len

    def snapshot(self):
        return {"total": self.total, "prompt": self.prompt,
                "completion": self.completion, "n_calls": self.n_calls}


# ----- Evaluate a program on a dataset -----
def evaluate_program_on_set(program, dataset):
    """Return binary [0,1] array of correctness on dataset."""
    binary = []
    for ex in dataset:
        try:
            pred = program(question=ex.question, options=ex.options)
            ok = int(extract_letter(pred.answer) == ex.answer)
        except Exception:
            ok = 0
        binary.append(ok)
    return np.array(binary, dtype=float)


def evaluate_program_with_tokens(program, dataset, tracker):
    """Evaluate + measure tokens consumed in this evaluation. Return (accuracy, tokens_delta)."""
    tokens_before = tracker.total
    tracker.update()  # ensure we start from clean state
    tokens_before = tracker.total

    binary = evaluate_program_on_set(program, dataset)
    tracker.update()

    tokens_after = tracker.total
    return float(binary.mean()), binary, tokens_after - tokens_before


# ============================================================
# M1–M7 inference methods (same as random search / MIPROv2 versions)
# ============================================================
def softmax(x, tau):
    z = x / tau - (x / tau).max()
    e = np.exp(z)
    return e / e.sum()


def run_all_methods(scores, rho, tau, R, n_boot, seed):
    lr = np.random.default_rng(seed)
    M_, K = scores.shape
    n_dev = int(rho * M_)

    col_means = scores.mean(axis=0)
    m1 = float(col_means.max())
    m2 = m1
    se = np.sqrt(m1 * (1 - m1) / M_)
    m3 = [m1, [m1 - 1.96 * se, m1 + 1.96 * se]]

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
    se4 = np.sqrt(t4 * (1 - t4) / (M_ // 2))
    m4 = [t4, [t4 - 1.96 * se4, t4 + 1.96 * se4]]
    t5 = np.mean(Y5)
    se5 = np.std(Y5) / np.sqrt(R)
    m5 = [t5, [t5 - 1.96 * se5, t5 + 1.96 * se5]]

    def boot_ci(psi_vec, seed_offset):
        br = np.random.default_rng(seed + seed_offset)
        b = np.array([theta + (br.standard_normal(M_) * psi_vec).mean() for _ in range(n_boot)])
        return [float(theta), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]]

    m6 = boot_ci(psi6, 6666)
    m7 = boot_ci(psi7, 7777)

    return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6, "m7": m7}


def run_m7_single(scores, rho, tau, R, n_boot, seed):
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


# ============================================================
# PHASE 1: Pool Generation (one-time, NOT in budget)
# ============================================================
print(f"\n{'='*70}\nPHASE 1: Pool Generation ({args.pool_size} candidates)\n{'='*70}")

tracker = TokenTracker()
tracker.update()  # initialize _last_history_len
pool_gen_start_tokens = tracker.total

pool = []
pool_meta = []
phase1_start_time = time.time()

# Slot 0: Baseline
pool.append(MCQClassifier())
pool_meta.append({"idx": 0, "type": "baseline", "n_demos": 0, "config": None})
print(f"[{0}/{args.pool_size}] baseline (no demos)")

# Slot 1: LabeledFewShot (k=2)
try:
    labeled = LabeledFewShot(k=2)
    cand = labeled.compile(MCQClassifier(), trainset=dspy_trainset)
    pool.append(cand)
    n_demos = len(list(cand.predictors())[0].demos or []) if list(cand.predictors()) else 0
    pool_meta.append({"idx": 1, "type": "labeled", "n_demos": n_demos, "config": "k=2"})
    print(f"[{1}/{args.pool_size}] labeled k=2 → {n_demos} demos")
except Exception as e:
    print(f"[{1}/{args.pool_size}] labeled FAILED: {e}")

# Slots 2+: BootstrapFewShot with diverse seeds & configs
for seed in range(2, args.pool_size):
    rng_local = np.random.default_rng(seed)
    shuffled = [dspy_trainset[i] for i in rng_local.permutation(len(dspy_trainset))]
    n_boot = int(rng_local.integers(0, args.max_demo_range))
    n_lab = int(rng_local.integers(0, args.max_demo_range))

    # Skip (0, 0) config — equivalent to baseline
    if n_boot == 0 and n_lab == 0:
        # Re-roll once
        n_boot = max(1, int(rng_local.integers(1, args.max_demo_range)))

    try:
        bs = BootstrapFewShot(
            metric=mcq_metric,
            max_bootstrapped_demos=n_boot,
            max_labeled_demos=n_lab,
            max_rounds=1,
        )
        cand = bs.compile(MCQClassifier(), trainset=shuffled)
        n_demos = len(list(cand.predictors())[0].demos or []) if list(cand.predictors()) else 0
        pool.append(cand)
        pool_meta.append({
            "idx": len(pool) - 1, "type": "bootstrap",
            "seed": seed, "n_boot": n_boot, "n_lab": n_lab,
            "n_demos": n_demos,
        })
        if seed <= 5 or seed % 10 == 0:
            print(f"[{len(pool)-1}/{args.pool_size}] bootstrap seed={seed} (boot={n_boot}, lab={n_lab}) → {n_demos} demos")
    except Exception as e:
        print(f"[seed={seed}] bootstrap FAILED: {type(e).__name__}: {str(e)[:100]}")

tracker.update()
pool_gen_time = time.time() - phase1_start_time
pool_gen_tokens = tracker.total - pool_gen_start_tokens

print(f"\n[Phase 1] Pool generated: {len(pool)} candidates")
print(f"[Phase 1] Time: {pool_gen_time:.1f}s ({pool_gen_time/60:.1f} min)")
print(f"[Phase 1] Tokens: {pool_gen_tokens:,}")

# Save pool metadata
with open(RESULTS_DIR / "pool_metadata.json", "w") as f:
    json.dump({
        "pool_size": len(pool),
        "pool_gen_tokens": pool_gen_tokens,
        "pool_gen_time_sec": pool_gen_time,
        "candidates": pool_meta,
    }, f, indent=2, default=str)


# ============================================================
# PHASE 2: System B (baseline) evaluation on evalset
# ============================================================
print(f"\n{'='*70}\nPHASE 2: System B Evaluation\n{'='*70}")

sysb_start_tokens = tracker.total
t0 = time.time()
system_b_program = MCQClassifier()
col_b = evaluate_program_on_set(system_b_program, dspy_evalset)
tracker.update()
sysb_tokens = tracker.total - sysb_start_tokens
print(f"[System B] accuracy={col_b.mean():.3f} ({time.time()-t0:.0f}s, "
      f"{sysb_tokens:,} tokens)")
np.save(RESULTS_DIR / "col_b.npy", col_b)


# ============================================================
# PHASE 3: Per-budget random search + tensor + inference
# ============================================================
all_results = {}

for B in BUDGETS:
    bk = str(B)
    print(f"\n{'='*70}\nBUDGET B={B:,}\n{'='*70}")

    # ---- Phase 3a–c: Budget-controlled random search ----
    search_start_tokens = tracker.total
    search_start_time = time.time()

    rng_search = np.random.default_rng(args.seed + B)  # per-budget shuffle
    search_order = rng_search.permutation(len(pool))

    trainset_scores = {}  # pool_idx → trainset accuracy
    candidates_evaluated = 0

    for rank, pool_idx in enumerate(search_order):
        # Check budget before evaluating
        tokens_used = tracker.total - search_start_tokens
        if tokens_used >= B:
            print(f"  Budget {B:,} exhausted after {candidates_evaluated} candidates")
            break

        candidate = pool[pool_idx]
        pre_eval_tokens = tracker.total
        try:
            score = evaluate_program_on_set(candidate, dspy_trainset).mean()
        except Exception as e:
            print(f"  [rank={rank}] candidate_{pool_idx} eval FAILED: {e}")
            continue
        tracker.update()
        cand_tokens = tracker.total - pre_eval_tokens
        trainset_scores[int(pool_idx)] = float(score)
        candidates_evaluated += 1

        if rank < 3 or rank % 10 == 0:
            cum = tracker.total - search_start_tokens
            print(f"  [{rank+1}] candidate_{pool_idx}: train_acc={score:.3f} "
                  f"({cand_tokens:,} tokens, cum={cum:,}/{B:,})")

    search_time = time.time() - search_start_time
    search_tokens = tracker.total - search_start_tokens
    print(f"[Phase 3-search] {candidates_evaluated} candidates evaluated, "
          f"{search_tokens:,} tokens ({search_tokens/B*100:.1f}% of budget), "
          f"time={search_time:.0f}s")

    if candidates_evaluated == 0:
        print(f"  ⚠️  NO candidates evaluated within budget! Saving stub.")
        all_results[bk] = {"error": "no candidates within budget",
                           "target_budget": B,
                           "search_tokens": search_tokens}
        continue

    # ---- Phase 3d: Top-K selection ----
    sorted_by_score = sorted(trainset_scores.items(), key=lambda x: -x[1])
    top_k_indices = [idx for idx, _ in sorted_by_score[:args.K]]
    K_actual = len(top_k_indices)
    top_k_scores = [trainset_scores[i] for i in top_k_indices]
    print(f"[Phase 3-topk] K_actual={K_actual}, train_scores={[round(s, 3) for s in top_k_scores]}")

    # ---- Phase 3e: Full evaluation on evalset → tensor ----
    print(f"[Phase 3-tensor] Building tensor [{M} × {K_actual}]...")
    phase2_start = time.time()
    phase2_start_tokens = tracker.total
    scores_a = np.zeros((M, K_actual))
    for k, pool_idx in enumerate(top_k_indices):
        t0 = time.time()
        col = evaluate_program_on_set(pool[pool_idx], dspy_evalset)
        scores_a[:, k] = col
        print(f"  {k+1}/{K_actual}: candidate_{pool_idx} eval_acc={col.mean():.3f} "
              f"({time.time()-t0:.0f}s)")
    tracker.update()
    phase2_tokens = tracker.total - phase2_start_tokens
    print(f"[Phase 3-tensor] Done in {time.time()-phase2_start:.0f}s, {phase2_tokens:,} tokens")

    np.save(RESULTS_DIR / f"tensor_a_B{B}.npy", scores_a)

    # ---- Phase 3f: M1–M7 inference ----
    print(f"[Phase 3-inference] Running M1–M7...")
    ta, la, ha = run_m7_single(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    tb, lb, hb = run_m7_single(col_b.reshape(-1, 1), args.rho, args.tau,
                                args.R, args.n_boot, args.seed + 1)
    comp = run_all_methods(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    comp["system_b"] = float(col_b.mean())
    gt_a, gt_std = compute_ground_truth(scores_a, args.rho, args.tau)
    gt_b = float(col_b.mean())

    print(f"  M7: θ̃_A={ta:.4f} [{la:.3f},{ha:.3f}] | θ̃_B={tb:.4f}")
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
        "target_budget": B,
        "search_tokens": int(search_tokens),
        "phase2_tokens": int(phase2_tokens),
        "search_time_sec": float(search_time),
        "candidates_evaluated": candidates_evaluated,
        "K_actual": K_actual,
        "top_k_pool_indices": [int(i) for i in top_k_indices],
        "top_k_train_scores": top_k_scores,
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


# ============================================================
# PHASE 4: Save
# ============================================================
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


final_output = {
    "model": args.model,
    "tuner": "DSPy-RandomSearch-TokenBudget",
    "config": vars(args),
    "pool_gen": {
        "pool_size": len(pool),
        "tokens": int(pool_gen_tokens),
        "time_sec": float(pool_gen_time),
    },
    "system_b": {
        "accuracy": float(col_b.mean()),
        "tokens": int(sysb_tokens),
    },
    "results": serialize(all_results),
}

with open(RESULTS_DIR / "study_e_dspy_randsearch_results.json", "w") as f:
    json.dump(final_output, f, indent=2)

print(f"\n{'='*70}\nDONE — {args.model}\nResults: {RESULTS_DIR}/\n{'='*70}")
