"""
Study E — Multi-Model Version (API-enabled, parallelized)
==========================================================
Extends study_e_multimodel.py to support:
  - vLLM local servers (existing)
  - OpenAI (gpt-4o-mini, etc.)
  - Anthropic (claude-haiku-4-5, etc.)
  - Google Gemini via AI Studio API key (gemini-2.5-flash, etc.)
  - Google Gemini via Vertex AI (for Google AI Ultra Cloud credits)
  - Together AI, OpenRouter (optional)

Key changes vs. original:
  1. build_lm() dispatches on --provider
  2. score_artifact runs in a ThreadPoolExecutor (critical for API latency)
  3. /no_think only appended for Qwen3 on vLLM
  4. Per-run cost summary via litellm.completion_cost
  5. Exponential-backoff retry via tenacity

Usage (after setting API keys in .env):

    # SMOKE TEST — run this FIRST to verify pipeline (~$0.50, ~5 min)
    python study_e_multimodel_api.py \
        --provider openai --model gpt-4o-mini \
        --budgets 500000 --n-concurrent 10 \
        --results-dir results/multimodel/smoke_test_gpt4o_mini

    # FULL RUNS
    python study_e_multimodel_api.py \
        --provider openai --model gpt-4o-mini \
        --results-dir results/multimodel/openai_gpt-4o-mini

    python study_e_multimodel_api.py \
        --provider anthropic --model claude-haiku-4-5 \
        --results-dir results/multimodel/anthropic_claude-haiku-4-5

    # Gemini via AI Studio direct API (simple; requires GEMINI_API_KEY)
    python study_e_multimodel_api.py \
        --provider gemini --model gemini-2.5-flash \
        --results-dir results/multimodel/gemini_2.5-flash

    # Gemini via Vertex AI (uses Google AI Ultra / Free Trial Cloud credits)
    # Prereqs:
    #   gcloud auth application-default login
    #   gcloud services enable aiplatform.googleapis.com
    #   export GOOGLE_CLOUD_PROJECT=<your-project-id>
    #   export GOOGLE_CLOUD_LOCATION=us-central1
    #   unset GEMINI_API_KEY GOOGLE_API_KEY
    python study_e_multimodel_api.py \
        --provider vertex_ai --model gemini-2.5-flash \
        --results-dir results/multimodel/vertex_gemini_2.5-flash
"""

import argparse
import json
import re
import time
import os
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

import numpy as np

# --- Load .env if present ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional, keys can come from shell env

# --- Retry decorator ---
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--provider", default="vllm",
                    choices=["vllm", "openai", "anthropic", "gemini",
                             "vertex_ai", "together", "openrouter"],
                    help="Which backend. vllm=local server, rest=cloud APIs. "
                         "vertex_ai uses Google Cloud Vertex AI (for Ultra "
                         "credits); gemini uses AI Studio direct API key.")
parser.add_argument("--model", required=True,
                    help="Model string. For vllm: HF name. For APIs: provider's model ID")
parser.add_argument("--port", type=int, default=8000, help="vLLM port (vllm only)")
parser.add_argument("--results-dir", required=True)
parser.add_argument("--budgets", default="500000,1500000,3000000,6500000")
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--R", type=int, default=10)
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--n-boot", type=int, default=2000)
parser.add_argument("--n-train", type=int, default=100)
parser.add_argument("--n-concurrent", type=int, default=10,
                    help="ThreadPoolExecutor workers for API calls. "
                         "Default 10 is safe for most provider tiers. "
                         "Raise cautiously (20-30) only after verifying no 429s.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--subject", default="math", choices=["math", "law"],
                    help="MMLU-Pro subject filter (math or law)")
args = parser.parse_args()

BUDGETS = [int(b) for b in args.budgets.split(",")]
RESULTS_DIR = Path(args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Resource-limit sanity check
# ============================================================
# High-concurrency API runs open thousands of HTTP sockets at once. Each
# socket = 1 file descriptor. The default Linux ulimit -n is 1024, which
# we will exhaust easily and crash with "Too many open files" partway
# through. Try to raise it; if we can't, warn loudly.
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = max(8192, args.n_concurrent * 64)
    if soft < needed:
        new_soft = min(hard, max(soft, needed))
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"[setup] Raised RLIMIT_NOFILE: {soft} -> {new_soft} "
                  f"(hard={hard})")
        except Exception:
            print(f"[setup] WARNING: file-descriptor soft limit is {soft}, "
                  f"recommended >= {needed}. If you see 'Too many open files' "
                  f"errors, run before launching:\n"
                  f"           ulimit -n 65536\n"
                  f"         (or edit /etc/security/limits.conf for permanent)")
except Exception:
    pass

# ============================================================
# DSPy / LM setup
# ============================================================
import dspy

def build_lm(args):
    """Dispatch DSPy LM constructor based on provider."""
    # max_tokens=256: keeps outputs short but leaves room for reasoning-style
    #   models (e.g. Gemini 2.5 Flash thinking mode) to finish cleanly.
    # timeout=60: upper bound on a single request; prevents one slow call from
    #   hanging the whole ThreadPoolExecutor.
    common = dict(max_tokens=256, temperature=0.0, cache=True,
                  num_retries=2, timeout=60)

    if args.provider == "vllm":
        return dspy.LM(
            f"openai/{args.model}",
            api_base=f"http://localhost:{args.port}/v1",
            api_key="dummy",
            **common,
        )
    if args.provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("ERROR: OPENAI_API_KEY not set")
        return dspy.LM(f"openai/{args.model}", **common)
    if args.provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("ERROR: ANTHROPIC_API_KEY not set")
        return dspy.LM(f"anthropic/{args.model}", **common)
    if args.provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("ERROR: GEMINI_API_KEY not set")
        extra = {}
        if "2.5-flash" in args.model.lower() or "2.5-flash-lite" in args.model.lower():
            extra["reasoning_effort"] = "disable"
        return dspy.LM(f"gemini/{args.model}", **extra, **common)
    if args.provider == "vertex_ai":
        project = (os.environ.get("GOOGLE_CLOUD_PROJECT")
                   or os.environ.get("VERTEXAI_PROJECT"))
        location = (os.environ.get("GOOGLE_CLOUD_LOCATION")
                    or os.environ.get("VERTEXAI_LOCATION")
                    or "us-central1")
        if not project:
            sys.exit("ERROR: GOOGLE_CLOUD_PROJECT not set. "
                     "Run: export GOOGLE_CLOUD_PROJECT=<your-project-id>\n"
                     "Also make sure `gcloud auth application-default login` "
                     "has been run.")
        # Gemini 2.5 Flash ships with thinking mode ON by default. For MCQ
        # where we only want a single letter, reasoning tokens eat budget
        # and cause truncation. Force-disable thinking.
        # (Not supported on Gemini 2.5 Pro or Gemini 3+ models.)
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
    if args.provider == "together":
        if not os.environ.get("TOGETHER_API_KEY"):
            sys.exit("ERROR: TOGETHER_API_KEY not set")
        return dspy.LM(f"together_ai/{args.model}", **common)
    if args.provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("ERROR: OPENROUTER_API_KEY not set")
        return dspy.LM(f"openrouter/{args.model}", **common)
    raise ValueError(f"Unknown provider: {args.provider}")

lm = build_lm(args)
dspy.configure(lm=lm)

print("=" * 60)
print(f"STUDY E — Multi-Model API | provider={args.provider} | model={args.model}")
print(f"Concurrency: {args.n_concurrent}")
print("=" * 60)


# ============================================================
# Data loading (unchanged)
# ============================================================
items = []
with open("data/mmlu_pro_test.json") as f:
    for line in f:
        item = json.loads(line.strip())
        if item.get("category") == args.subject:
            items.append(item)

rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(items))
trainset = [items[i] for i in perm[:args.n_train]]
evalset = [items[i] for i in perm[args.n_train:]]
M = len(evalset)
print(f"[Data] Math items: {len(items)} | Train: {len(trainset)} | Eval: {M}")


# ============================================================
# Helpers (unchanged, plus parallel scoring)
# ============================================================
def extract_answer(text):
    if not text:
        return "?"
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if text and text[0] in "ABCDEFGHIJ":
        rest = text[1:].strip()
        if not rest or rest[0] in ".\n,) ":
            return text[0]
    m = re.search(r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*([A-J])', text)
    if m:
        return m.group(1)
    matches = re.findall(r'\b([A-J])\b', text)
    return matches[-1] if matches else "?"


def format_question(item):
    q = item["question"]
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(item["options"]))
    return f"{q}\n\n{opts}"


def _make_messages(artifact, item):
    """Build the message list for one (artifact, item) pair."""
    instruction = artifact["instruction"]
    system_prompt = artifact.get("system_prompt", "")
    demos = artifact.get("demos", [])

    demo_str = ""
    if demos:
        demo_str = "Here are some examples:\n\n"
        for d in demos:
            demo_str += (f"Example question: {format_question(d)}\n"
                         f"Correct answer: {d['answer']}\n\n")
        demo_str += "Now answer the following question:\n\n"

    q_text = format_question(item)
    user_msg = f"{instruction}\n\n{demo_str}{q_text}"

    # /no_think only for Qwen3 via local vLLM (API models don't understand it)
    if args.provider == "vllm" and "Qwen3" in args.model:
        user_msg += " /no_think"

    messages = []
    # Anthropic/Gemini reject empty system prompts; guard against that
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_msg})
    return messages


# Global error counter — keeps log readable when one error type repeats
# across thousands of futures. Reset between artifacts is not necessary;
# the suppression message tells the user to expect dedup.
_LM_ERROR_COUNTS: dict = {}


def _lm_call_with_retry(messages):
    """One LM call with exponential backoff on failure."""
    if HAS_TENACITY:
        @retry(
            stop=stop_after_attempt(5),
            # max=60s lets us absorb rate-limit windows (most cloud APIs
            # reset quota on a 60s sliding window)
            wait=wait_exponential(min=2, max=60),
            reraise=True,  # surface the real exception, not RetryError wrapper
        )
        def _inner():
            return lm(messages=messages)
        try:
            return _inner()
        except Exception as e:
            # Print only the first few unique errors per artifact to avoid log spam
            err_key = f"{type(e).__name__}:{str(e)[:80]}"
            _LM_ERROR_COUNTS[err_key] = _LM_ERROR_COUNTS.get(err_key, 0) + 1
            if _LM_ERROR_COUNTS[err_key] <= 3:
                print(f"  [call failed] {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
            elif _LM_ERROR_COUNTS[err_key] == 4:
                print(f"  [call failed] (suppressing further '{type(e).__name__}' "
                      f"messages — see summary at end)", flush=True)
            return None
    else:
        try:
            return lm(messages=messages)
        except Exception as e:
            print(f"  [call failed] {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            return None


def score_artifact(artifact, item_list, return_binary=False,
                    n_concurrent=None):
    """
    Score artifact on item_list. Parallelized via ThreadPoolExecutor for APIs.
    For local vLLM, still parallel — vLLM batches on its side, so ~20 concurrent
    requests are fine.
    """
    n_workers = n_concurrent or args.n_concurrent
    binary = [0] * len(item_list)

    def score_one(idx_item):
        idx, item = idx_item
        messages = _make_messages(artifact, item)
        resp = _lm_call_with_retry(messages)
        if resp is None:
            text = ""
        else:
            text = resp[0] if isinstance(resp, list) else str(resp)
        pred = extract_answer(text)
        return idx, 1 if pred == item["answer"] else 0

    # Per-future hard timeout (seconds). Defensive: even with `timeout=60` on
    # the LM + 3 retries, something weird upstream (e.g. hung socket) could
    # leave a future pending forever. 5 min is the ceiling; anything past that
    # we count as a failed item (score=0) and move on.
    PER_FUTURE_TIMEOUT_SEC = 300

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(score_one, (i, it)): i
                   for i, it in enumerate(item_list)}
        try:
            for f in as_completed(futures, timeout=PER_FUTURE_TIMEOUT_SEC):
                idx, score = f.result()
                binary[idx] = score
        except FuturesTimeout:
            # Some futures never returned. Count them as 0s and continue.
            unfinished = [i for fut, i in futures.items() if not fut.done()]
            print(f"  [timeout] {len(unfinished)} items did not return within "
                  f"{PER_FUTURE_TIMEOUT_SEC}s; counting as failures", flush=True)
            for fut, idx in futures.items():
                if fut.done():
                    try:
                        i2, s = fut.result(timeout=0)
                        binary[i2] = s
                    except Exception:
                        binary[idx] = 0
                else:
                    fut.cancel()
                    binary[idx] = 0

    binary = np.array(binary, dtype=float)
    acc = float(binary.mean())
    # token counting delegated to lm.history / litellm.completion_cost at end
    if return_binary:
        return acc, binary, 0
    return acc, 0


def select_demos(n, strategy, trainset, rng):
    if n == 0:
        return []
    idx = rng.choice(len(trainset), size=min(n, len(trainset)), replace=False)
    return [trainset[i] for i in idx]


def softmax(x, tau):
    z = x / tau - (x / tau).max()
    e = np.exp(z)
    return e / e.sum()


def run_m7(scores, rho, tau, R, n_boot, seed):
    lr = np.random.default_rng(seed)
    M, K = scores.shape
    n_dev = int(rho * M)
    n_eval = M - n_dev
    psi = np.zeros(M)
    thetas = []
    for r in range(R):
        idx = lr.permutation(M)
        d, e = idx[:n_dev], idx[n_dev:]
        S = scores[d].mean(0)
        q = softmax(S, tau)
        T = scores[e].mean(0)
        Y = q @ T
        thetas.append(Y)
        w = 1.0 / R
        psi[e] += w * (M / n_eval) * ((scores[e] - T) @ q)
        c = q * (T - q @ T) / tau
        psi[d] += w * (M / n_dev) * ((scores[d] - S) @ c)
    theta = np.mean(thetas)
    boots = np.array([theta + (lr.standard_normal(M) * psi).mean()
                      for _ in range(n_boot)])
    return theta, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def run_all_methods(scores, rho, tau, R, n_boot, seed):
    lr = np.random.default_rng(seed)
    M, K = scores.shape
    n_dev = int(rho * M)
    m1 = float(scores[:, 0].mean())
    m2 = m1
    se = np.sqrt(m1 * (1 - m1) / M)
    m3 = [m1, [m1 - 1.96*se, m1 + 1.96*se]]

    Y4, Y5, thetas = [], [], []
    psi6, psi7 = np.zeros(M), np.zeros(M)

    for r in range(R):
        idx = lr.permutation(M)
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
        psi6[e] += w * (M / n_eval) * (er @ q)
        psi7[e] += w * (M / n_eval) * (er @ q)
        c = q * (T - q @ T) / tau
        psi7[d] += w * (M / n_dev) * ((scores[d] - S) @ c)

    theta = np.mean(thetas)

    t4 = Y4[0]
    se4 = np.sqrt(t4*(1-t4)/(M//2))
    m4 = [t4, [t4-1.96*se4, t4+1.96*se4]]

    t5 = np.mean(Y5)
    se5 = np.std(Y5)/np.sqrt(R)
    m5 = [t5, [t5-1.96*se5, t5+1.96*se5]]

    def boot_ci(psi_vec, seed_offset):
        br = np.random.default_rng(seed + seed_offset)
        b = np.array([theta + (br.standard_normal(M) * psi_vec).mean()
                      for _ in range(n_boot)])
        return [float(theta),
                [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]]

    m6 = boot_ci(psi6, 6666)
    m7 = boot_ci(psi7, 7777)

    return {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6, "m7": m7}


def compute_ground_truth(scores, rho, tau, n_mc=100000, seed=42):
    mr = np.random.default_rng(seed)
    M, K = scores.shape
    nd = int(rho * M)
    res = np.empty(n_mc)
    for i in range(n_mc):
        idx = mr.permutation(M)
        S = scores[idx[:nd]].mean(0)
        q = softmax(S, tau)
        T = scores[idx[nd:]].mean(0)
        res[i] = q @ T
    return float(res.mean()), float(res.std())


# ============================================================
# Cost tracking
# ============================================================
def report_cost(stage_label):
    """Print running cost. Call this between major stages."""
    try:
        import litellm
        total_cost = 0.0
        n = 0
        for h in lm.history:
            try:
                # Modern DSPy stores the full litellm response under 'response'
                resp_obj = h.get("response") or h
                c = litellm.completion_cost(completion_response=resp_obj)
                total_cost += c
                n += 1
            except Exception:
                continue
        print(f"  [cost so far — {stage_label}] ${total_cost:.4f} "
              f"across {n}/{len(lm.history)} trackable calls", flush=True)
        return total_cost
    except Exception as e:
        print(f"  [cost tracking unavailable: {e}]")
        return None


# ============================================================
# PHASE 1: Search (parallelized when possible)
# ============================================================
print(f"\n{'='*60}\nPHASE 1 — DSPy Search\n{'='*60}")

SYSTEM_PROMPTS = [
    "You are a helpful assistant that answers math questions concisely.",
    "You are an expert mathematics professor with decades of experience.",
    "You specialize in solving university-level mathematics problems.",
    "You are a precise math tutor who always gives the correct answer.",
    "You are a careful student taking an important mathematics examination.",
    "You are a mathematics olympiad coach who thinks carefully before answering.",
    "",
]
BASE_INSTRUCTIONS = [
    "Answer the following multiple-choice math question. Output ONLY the letter of the correct answer.",
    "Choose the correct option and reply with the letter only.",
    "Select the correct answer and reply with the letter only.",
    "Answer the math question by selecting the correct option. Provide only the letter of the correct answer.",
    "Carefully evaluate the math problem and output only the correct letter.",
    "Solve this math problem and choose the correct option. Reply with one letter only.",
    "Select the correct answer from the options. Output only the letter corresponding to the correct choice.",
    "Choose the correct answer and output only the letter. No explanation required.",
    "You are a math expert. Provide only the correct answer letter.",
    "Solve the problem and choose the correct option. Output only the letter.",
]

# Generate more instructions via LLM (one call, blocking)
try:
    gen_resp = lm(messages=[{"role": "user", "content":
        "Generate 40 different instruction variants for a math MCQ task. "
        "Each should tell the model to output only the answer letter (A-J). "
        "One per line, numbered 1-40."}])
    gen_text = gen_resp[0] if isinstance(gen_resp, list) else str(gen_resp)
    generated = [re.sub(r'^\d+[\.\)]\s*', '', l.strip())
                 for l in gen_text.strip().split("\n")]
    generated = [g for g in generated if 20 < len(g) < 200]
    all_instructions = list(set(BASE_INSTRUCTIONS + generated))
    print(f"  Generated {len(generated)} instructions, total: {len(all_instructions)}")
except Exception as e:
    all_instructions = BASE_INSTRUCTIONS
    print(f"  Using {len(all_instructions)} base instructions (gen failed: {e})")

# Build candidate pool
candidates = []
for instr in all_instructions:
    for sp in SYSTEM_PROMPTS:
        for nd in [0, 1, 2, 3]:
            for strat in ["random", "diverse", "hard", "easy"]:
                candidates.append({"instruction": instr, "system_prompt": sp,
                                   "n_demos": nd, "demo_strategy": strat})
rng.shuffle(candidates)
print(f"[Search] {len(candidates)} candidates, budgets: {BUDGETS}")

# Run search (approximate token budget — we can't know exact tokens without
# tokenizer access for all providers, so we approximate per-call)
TOKENS_PER_CALL_APPROX = 800  # input + output for an MCQ call

evaluated = []
cum_tok = 0
max_b = max(BUDGETS)
cp_pool = {str(b): [] for b in BUDGETS}

for i, c in enumerate(candidates):
    if cum_tok >= max_b:
        break
    c["demos"] = select_demos(c["n_demos"], c["demo_strategy"], trainset, rng)
    c["id"] = f"candidate_{i:04d}"
    acc, _ = score_artifact(c, trainset)
    # Approximate tokens per search step: n_train items * per-call estimate
    step_tokens = len(trainset) * TOKENS_PER_CALL_APPROX
    cum_tok += step_tokens
    c["score"] = acc
    evaluated.append(c)
    for b in BUDGETS:
        if cum_tok <= b:
            cp_pool[str(b)].append(c)
    if (i + 1) % 10 == 0:
        print(f"  #{i+1} | {cum_tok:,}/{max_b:,} | score={acc:.3f}")

print(f"  Evaluated: {len(evaluated)} | Tokens (approx): {cum_tok:,}")
report_cost("after search phase")

# Top-K per budget
checkpoints = {}
for b in BUDGETS:
    bk = str(b)
    pool = sorted(cp_pool[bk], key=lambda x: x["score"], reverse=True)[:args.K]
    checkpoints[bk] = [{"id": c["id"], "score": c["score"],
                         "instruction": c["instruction"],
                         "system_prompt": c["system_prompt"],
                         "n_demos": c["n_demos"],
                         "demo_strategy": c["demo_strategy"]} for c in pool]
    print(f"  B={b:>9,}: top scores = {[c['score'] for c in pool[:5]]}")

with open(RESULTS_DIR / "checkpoints.json", "w") as f:
    json.dump(checkpoints, f, indent=2)

# ============================================================
# System B
# ============================================================
print(f"\n[System B] Scoring fixed prompt...")
sys_b = {"instruction": "Answer the following multiple-choice math question. "
                         "Output ONLY the letter of the correct answer.",
         "system_prompt": "You are a helpful assistant that answers math "
                           "questions concisely.", "demos": []}
_, col_b, _ = score_artifact(sys_b, evalset, return_binary=True)
np.save(RESULTS_DIR / "col_b.npy", col_b)
print(f"  System B accuracy: {col_b.mean():.3f}")
report_cost("after System B")

# ============================================================
# Per-budget: tensor + inference + ground truth
# ============================================================
all_results = {}

for B in BUDGETS:
    bk = str(B)
    print(f"\n{'='*60}\nBUDGET B = {B:,}\n{'='*60}")
    top_k = checkpoints[bk]
    K = len(top_k)
    if K == 0:
        print("  [SKIP]")
        continue

    scores_a = np.zeros((M, K))
    for ki, art in enumerate(top_k):
        a = {"instruction": art["instruction"],
             "system_prompt": art["system_prompt"],
             "demos": select_demos(art["n_demos"], art["demo_strategy"],
                                    trainset, rng)}
        t0 = time.time()
        acc, binary, _ = score_artifact(a, evalset, return_binary=True)
        scores_a[:, ki] = binary
        print(f"  {ki+1}/{K}: {art['id']} acc={acc:.3f} ({time.time()-t0:.0f}s)")

    np.save(RESULTS_DIR / f"tensor_a_B{B}.npy", scores_a)
    report_cost(f"after tensor B={B}")

    # M7 for System A and B
    ta, la, ha = run_m7(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    tb, lb, hb = run_m7(col_b.reshape(-1, 1), args.rho, args.tau, args.R,
                         args.n_boot, args.seed + 1)

    comp = run_all_methods(scores_a, args.rho, args.tau, args.R,
                            args.n_boot, args.seed)
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
            aw = "A>B" if lo > gt_b else "   "
            print(f"  {name}: {est:.4f} [{lo:.3f},{hi:.3f}] {aw}")
        else:
            print(f"  {name}: {v:.4f}")

    all_results[bk] = {
        "m7": {"system_a": {"theta": ta, "ci_lo": la, "ci_hi": ha},
               "system_b": {"theta": tb, "ci_lo": lb, "ci_hi": hb},
               "delta": ta - tb},
        "comparison": comp,
        "top_k_scores": [c["score"] for c in top_k],
        "ground_truth": {"theta_true_a": gt_a, "theta_true_b": gt_b,
                         "true_delta": gt_a - gt_b,
                         "m7_covers_a": bool(la <= gt_a <= ha),
                         "m7_covers_b": bool(lb <= gt_b <= hb)},
    }

# --- Save ---
def serialize(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [serialize(v) for v in obj]
    return obj

final_cost = report_cost("FINAL")

output = {
    "model": args.model,
    "provider": args.provider,
    "config": vars(args),
    "total_cost_usd": final_cost,
    "results": serialize(all_results),
}

with open(RESULTS_DIR / "study_e_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE — {args.provider}/{args.model}")
print(f"Total cost: ${final_cost:.2f}" if final_cost else "Cost: unavailable")
print(f"Results: {RESULTS_DIR}/")
if _LM_ERROR_COUNTS:
    print(f"\n  LM call error summary (returned None, scored as 0):")
    total_failed = sum(_LM_ERROR_COUNTS.values())
    for err, count in sorted(_LM_ERROR_COUNTS.items(),
                              key=lambda x: -x[1]):
        print(f"    [{count:>5d}] {err}")
    print(f"  Total failed calls: {total_failed}")
print(f"{'='*60}")
