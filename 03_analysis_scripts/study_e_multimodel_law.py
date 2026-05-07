"""
Study E — Multi-Model Version
================================
Parameterized version of study_e_v2.py for any vLLM-served model.

Usage:
    python study_e_multimodel.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8000 \
        --results-dir results/multimodel/meta-llama_Llama-3.1-8B-Instruct

Requires: vLLM server already running on --port.
"""

import argparse, json, re, time, os, sys
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--results-dir", required=True)
parser.add_argument("--subject", default="math",
                    help="MMLU-Pro subject (e.g. math, law, physics, chemistry)")
parser.add_argument("--budgets", default="500000,1500000,3000000,6500000")
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--R", type=int, default=10)
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--n-boot", type=int, default=2000)
parser.add_argument("--n-train", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

BUDGETS = [int(b) for b in args.budgets.split(",")]
RESULTS_DIR = Path(args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- DSPy ---
import dspy
lm = dspy.LM(
    f"openai/{args.model}",
    api_base=f"http://localhost:{args.port}/v1",
    api_key="dummy",
    max_tokens=64,
    temperature=0.0,
)
dspy.configure(lm=lm)

print("=" * 60)
print(f"STUDY E — Multi-Model | {args.model}")
print("=" * 60)

# --- Load data ---
items = []
with open("data/mmlu_pro_test.json") as f:
    for line in f:
        item = json.loads(line.strip())
        if item.get("category") == args.subject:
            items.append(item)

if len(items) == 0:
    print(f"ERROR: no items found for subject={args.subject!r}")
    print("Check data/mmlu_pro_test.json — valid subjects are the MMLU-Pro category names.")
    sys.exit(1)

rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(items))
trainset = [items[i] for i in perm[:args.n_train]]
evalset = [items[i] for i in perm[args.n_train:]]
M = len(evalset)
print(f"[Data] Subject={args.subject} | Total items: {len(items)} | Train: {len(trainset)} | Eval: {M}")


# --- Helpers ---
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


def score_artifact(artifact, item_list, return_binary=False):
    instruction = artifact["instruction"]
    system_prompt = artifact.get("system_prompt", "")
    demos = artifact.get("demos", [])

    demo_str = ""
    if demos:
        demo_str = "Here are some examples:\n\n"
        for d in demos:
            demo_str += f"Example question: {format_question(d)}\nCorrect answer: {d['answer']}\n\n"
        demo_str += "Now answer the following question:\n\n"

    binary = []
    total_tokens = 0
    for item in item_list:
        q_text = format_question(item)
        user_msg = f"{instruction}\n\n{demo_str}{q_text}"
        if "Qwen3" in args.model:
            user_msg += " /no_think"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_msg})
        
        try:
            resp = lm(messages=messages)
            text = resp[0] if isinstance(resp, list) else str(resp)
        except:
            text = ""
        
        pred = extract_answer(text)
        binary.append(1 if pred == item["answer"] else 0)
        total_tokens += len(user_msg.split()) * 2 + 64

    acc = np.mean(binary)
    if return_binary:
        return acc, np.array(binary, dtype=float), total_tokens
    return acc, total_tokens


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
    boots = np.array([theta + (lr.standard_normal(M) * psi).mean() for _ in range(n_boot)])
    return theta, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def run_all_methods(scores, rho, tau, R, n_boot, seed):
    lr = np.random.default_rng(seed)
    M, K = scores.shape
    n_dev = int(rho * M)
    # M1: naive best-of (argmax over empirical column means).
    # Fixed: was scores[:, 0].mean(), which is shortlist[0] — not always the
    # empirical winner. Now correctly takes max over all K columns.
    col_means = scores.mean(axis=0)
    m1 = float(col_means.max())
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
        b = np.array([theta + (br.standard_normal(M) * psi_vec).mean() for _ in range(n_boot)])
        return [float(theta), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]]
    
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
# PHASE 1: Search
# ============================================================
print(f"\n{'='*60}\nPHASE 1 — DSPy Search\n{'='*60}")

SUBJECT_LABELS = {
    "math": ("math", "mathematics", "mathematics professor", "math tutor", "mathematics examination", "mathematics olympiad coach"),
    "law": ("law", "legal", "law professor", "legal expert", "bar examination", "constitutional law scholar"),
    "physics": ("physics", "physics", "physics professor", "physics tutor", "physics examination", "physics researcher"),
    "chemistry": ("chemistry", "chemistry", "chemistry professor", "chemistry tutor", "chemistry examination", "organic chemist"),
    "biology": ("biology", "biology", "biology professor", "biology tutor", "biology examination", "molecular biologist"),
    "engineering": ("engineering", "engineering", "engineering professor", "engineering tutor", "engineering examination", "engineering consultant"),
    "computer science": ("computer science", "computer science", "computer science professor", "CS tutor", "computer science examination", "software engineer"),
    "economics": ("economics", "economics", "economics professor", "economics tutor", "economics examination", "econometrician"),
    "business": ("business", "business", "business professor", "business consultant", "business examination", "strategy consultant"),
    "psychology": ("psychology", "psychology", "psychology professor", "psychology tutor", "psychology examination", "clinical psychologist"),
    "health": ("health", "health", "medical expert", "health tutor", "medical examination", "clinician"),
    "history": ("history", "history", "history professor", "history tutor", "history examination", "historian"),
    "philosophy": ("philosophy", "philosophy", "philosophy professor", "philosophy tutor", "philosophy examination", "philosopher"),
    "other": ("general knowledge", "general", "expert", "tutor", "examination", "scholar"),
}

# Fall back to generic wording if subject not in table
_lbl = SUBJECT_LABELS.get(args.subject,
                          (args.subject, args.subject, f"{args.subject} professor",
                           f"{args.subject} tutor", f"{args.subject} examination",
                           f"{args.subject} expert"))
short_name, field, prof, tutor, exam, coach = _lbl

SYSTEM_PROMPTS = [
    f"You are a helpful assistant that answers {short_name} questions concisely.",
    f"You are an expert {prof} with decades of experience.",
    f"You specialize in solving university-level {field} problems.",
    f"You are a precise {tutor} who always gives the correct answer.",
    f"You are a careful student taking an important {exam}.",
    f"You are a {coach} who thinks carefully before answering.",
    "",
]
BASE_INSTRUCTIONS = [
    f"Answer the following multiple-choice {short_name} question. Output ONLY the letter of the correct answer.",
    "Choose the correct option and reply with the letter only.",
    "Select the correct answer and reply with the letter only.",
    f"Answer the {short_name} question by selecting the correct option. Provide only the letter of the correct answer.",
    f"Carefully evaluate the {short_name} problem and output only the correct letter.",
    f"Analyze this {short_name} question and choose the correct option. Reply with one letter only.",
    "Select the correct answer from the options. Output only the letter corresponding to the correct choice.",
    "Choose the correct answer and output only the letter. No explanation required.",
    f"You are a {short_name} expert. Provide only the correct answer letter.",
    "Read the question carefully and choose the correct option. Output only the letter.",
]

# Generate more instructions via LLM
try:
    gen_resp = lm(messages=[{"role": "user", "content":
        f"Generate 40 different instruction variants for a {short_name} multiple-choice task. "
        f"Each should tell the model to output only the answer letter (A-J). "
        f"One per line, numbered 1-40."}])
    gen_text = gen_resp[0] if isinstance(gen_resp, list) else str(gen_resp)
    generated = [re.sub(r'^\d+[\.\)]\s*', '', l.strip()) for l in gen_text.strip().split("\n")]
    generated = [g for g in generated if 20 < len(g) < 200]
    all_instructions = list(set(BASE_INSTRUCTIONS + generated))
    print(f"  Generated {len(generated)} instructions, total: {len(all_instructions)}")
except:
    all_instructions = BASE_INSTRUCTIONS
    print(f"  Using {len(all_instructions)} base instructions")

# Build candidates
candidates = []
for instr in all_instructions:
    for sp in SYSTEM_PROMPTS:
        for nd in [0, 1, 2, 3]:
            for strat in ["random", "diverse", "hard", "easy"]:
                candidates.append({"instruction": instr, "system_prompt": sp,
                                   "n_demos": nd, "demo_strategy": strat})
rng.shuffle(candidates)
print(f"[Search] {len(candidates)} candidates, budgets: {BUDGETS}")

# Run search
evaluated = []
cum_tok = 0
max_b = max(BUDGETS)
cp_pool = {str(b): [] for b in BUDGETS}

for i, c in enumerate(candidates):
    if cum_tok >= max_b:
        break
    c["demos"] = select_demos(c["n_demos"], c["demo_strategy"], trainset, rng)
    c["id"] = f"candidate_{i:04d}"
    acc, tok = score_artifact(c, trainset)
    cum_tok += tok
    c["score"] = acc
    evaluated.append(c)
    for b in BUDGETS:
        if cum_tok <= b:
            cp_pool[str(b)].append(c)
    if (i+1) % 10 == 0:
        print(f"  #{i+1} | {cum_tok:,}/{max_b:,} | score={acc:.3f}")

print(f"  Evaluated: {len(evaluated)} | Tokens: {cum_tok:,}")

# Top-K per budget
checkpoints = {}
for b in BUDGETS:
    bk = str(b)
    pool = sorted(cp_pool[bk], key=lambda x: x["score"], reverse=True)[:args.K]
    checkpoints[bk] = [{"id": c["id"], "score": c["score"],
                         "instruction": c["instruction"], "system_prompt": c["system_prompt"],
                         "n_demos": c["n_demos"], "demo_strategy": c["demo_strategy"]} for c in pool]
    print(f"  B={b:>9,}: top scores = {[c['score'] for c in pool[:5]]}")

with open(RESULTS_DIR / "checkpoints.json", "w") as f:
    json.dump(checkpoints, f, indent=2)

# ============================================================
# System B
# ============================================================
print(f"\n[System B] Scoring fixed prompt...")
sys_b = {"instruction": BASE_INSTRUCTIONS[0],
         "system_prompt": SYSTEM_PROMPTS[0], "demos": []}
_, col_b, _ = score_artifact(sys_b, evalset, return_binary=True)
np.save(RESULTS_DIR / "col_b.npy", col_b)
print(f"  System B accuracy: {col_b.mean():.3f}")

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
        print("  [SKIP]"); continue

    # Build tensor
    scores_a = np.zeros((M, K))
    for ki, art in enumerate(top_k):
        a = {"instruction": art["instruction"], "system_prompt": art["system_prompt"],
             "demos": select_demos(art["n_demos"], art["demo_strategy"], trainset, rng)}
        t0 = time.time()
        acc, binary, _ = score_artifact(a, evalset, return_binary=True)
        scores_a[:, ki] = binary
        print(f"  {ki+1}/{K}: {art['id']} acc={acc:.3f} ({time.time()-t0:.0f}s)")
    
    np.save(RESULTS_DIR / f"tensor_a_B{B}.npy", scores_a)

    # M7 for System A and B
    ta, la, ha = run_m7(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    tb, lb, hb = run_m7(col_b.reshape(-1,1), args.rho, args.tau, args.R, args.n_boot, args.seed+1)
    
    # All methods
    comp = run_all_methods(scores_a, args.rho, args.tau, args.R, args.n_boot, args.seed)
    comp["system_b"] = float(col_b.mean())
    
    # Ground truth
    gt_a, gt_std = compute_ground_truth(scores_a, args.rho, args.tau)
    gt_b = float(col_b.mean())
    
    print(f"\n  M7: θ̃_A={ta:.4f} [{la:.3f},{ha:.3f}] | θ̃_B={tb:.4f}")
    print(f"  GT: θ*_A={gt_a:.4f} | θ*_B={gt_b:.4f}")
    print(f"  M7 covers θ*_A? {'YES' if la <= gt_a <= ha else 'NO'}")
    
    for name in ["m1","m3","m4","m5","m6","m7"]:
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

output = {"model": args.model,
          "config": vars(args),
          "results": serialize(all_results)}

with open(RESULTS_DIR / "study_e_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE — {args.model}")
print(f"Results: {RESULTS_DIR}/")
print(f"{'='*60}")
