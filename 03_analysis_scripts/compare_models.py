"""
Compare Study E results across all models.
Usage: python compare_models.py
"""
import json
from pathlib import Path

RESULTS_ROOT = Path("results/multimodel")
QWEN_DIR = Path("results/study_e_overnight")

def load_qwen():
    r_file = QWEN_DIR / "study_e_overnight_results.json"
    gt_file = QWEN_DIR / "ground_truth_results.json"
    if not r_file.exists(): return None
    with open(r_file) as f: results = json.load(f)
    gt = {}
    if gt_file.exists():
        with open(gt_file) as f: gt = json.load(f)
    return {"model": "Qwen/Qwen3-8B", "results": results, "gt": gt}

def load_multimodel(d):
    f = d / "study_e_results.json"
    if not f.exists(): return None
    with open(f) as fh: data = json.load(fh)
    gt = {}
    for b, br in data.get("results", {}).items():
        if "ground_truth" in br:
            gt[b] = br["ground_truth"]
    return {"model": data["model"], "results": data["results"], "gt": gt}

def main():
    models = {}
    q = load_qwen()
    if q: models[q["model"]] = q
    if RESULTS_ROOT.exists():
        for d in sorted(RESULTS_ROOT.iterdir()):
            if d.is_dir():
                m = load_multimodel(d)
                if m: models[m["model"]] = m

    budgets = ["500000","1500000","3000000","6500000"]
    print(f"Models found: {list(models.keys())}\n")

    # Table: M7 theta per budget
    print(f"{'Model':<42s} | {'B=500k':>8s} | {'B=1.5M':>8s} | {'B=3M':>8s} | {'B=6.5M':>8s} | Coverage")
    print("-"*100)
    for name, data in models.items():
        r = data["results"]; gt = data["gt"]
        row = f"{name:<42s}"
        cov = 0; tot = 0
        for b in budgets:
            if b in r:
                br = r[b]
                if isinstance(br.get("m7"), dict) and "system_a" in br["m7"]:
                    t = float(br["m7"]["system_a"]["theta"])
                elif isinstance(br.get("m7"), dict) and "theta" in br["m7"]:
                    t = float(br["m7"]["theta"])
                else: t = None
                
                covers = gt.get(b, {}).get("m7_covers_a")
                if covers in [True, "True"]: cov += 1
                tot += 1
                row += f" | {t:.3f}  " if t else " |   ---   "
            else:
                row += f" |   ---   "
        row += f" | {cov}/{tot}"
        print(row)

    # B=500k detail
    print(f"\n{'='*80}\nB=500k: Conclusion Reversal Check\n{'='*80}")
    for name, data in models.items():
        r = data["results"]; gt = data["gt"]
        if "500000" not in r: continue
        br = r["500000"]
        gt_a = gt.get("500000", {}).get("theta_true_a", "?")
        gt_b = gt.get("500000", {}).get("theta_true_b", "?")
        true_dir = "A>B" if isinstance(gt_a, float) and isinstance(gt_b, float) and gt_a > gt_b else "A<B"
        print(f"\n  {name}: θ*_A={gt_a}, θ*_B={gt_b} → {true_dir}")
        comp = br.get("comparison", {})
        for m in ["m1","m7"]:
            v = comp.get(m)
            if v is None: continue
            est = v[0] if isinstance(v, list) else v
            sb = comp.get("system_b", gt_b)
            says = "A>B" if est > sb else "A<B"
            ok = "✓" if says == true_dir else "✗"
            print(f"    {m}: {float(est):.4f} → {says} {ok}")

if __name__ == "__main__":
    main()
