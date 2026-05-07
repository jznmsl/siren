"""
Ground Truth Computation for Study E
=====================================
For each budget checkpoint, compute the exact procedure-level ground truth
by Monte Carlo over all possible dev/held-out splits.

Then check whether M7's CI covers the ground truth.

Usage:
    cd <WORKDIR>
    python compute_ground_truth.py
"""

import numpy as np
import json
from pathlib import Path

# === Config ===
RESULTS_DIR = Path("results/study_e_overnight")
RHO = 0.5          # dev fraction (same as study_e_v2.py)
TAU = 1.0           # softmax temperature
R = 10              # number of repeated splits (same as overnight run)
N_MC = 100_000      # Monte Carlo samples for ground truth
SEED = 42

BUDGETS = [500_000, 1_500_000, 3_000_000, 6_500_000]

def softmax(x, tau):
    z = x / tau - (x / tau).max()
    e = np.exp(z)
    return e / e.sum()

def compute_ground_truth(scores, rho, tau, n_mc, rng):
    """
    Ground truth = E_split[ softmax(mean_dev_scores / tau) · mean_heldout_scores ]
    
    scores: [M, K] binary score matrix
    Returns: (theta_true, std_of_mc_estimates)
    """
    M, K = scores.shape
    n_dev = int(rho * M)
    
    results = np.empty(n_mc)
    for i in range(n_mc):
        idx = rng.permutation(M)
        dev_idx = idx[:n_dev]
        heldout_idx = idx[n_dev:]
        
        S_dev = scores[dev_idx].mean(axis=0)     # [K]
        q = softmax(S_dev, tau)                    # [K]
        T_heldout = scores[heldout_idx].mean(axis=0)  # [K]
        results[i] = q @ T_heldout
    
    return results.mean(), results.std()

def main():
    rng = np.random.default_rng(SEED)
    
    # Load overnight results for M7 CIs
    with open(RESULTS_DIR / "study_e_overnight_results.json") as f:
        overnight = json.load(f)
    
    # Load System B scores
    col_b = np.load(RESULTS_DIR / "col_b.npy")
    M = len(col_b)
    
    print("=" * 70)
    print("Ground Truth Computation for Study E")
    print(f"M={M} items, rho={RHO}, tau={TAU}, MC samples={N_MC}")
    print("=" * 70)
    
    all_results = {}
    
    for B in BUDGETS:
        print(f"\n--- Budget B = {B:,} ---")
        
        # Load System A score tensor for this budget
        tensor_path = RESULTS_DIR / f"tensor_a_B{B}.npy"
        if not tensor_path.exists():
            print(f"  [SKIP] {tensor_path} not found")
            continue
        
        scores_a = np.load(tensor_path)  # [M, K]
        K = scores_a.shape[1]
        print(f"  Score tensor: [{M}, {K}]")
        
        # --- System A ground truth ---
        theta_true_a, mc_std_a = compute_ground_truth(scores_a, RHO, TAU, N_MC, rng)
        
        # --- System B ground truth ---
        # System B has K=1 (single fixed prompt), no selection needed
        # Ground truth is simply the population mean
        theta_true_b = col_b.mean()
        
        # --- Load M7 results ---
        bkey = str(B)
        m7_a = overnight[bkey]["m7"]["system_a"]
        m7_b = overnight[bkey]["m7"]["system_b"]
        
        ci_a = (m7_a["ci_lo"], m7_a["ci_hi"])
        ci_b = (m7_b["ci_lo"], m7_b["ci_hi"])
        
        covers_a = ci_a[0] <= theta_true_a <= ci_a[1]
        covers_b = ci_b[0] <= theta_true_b <= ci_b[1]
        
        # --- Also check M1 ---
        m1_val = overnight[bkey]["comparison"]["m1"]
        
        print(f"\n  System A:")
        print(f"    Ground truth θ*   = {theta_true_a:.4f}  (MC std: {mc_std_a:.5f})")
        print(f"    M7 estimate θ̃     = {m7_a['theta']:.4f}")
        print(f"    M7 CI             = [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
        print(f"    M7 covers θ*?     = {'YES ✓' if covers_a else 'NO ✗'}")
        print(f"    M1 (naive)        = {m1_val:.4f}")
        print(f"    M1 bias           = {m1_val - theta_true_a:+.4f}")
        
        print(f"\n  System B:")
        print(f"    Ground truth θ*   = {theta_true_b:.4f}")
        print(f"    M7 estimate θ̃     = {m7_b['theta']:.4f}")
        print(f"    M7 CI             = [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
        print(f"    M7 covers θ*?     = {'YES ✓' if covers_b else 'NO ✗'}")
        
        # --- True comparison ---
        true_delta = theta_true_a - theta_true_b
        m7_delta = m7_a['theta'] - m7_b['theta']
        m1_delta = m1_val - theta_true_b
        
        print(f"\n  Comparison A vs B:")
        print(f"    True Δ = θ*_A - θ*_B     = {true_delta:+.4f}  ({'A>B' if true_delta > 0 else 'A<B'})")
        print(f"    M7   Δ = θ̃_A - θ̃_B      = {m7_delta:+.4f}  ({'A>B' if m7_delta > 0 else 'A<B'})  {'✓' if (true_delta > 0) == (m7_delta > 0) else '✗'}")
        print(f"    M1   Δ = M1  - θ*_B      = {m1_delta:+.4f}  ({'A>B' if m1_delta > 0 else 'A<B'})  {'✓' if (true_delta > 0) == (m1_delta > 0) else '✗'}")
        
        all_results[B] = {
            "theta_true_a": float(theta_true_a),
            "theta_true_b": float(theta_true_b),
            "true_delta": float(true_delta),
            "m7_covers_a": bool(covers_a),
            "m7_covers_b": bool(covers_b),
            "m1_bias": float(m1_val - theta_true_a),
        }
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    coverage_count = sum(1 for r in all_results.values() if r["m7_covers_a"])
    total = len(all_results)
    print(f"\nM7 CI coverage (System A): {coverage_count}/{total}")
    
    coverage_count_b = sum(1 for r in all_results.values() if r["m7_covers_b"])
    print(f"M7 CI coverage (System B): {coverage_count_b}/{total}")
    
    print(f"\nM1 bias across budgets:")
    for B, r in all_results.items():
        print(f"  B={B:>9,}: M1 overestimates by {r['m1_bias']:+.4f} ({r['m1_bias']*100:+.1f}pp)")
    
    print(f"\nDirection correctness:")
    for B, r in all_results.items():
        direction = "A>B" if r["true_delta"] > 0 else "A<B"
        print(f"  B={B:>9,}: true={direction}, Δ={r['true_delta']:+.4f}")
    
    # Save results
    out_path = RESULTS_DIR / "ground_truth_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()