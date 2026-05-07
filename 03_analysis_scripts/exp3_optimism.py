#!/usr/bin/env python
"""
Experiment 3: Same-Data Optimism
=================================
Validates Proposition 4 (same-data best-of reporting is intrinsically
optimistic) and shows that unequal search effort creates spurious
ranking advantages between systems that are actually identical.

Setup:
  - System A has H_A prompt variants  (sweep: 3 to 50)
  - System B has H_B = 3 prompt variants  (fixed)
  - ALL artifacts across both systems have IDENTICAL true quality
  - The ONLY difference is how many variants each system searched

Paper reference:
  - Proposition 4  (optimism of same-data best-of reporting)
  - Study C         (Section 5.4)

Usage:
  python exp3_optimism.py              # full run  (~5-10 min)
  python exp3_optimism.py --quick      # quick test (~1-2 min)
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import expit
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed


# ================================================================
#  Configuration
# ================================================================
@dataclass
class Exp3Config:
    # --- DGP ---
    q_0: float = 0.5          # ALL artifacts have the same quality
    diff_low: float = -2.0
    diff_high: float = 2.0

    # --- Sweep ---
    HA_list: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    HB: int = 3               # System B has 3 variants (fixed)

    # --- Fixed ---
    M: int = 500
    R: int = 5
    rho: float = 0.5
    tau: float = 0.1
    n_bootstrap: int = 1000
    alpha: float = 0.05

    # --- Simulation ---
    N_sim: int = 2000
    N_gt: int = 50_000        # large, for precise ground truth

    # --- Compute ---
    n_jobs: int = -1
    seed: int = 7777

    # --- Output ---
    results_dir: str = "results"
    figures_dir: str = "figures"


def quick_config() -> Exp3Config:
    return Exp3Config(
        HA_list=[3, 10, 50],
        N_sim=500,
        N_gt=20_000,
    )


# ================================================================
#  Shared helpers
# ================================================================
def generate_scores(M, K, q_0, rng, difficulties):
    """
    Generate M x K Bernoulli score matrix.  ALL K artifacts share quality q_0.
    Items (difficulties) are passed in so both systems use the same items.
    """
    probs = expit(q_0 - difficulties)                                     # [M]
    scores = rng.binomial(1, probs[:, np.newaxis] * np.ones((1, K)))      # [M, K]
    return scores.astype(np.float64)


def softmax(x, tau):
    z = x / tau;  z -= z.max();  e = np.exp(z)
    return e / e.sum()


def compute_ground_truth(q_0, diff_low, diff_high, N_gt, rng):
    """E[ sigmoid(q_0 - delta) ] for delta ~ Uniform."""
    deltas = rng.uniform(diff_low, diff_high, size=N_gt)
    return float(expit(q_0 - deltas).mean())


# ================================================================
#  Method 1:  Same-data best-of  (the broken baseline)
# ================================================================
def same_data_bestof(scores):
    """Pick highest-mean artifact, report that mean on the SAME data."""
    return float(scores.mean(axis=0).max())


# ================================================================
#  Method 2:  Single-split best-of  (partial fix)
# ================================================================
def single_split_bestof(scores, rho, rng):
    """Split once.  Pick best on dev, evaluate on held-out eval."""
    M = scores.shape[0]
    n_dev = int(M * rho)
    perm = rng.permutation(M)
    dev_idx, eval_idx = perm[:n_dev], perm[n_dev:]
    k_star = int(scores[dev_idx].mean(axis=0).argmax())
    return float(scores[eval_idx, k_star].mean())


# ================================================================
#  Method 3:  Proposed  (repeated-split + soft + bootstrap CI)
# ================================================================
def proposed_method(scores, splits, tau, n_bootstrap, alpha, rng):
    """Full pipeline for one system.  Returns (theta_tilde, half_width)."""
    M, K = scores.shape
    R = len(splits)
    eval_sizes = np.array([len(s[1]) for s in splits])
    weights = eval_sizes / eval_sizes.sum()

    Y_hats = np.empty(R)
    soft_info = []
    for r, (dev_idx, eval_idx) in enumerate(splits):
        S = scores[dev_idx].mean(axis=0)
        T = scores[eval_idx].mean(axis=0)
        q = softmax(S, tau)
        Y_hats[r] = q @ T
        soft_info.append((dev_idx, eval_idx, S, T, q))
    theta = float(weights @ Y_hats)

    # Influence (eval + dev)
    psi = np.zeros(M)
    for r in range(R):
        d, e, S, T, q = soft_info[r]
        nd, ne, w = len(d), len(e), weights[r]
        psi[e] += w * (M / ne) * ((scores[e] - T) @ q)
        c = q * (T - q @ T) / tau
        psi[d] += w * (M / nd) * ((scores[d] - S) @ c)

    # Bootstrap
    xi = rng.standard_normal((n_bootstrap, M))
    G = xi @ psi / np.sqrt(M)
    hw = float(np.quantile(np.abs(G), 1 - alpha) / np.sqrt(M))
    return theta, hw


def generate_splits(M, R, rho, rng):
    n_dev = int(M * rho)
    splits = []
    for _ in range(R):
        perm = rng.permutation(M)
        splits.append((perm[:n_dev].copy(), perm[n_dev:].copy()))
    return splits


# ================================================================
#  Run one H_A level
# ================================================================
def run_one_HA(HA, cfg: Exp3Config, config_seed):

    gt_stream, sim_stream = config_seed.spawn(2)
    rng_gt  = np.random.default_rng(gt_stream)
    rng_sim = np.random.default_rng(sim_stream)

    gt = compute_ground_truth(cfg.q_0, cfg.diff_low, cfg.diff_high,
                              cfg.N_gt, rng_gt)

    # Storage
    sd_A = np.empty(cfg.N_sim);  sd_B = np.empty(cfg.N_sim)
    ss_A = np.empty(cfg.N_sim);  ss_B = np.empty(cfg.N_sim)
    pr_A = np.empty(cfg.N_sim);  pr_B = np.empty(cfg.N_sim)
    hw_A = np.empty(cfg.N_sim);  hw_B = np.empty(cfg.N_sim)

    for t in range(cfg.N_sim):
        diffs = rng_sim.uniform(cfg.diff_low, cfg.diff_high, size=cfg.M)
        sA = generate_scores(cfg.M, HA,     cfg.q_0, rng_sim, diffs)
        sB = generate_scores(cfg.M, cfg.HB, cfg.q_0, rng_sim, diffs)

        # Same-data
        sd_A[t] = same_data_bestof(sA)
        sd_B[t] = same_data_bestof(sB)

        # Single-split
        ss_A[t] = single_split_bestof(sA, cfg.rho, rng_sim)
        ss_B[t] = single_split_bestof(sB, cfg.rho, rng_sim)

        # Proposed
        splits = generate_splits(cfg.M, cfg.R, cfg.rho, rng_sim)
        pr_A[t], hw_A[t] = proposed_method(sA, splits, cfg.tau,
                                           cfg.n_bootstrap, cfg.alpha, rng_sim)
        pr_B[t], hw_B[t] = proposed_method(sB, splits, cfg.tau,
                                           cfg.n_bootstrap, cfg.alpha, rng_sim)

    # ---- Metrics ----
    a_wins_prop = (pr_A - hw_A) > (pr_B + hw_B)
    b_wins_prop = (pr_B - hw_B) > (pr_A + hw_A)

    return {
        "HA":              HA,
        "HB":              cfg.HB,
        "gt":              gt,
        "opt_samedata_A":  float((sd_A - gt).mean()),
        "opt_samedata_B":  float((sd_B - gt).mean()),
        "opt_proposed_A":  float((pr_A - gt).mean()),
        "opt_proposed_B":  float((pr_B - gt).mean()),
        "spurious_gap":    float((sd_A - sd_B).mean()),
        "fwr_samedata":    float((sd_A > sd_B).mean()),
        "fwr_split":       float((ss_A > ss_B).mean()),
        "fwr_proposed_A":  float(a_wins_prop.mean()),
        "fwr_proposed_B":  float(b_wins_prop.mean()),
        "fwr_inconclusive": float((~a_wins_prop & ~b_wins_prop).mean()),
    }


# ================================================================
#  Sweep
# ================================================================
def run_sweep(cfg: Exp3Config) -> pd.DataFrame:
    n = len(cfg.HA_list)
    ss = np.random.SeedSequence(cfg.seed)
    seeds = ss.spawn(n)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Experiment 3: Same-Data Optimism Bias           ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  H_A values : {str(cfg.HA_list):<36s}║")
    print(f"║  H_B (fixed): {cfg.HB:>5d}                              ║")
    print(f"║  M          : {cfg.M:>5d}                              ║")
    print(f"║  N_sim      : {cfg.N_sim:>5d}  trials per H_A            ║")
    print(f"║  All artifacts have IDENTICAL quality            ║")
    print(f"╚══════════════════════════════════════════════════╝\n")

    t0 = time.time()
    print(f"  Pilot (H_A={cfg.HA_list[0]}) ...", flush=True)
    tp = time.time()
    _ = run_one_HA(cfg.HA_list[0], cfg, seeds[0])
    dt = time.time() - tp
    nw = os.cpu_count() if cfg.n_jobs == -1 else cfg.n_jobs
    print(f"  Pilot: {dt:.1f}s  →  est total: {dt*n/max(1,nw):.0f}s\n", flush=True)

    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_one_HA)(h, cfg, seeds[i]) for i, h in enumerate(cfg.HA_list)
    )
    print(f"\n  Wall-clock: {time.time()-t0:.1f}s")
    return pd.DataFrame(results)


# ================================================================
#  Plotting
# ================================================================
def make_plots(df: pd.DataFrame, cfg: Exp3Config):
    os.makedirs(cfg.figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.15)
    pal = sns.color_palette("colorblind")
    df = df.sort_values("HA")
    labels = df["HA"].astype(str).tolist()

    gt = df["gt"].iloc[0]
    sigma = np.sqrt(gt * (1 - gt))
    theory = sigma / np.sqrt(cfg.M) * np.sqrt(2 * np.log(df["HA"].values))

    # ---- Plot 1: Optimism bias ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(labels, df["opt_samedata_A"] * 100, "o-",  color=pal[3], lw=2.2, ms=8,
            label="Same-data  System A")
    ax.plot(labels, df["opt_samedata_B"] * 100, "s-",  color=pal[0], lw=2.2, ms=8,
            label=f"Same-data  System B (H={cfg.HB})")
    ax.plot(labels, df["opt_proposed_A"] * 100, "^--", color=pal[2], lw=2, ms=8,
            label="Proposed  System A")
    ax.plot(labels, theory * 100, "k:", lw=1.5, alpha=0.5,
            label=r"Theory: $\frac{\sigma}{\sqrt{M}}\sqrt{2\log H_A}$")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("System A library size  $H_A$")
    ax.set_ylabel("Optimism bias  (pct points)")
    ax.set_title("Same-Data Reporting Creates Spurious Optimism")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.figures_dir, "exp3_optimism_bias.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved exp3_optimism_bias.png")

    # ---- Plot 2: False winner rate ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(labels, df["fwr_samedata"] * 100, "o-",  color=pal[3], lw=2.2, ms=8,
            label="Same-data best-of")
    ax.plot(labels, df["fwr_split"] * 100,    "D-",  color=pal[1], lw=2.2, ms=8,
            label="Single-split")
    ax.plot(labels, df["fwr_proposed_A"] * 100, "^-", color=pal[2], lw=2.2, ms=8,
            label="Proposed (declares A)")
    ax.axhline(50, ls="--", color="grey", lw=1.2, label="Fair baseline = 50%")
    ax.set_xlabel("System A library size  $H_A$")
    ax.set_ylabel("% trials declaring System A the winner")
    ax.set_ylim(-2, 105)
    ax.set_title("Unequal Search Effort → Unfair Rankings")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.figures_dir, "exp3_false_winner_rate.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved exp3_false_winner_rate.png")

    # ---- Plot 3: Proposed decisions (stacked bar) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.5
    bot = np.zeros(len(df))
    ax.bar(labels, df["fwr_proposed_A"]*100, w, color=pal[3], label="A wins")
    bot += df["fwr_proposed_A"].values*100
    ax.bar(labels, df["fwr_proposed_B"]*100, w, bottom=bot, color=pal[0], label="B wins")
    bot += df["fwr_proposed_B"].values*100
    ax.bar(labels, df["fwr_inconclusive"]*100, w, bottom=bot,
           color="lightgrey", edgecolor="white", label="Inconclusive")
    ax.set_xlabel("System A library size  $H_A$")
    ax.set_ylabel("% of trials")
    ax.set_title("Proposed Method: Almost Always Reports 'Inconclusive'")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.figures_dir, "exp3_proposed_decisions.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved exp3_proposed_decisions.png")

    # ---- Plot 4: Spurious gap ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, df["spurious_gap"]*100, color=pal[3], edgecolor="white", width=0.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("System A library size  $H_A$")
    ax.set_ylabel("Spurious gap  (pct points)")
    ax.set_title("Same-Data: Fabricated Advantage for the System that Searched More")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.figures_dir, "exp3_spurious_gap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved exp3_spurious_gap.png")


# ================================================================
#  Summary
# ================================================================
def print_summary(df, cfg):
    print("\n" + "=" * 88)
    print("  RESULTS — Experiment 3: Same-Data Optimism")
    print("=" * 88)
    print(f"  Ground truth: {df['gt'].iloc[0]:.4f}   (all systems identical)")
    print(f"  System B: H_B = {cfg.HB}")
    print()
    print(f"  {'H_A':>4s} │ Optim_A  Optim_B  Gap   │ SD%A   Split%A  Prop%A  Incon%")
    print("  " + "─" * 74)
    for _, r in df.sort_values("HA").iterrows():
        print(f"  {r['HA']:4.0f} │ {r['opt_samedata_A']*100:+6.2f}%"
              f"  {r['opt_samedata_B']*100:+6.2f}%"
              f"  {r['spurious_gap']*100:+5.2f}% │"
              f" {r['fwr_samedata']*100:5.1f}"
              f"   {r['fwr_split']*100:7.1f}"
              f"    {r['fwr_proposed_A']*100:4.1f}"
              f"    {r['fwr_inconclusive']*100:5.1f}")
    print("=" * 88)

    big = df.loc[df["HA"].idxmax()]
    print(f"\n  Headlines (H_A = {big['HA']:.0f} vs H_B = {cfg.HB}):")
    print(f"    Same-data gives A a fabricated {big['spurious_gap']*100:+.2f} pct-pt advantage")
    print(f"    Same-data declares A winner {big['fwr_samedata']*100:.1f}% of the time"
          f"  (fair = 50%)")
    print(f"    Proposed method declares A winner {big['fwr_proposed_A']*100:.1f}%"
          f"  (inconclusive {big['fwr_inconclusive']*100:.1f}%)")

    if big["fwr_samedata"] > 0.60:
        print(f"\n    ✓ Same-data reporting is unfair when search effort differs")
    if big["fwr_proposed_A"] < 0.05:
        print(f"    ✓ Proposed method is immune to the search-effort bias")
    print()


# ================================================================
def main():
    quick = "--quick" in sys.argv
    if quick:
        print(">>> QUICK mode <<<\n")
        cfg = quick_config()
    else:
        cfg = Exp3Config()

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    df = run_sweep(cfg)
    csv = os.path.join(cfg.results_dir, "exp3_results.csv")
    df.to_csv(csv, index=False, float_format="%.6f")
    print(f"\n  Saved {csv}")

    print_summary(df, cfg)
    print("  Generating figures ...")
    make_plots(df, cfg)
    print("\n  Done!\n")

if __name__ == "__main__":
    main()
