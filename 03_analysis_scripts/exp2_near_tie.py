#!/usr/bin/env python
"""
Experiment 2: Near-Tie Hard vs Soft Selection
==============================================
Validates Proposition 3 (hard-selection nonregularity near ties).

Shows that hard argmax selection breaks down (severe undercoverage)
when the top two artifacts are close in quality, while soft (softmax)
selection maintains correct coverage.  Also tests the instability-
triggered adaptive rule from the paper's Section 5.1.

Paper reference:
  - Proposition 3  (hard-selection nonregularity near ties)
  - Proposition 2  (hard-selection add-on under positive margin)
  - Study B        (Section 5.3)

Usage:
  python exp2_near_tie.py              # full run  (~5-15 min)
  python exp2_near_tie.py --quick      # quick test (~1-2 min)

Required packages:
  pip install numpy scipy matplotlib seaborn pandas joblib tqdm
"""

import sys
import os
import time
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.special import expit
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw):
        total = kw.get("total", None)
        desc  = kw.get("desc", "")
        for i, x in enumerate(iterable):
            if total and i % max(1, total // 10) == 0:
                print(f"  {desc} {i}/{total} ...", flush=True)
            yield x


# ================================================================
#  Configuration
# ================================================================
@dataclass
class Exp2Config:
    """All parameters for Experiment 2."""

    # --- DGP ---
    q_base: float = 0.5       # base quality for artifact 2 (fixed)
    diff_low: float = -2.0
    diff_high: float = 2.0

    # --- Sweep axis: quality gap ---
    # Fine-grained: 0.00 to 0.80 in steps of 0.01  (81 gap levels)
    gap_list: List[float] = field(
        default_factory=lambda: [round(i * 0.01, 2) for i in range(81)]
    )

    # --- Fixed ---
    K: int = 2                # two artifacts (matches Proposition 3)
    M: int = 500              # practical benchmark size
    R: int = 5                # recommended repeated splits
    rho: float = 0.5
    tau: float = 0.1          # softmax temperature
    instability_threshold: float = 0.10   # adaptive rule threshold
    n_bootstrap: int = 1000
    alpha: float = 0.05

    # --- Simulation scale ---
    N_sim: int = 2000
    N_gt: int = 10_000

    # --- Compute ---
    n_jobs: int = -1
    seed: int = 2024

    # --- Output ---
    results_dir: str = "results"
    figures_dir: str = "figures"


def quick_config() -> Exp2Config:
    return Exp2Config(
        gap_list=[round(i * 0.04, 2) for i in range(21)],  # 0.00 to 0.80 step 0.04
        N_sim=400,
        N_gt=2_000,
    )


# ================================================================
#  Shared DGP
# ================================================================
def generate_scores(
    M: int,
    qualities: np.ndarray,
    rng: np.random.Generator,
    diff_low: float = -2.0,
    diff_high: float = 2.0,
) -> np.ndarray:
    """Bernoulli score matrix:  scores[i,k] ~ Bern(sigmoid(q_k - delta_i))."""
    difficulties = rng.uniform(diff_low, diff_high, size=M)
    logits = qualities[np.newaxis, :] - difficulties[:, np.newaxis]
    probs = expit(logits)
    return rng.binomial(1, probs).astype(np.float64)


# ================================================================
#  Selection rules
# ================================================================
def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ================================================================
#  Split generation  (shared across methods within a trial)
# ================================================================
def generate_splits(M: int, R: int, rho: float, rng: np.random.Generator):
    """Create R random dev/eval splits.  Returns list of (dev_idx, eval_idx)."""
    n_dev = int(M * rho)
    splits = []
    for _ in range(R):
        perm = rng.permutation(M)
        splits.append((perm[:n_dev].copy(), perm[n_dev:].copy()))
    return splits


def compute_split_stats(scores: np.ndarray, splits: list):
    """Pre-compute dev/eval means for every split."""
    return [
        (d, e, scores[d].mean(axis=0), scores[e].mean(axis=0))
        for d, e in splits
    ]


# ================================================================
#  Method 1:  Hard argmax selection
# ================================================================
def run_hard(
    scores: np.ndarray,
    stats: list,
    weights: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Hard argmax on dev, evaluate on eval.

    Influence for hard selection (under positive margin, Proposition 2):
      The selector derivative Dg = 0 because argmax is locally constant
      away from tie points.  So the dev contribution vanishes, and:

        psi_i  =  sum_r omega_r * (M/|E_r|) * 1(i in E_r) * (Z_{i,k*_r} - T_hat_{r,k*_r})

    This is the ONLY source of variability the bootstrap captures.
    Near ties, the actual variability also includes the *discontinuous*
    selection jumps, which psi_hard completely misses  →  CI too narrow
    →  undercoverage.  This is exactly Proposition 3 in action.
    """
    M, K = scores.shape
    R = len(stats)

    Y_hats  = np.empty(R)
    winners = np.empty(R, dtype=int)

    for r, (dev_idx, eval_idx, S_hat, T_hat) in enumerate(stats):
        k_star = int(np.argmax(S_hat))
        winners[r] = k_star
        Y_hats[r] = T_hat[k_star]

    theta = float(weights @ Y_hats)

    # Influence — eval contribution only (no dev term for hard selection)
    psi = np.zeros(M)
    for r, (dev_idx, eval_idx, S_hat, T_hat) in enumerate(stats):
        k_star = winners[r]
        n_eval = len(eval_idx)
        psi[eval_idx] += weights[r] * (M / n_eval) * (
            scores[eval_idx, k_star] - T_hat[k_star]
        )

    return theta, psi, winners


# ================================================================
#  Method 2:  Soft (softmax) selection
# ================================================================
def run_soft(
    scores: np.ndarray,
    stats: list,
    weights: np.ndarray,
    tau: float,
) -> Tuple[float, np.ndarray]:
    """
    Softmax selection on dev, evaluate on eval.

    Influence (Theorem 1) has two terms per split:
      eval term:  q_hat^T (Z_i - T_hat) * M/|E_r|
      dev term :  c^T (Z_i - S_hat) * M/|D_r|
        where  c_j = q_j*(T_j - q·T) / tau   (softmax Jacobian)

    Because the softmax Jacobian is non-zero, the bootstrap correctly
    captures how dev-score perturbations propagate through selection.
    """
    M, K = scores.shape
    R = len(stats)

    Y_hats = np.empty(R)
    soft_info = []

    for r, (dev_idx, eval_idx, S_hat, T_hat) in enumerate(stats):
        q_hat = softmax(S_hat, tau)
        Y_hats[r] = q_hat @ T_hat
        soft_info.append((dev_idx, eval_idx, S_hat, T_hat, q_hat))

    theta = float(weights @ Y_hats)

    # Influence (eval + dev)
    psi = np.zeros(M)
    for r in range(R):
        dev_idx, eval_idx, S_hat, T_hat, q_hat = soft_info[r]
        n_dev  = len(dev_idx)
        n_eval = len(eval_idx)
        omega  = weights[r]

        # Eval contribution
        eval_resid = scores[eval_idx] - T_hat
        psi[eval_idx] += omega * (M / n_eval) * (eval_resid @ q_hat)

        # Dev contribution (via softmax Jacobian)
        q_dot_T = q_hat @ T_hat
        c = q_hat * (T_hat - q_dot_T) / tau
        dev_resid = scores[dev_idx] - S_hat
        psi[dev_idx] += omega * (M / n_dev) * (dev_resid @ c)

    return theta, psi


# ================================================================
#  Multiplier bootstrap CI
# ================================================================
def bootstrap_ci(
    psi: np.ndarray, M: int,
    n_bootstrap: int, alpha: float,
    rng: np.random.Generator,
) -> float:
    """Returns half-width of symmetric (1-alpha) CI."""
    xi = rng.standard_normal((n_bootstrap, M))
    G_star = xi @ psi / np.sqrt(M)
    c_alpha = np.quantile(np.abs(G_star), 1 - alpha)
    return float(c_alpha / np.sqrt(M))


# ================================================================
#  Winner instability diagnostic  (paper Section 5.1)
# ================================================================
def winner_instability(winners: np.ndarray) -> float:
    """
    pi_win = 1 - (fraction of splits won by the most frequent winner).

    pi_win = 0   → all splits agree (stable hard selection)
    pi_win ≈ 0.5 → winner changes almost every split (near tie)

    The adaptive rule: use hard if pi_win <= threshold, else switch to soft.
    """
    _, counts = np.unique(winners, return_counts=True)
    return 1.0 - counts.max() / len(winners)


# ================================================================
#  Ground truth (Monte Carlo, all three methods in one loop)
# ================================================================
def compute_ground_truths(
    M: int, qualities: np.ndarray, R: int,
    rho: float, tau: float, threshold: float,
    N_gt: int, rng: np.random.Generator,
    diff_low: float, diff_high: float,
) -> dict:
    """Estimate procedure-level ground truth for hard, soft, and adaptive."""
    gt_hard = np.empty(N_gt)
    gt_soft = np.empty(N_gt)
    gt_adaptive = np.empty(N_gt)

    for i in range(N_gt):
        scores = generate_scores(M, qualities, rng, diff_low, diff_high)
        splits = generate_splits(M, R, rho, rng)
        eval_sizes = np.array([len(s[1]) for s in splits])
        weights = eval_sizes / eval_sizes.sum()
        stats = compute_split_stats(scores, splits)

        th, _, winners = run_hard(scores, stats, weights)
        ts, _          = run_soft(scores, stats, weights, tau)

        gt_hard[i] = th
        gt_soft[i] = ts

        pi = winner_instability(winners)
        gt_adaptive[i] = th if pi <= threshold else ts

    return {
        "gt_hard":     float(gt_hard.mean()),
        "gt_soft":     float(gt_soft.mean()),
        "gt_adaptive": float(gt_adaptive.mean()),
    }


# ================================================================
#  Run one gap level  (all three methods, paired on the same data)
# ================================================================
def run_one_gap(
    gap: float,
    cfg: Exp2Config,
    config_seed: np.random.SeedSequence,
) -> dict:
    """For one quality gap, compute coverage for hard / soft / adaptive."""

    qualities = np.array([cfg.q_base + gap, cfg.q_base])  # artifact 0 is better

    gt_stream, sim_stream = config_seed.spawn(2)
    rng_gt  = np.random.default_rng(gt_stream)
    rng_sim = np.random.default_rng(sim_stream)

    # ---- Ground truth ----
    gts = compute_ground_truths(
        cfg.M, qualities, cfg.R, cfg.rho, cfg.tau,
        cfg.instability_threshold, cfg.N_gt, rng_gt,
        cfg.diff_low, cfg.diff_high,
    )

    # ---- Main simulation ----
    cov_hard = np.empty(cfg.N_sim)
    cov_soft = np.empty(cfg.N_sim)
    cov_adaptive = np.empty(cfg.N_sim)
    wid_hard = np.empty(cfg.N_sim)
    wid_soft = np.empty(cfg.N_sim)
    wid_adaptive = np.empty(cfg.N_sim)
    theta_hard_arr = np.empty(cfg.N_sim)   # point estimates
    theta_soft_arr = np.empty(cfg.N_sim)
    pi_wins = np.empty(cfg.N_sim)
    adaptive_chose_soft = np.empty(cfg.N_sim)

    for trial in range(cfg.N_sim):
        scores = generate_scores(
            cfg.M, qualities, rng_sim, cfg.diff_low, cfg.diff_high
        )
        # Same splits shared by all three methods
        splits = generate_splits(cfg.M, cfg.R, cfg.rho, rng_sim)
        eval_sizes = np.array([len(s[1]) for s in splits])
        weights = eval_sizes / eval_sizes.sum()
        stats = compute_split_stats(scores, splits)

        # --- Hard ---
        th, psi_h, winners = run_hard(scores, stats, weights)
        hw_h = bootstrap_ci(psi_h, cfg.M, cfg.n_bootstrap, cfg.alpha, rng_sim)
        cov_hard[trial] = float(gts["gt_hard"] >= th - hw_h and
                                gts["gt_hard"] <= th + hw_h)
        wid_hard[trial] = 2 * hw_h
        theta_hard_arr[trial] = th

        # --- Soft ---
        ts, psi_s = run_soft(scores, stats, weights, cfg.tau)
        hw_s = bootstrap_ci(psi_s, cfg.M, cfg.n_bootstrap, cfg.alpha, rng_sim)
        cov_soft[trial] = float(gts["gt_soft"] >= ts - hw_s and
                                gts["gt_soft"] <= ts + hw_s)
        wid_soft[trial] = 2 * hw_s
        theta_soft_arr[trial] = ts

        # --- Adaptive ---
        pi = winner_instability(winners)
        pi_wins[trial] = pi

        if pi <= cfg.instability_threshold:
            # Use hard
            cov_adaptive[trial] = cov_hard[trial]
            wid_adaptive[trial] = wid_hard[trial]
            adaptive_chose_soft[trial] = 0.0
        else:
            # Use soft
            cov_adaptive[trial] = cov_soft[trial]
            wid_adaptive[trial] = wid_soft[trial]
            adaptive_chose_soft[trial] = 1.0

    se = lambda arr: np.sqrt(arr.mean() * (1 - arr.mean()) / len(arr))

    return {
        "gap":                gap,
        "cov_hard":           float(cov_hard.mean()),
        "cov_soft":           float(cov_soft.mean()),
        "cov_adaptive":       float(cov_adaptive.mean()),
        "se_hard":            float(se(cov_hard)),
        "se_soft":            float(se(cov_soft)),
        "se_adaptive":        float(se(cov_adaptive)),
        "wid_hard":           float(wid_hard.mean()),
        "wid_soft":           float(wid_soft.mean()),
        "wid_adaptive":       float(wid_adaptive.mean()),
        "std_wid_hard":       float(wid_hard.std()),
        "std_wid_soft":       float(wid_soft.std()),
        "std_theta_hard":     float(theta_hard_arr.std()),
        "std_theta_soft":     float(theta_soft_arr.std()),
        "mean_theta_hard":    float(theta_hard_arr.mean()),
        "mean_theta_soft":    float(theta_soft_arr.mean()),
        "avg_pi_win":         float(pi_wins.mean()),
        "pct_adaptive_soft":  float(adaptive_chose_soft.mean()),
        "gt_hard":            gts["gt_hard"],
        "gt_soft":            gts["gt_soft"],
        "gt_adaptive":        gts["gt_adaptive"],
    }


# ================================================================
#  Full sweep
# ================================================================
def run_sweep(cfg: Exp2Config) -> pd.DataFrame:

    n_gaps = len(cfg.gap_list)
    master_ss = np.random.SeedSequence(cfg.seed)
    seeds = master_ss.spawn(n_gaps)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Experiment 2: Near-Tie Hard vs Soft Selection   ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Gaps    : {n_gaps:>4d}  quality gap levels              ║")
    print(f"║  M       : {cfg.M:>5d}  (fixed benchmark size)          ║")
    print(f"║  K       :     {cfg.K}  artifacts (Proposition 3)       ║")
    print(f"║  R       :     {cfg.R}  repeated splits                 ║")
    print(f"║  N_sim   : {cfg.N_sim:>5d}  trials per gap                 ║")
    print(f"║  N_gt    : {cfg.N_gt:>5d}  MC runs for ground truth       ║")
    print(f"╚══════════════════════════════════════════════════╝\n")

    t0 = time.time()

    # Pilot timing
    print(f"  Pilot run (gap={cfg.gap_list[0]}) ...", flush=True)
    tp = time.time()
    _ = run_one_gap(cfg.gap_list[0], cfg, seeds[0])
    dt = time.time() - tp
    est = dt * n_gaps / max(1, os.cpu_count() if cfg.n_jobs == -1 else cfg.n_jobs)
    print(f"  Pilot: {dt:.1f}s  →  estimated total: {est:.0f}s ({est/60:.1f} min)\n",
          flush=True)

    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_one_gap)(gap, cfg, seeds[i])
        for i, gap in enumerate(cfg.gap_list)
    )

    elapsed = time.time() - t0
    print(f"\n  Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return pd.DataFrame(results)


# ================================================================
#  Plotting
# ================================================================
def make_plots(df: pd.DataFrame, cfg: Exp2Config):
    os.makedirs(cfg.figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette = sns.color_palette("colorblind")
    nom = 1 - cfg.alpha
    mc_se = np.sqrt(nom * (1 - nom) / cfg.N_sim)
    df = df.sort_values("gap")

    gaps = df["gap"].values

    # ------------------------------------------------------------------
    #  Plot 1  (HEADLINE):  Coverage vs Gap  — smooth curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gaps, df["cov_hard"], "-", color=palette[3], lw=2.2,
            label="Hard argmax", alpha=0.85)
    ax.fill_between(gaps,
                     df["cov_hard"] - df["se_hard"],
                     df["cov_hard"] + df["se_hard"],
                     alpha=0.12, color=palette[3])

    ax.plot(gaps, df["cov_soft"], "-", color=palette[0], lw=2.2,
            label="Soft (softmax)", alpha=0.85)
    ax.fill_between(gaps,
                     df["cov_soft"] - df["se_soft"],
                     df["cov_soft"] + df["se_soft"],
                     alpha=0.12, color=palette[0])

    ax.plot(gaps, df["cov_adaptive"], "--", color=palette[2], lw=1.8,
            label="Adaptive rule", alpha=0.75)

    ax.axhline(nom, ls="--", color="grey", lw=1, label=f"Nominal {nom:.0%}")
    ax.axhspan(nom - 1.96 * mc_se, nom + 1.96 * mc_se,
               color="grey", alpha=0.06)

    # Mark the worst point for hard selection
    worst_idx = df["cov_hard"].idxmin()
    worst_row = df.loc[worst_idx]
    ax.plot(worst_row["gap"], worst_row["cov_hard"], "v", color=palette[3],
            ms=12, zorder=10)
    ax.annotate(f'Worst: {worst_row["cov_hard"]:.3f}\nat Δ={worst_row["gap"]:.2f}',
                xy=(worst_row["gap"], worst_row["cov_hard"]),
                xytext=(worst_row["gap"] + 0.08, worst_row["cov_hard"] - 0.015),
                fontsize=9, color=palette[3],
                arrowprops=dict(arrowstyle="-", color=palette[3], lw=0.8))

    ax.set_xlabel("Quality gap  Δ = q₁ − q₂")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"CI Coverage vs Quality Gap  (M={cfg.M}, K={cfg.K}, R={cfg.R})\n"
                 f"Fine-grained sweep: Δ from 0.00 to 0.80, step={gaps[1]-gaps[0]:.2f}")
    ax.set_xlim(-0.01, gaps.max() + 0.01)
    ax.set_ylim(max(0.60, df["cov_hard"].min() - 0.03), 1.0)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp2_coverage_vs_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 2:  Winner instability rate vs Gap  — continuous line
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gaps, df["avg_pi_win"], "-", color=palette[4], lw=2)
    ax.fill_between(gaps, 0, df["avg_pi_win"], alpha=0.15, color=palette[4])
    ax.axhline(cfg.instability_threshold, ls="--", color=palette[3], lw=1.2,
               label=f"Adaptive threshold = {cfg.instability_threshold}")
    ax.set_xlabel("Quality gap  Δ")
    ax.set_ylabel("Average winner instability  π̂_win")
    ax.set_title(f"Winner Instability vs Quality Gap  (M={cfg.M}, R={cfg.R})")
    ax.set_xlim(-0.01, gaps.max() + 0.01)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp2_instability_vs_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 3:  CI Width vs Gap  — three method curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(gaps, df["wid_hard"], "-", color=palette[3], lw=2, label="Hard argmax")
    ax.plot(gaps, df["wid_soft"], "-", color=palette[0], lw=2, label="Soft (softmax)")
    ax.plot(gaps, df["wid_adaptive"], "--", color=palette[2], lw=1.5, label="Adaptive rule")
    ax.set_xlabel("Quality gap  Δ")
    ax.set_ylabel("Average CI width")
    ax.set_title(f"CI Width vs Quality Gap  (M={cfg.M}, K={cfg.K}, R={cfg.R})")
    ax.set_xlim(-0.01, gaps.max() + 0.01)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp2_width_vs_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 4:  Combined diagnostic — coverage + instability, dual y-axis
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Coverage on left axis
    ln1 = ax1.plot(gaps, df["cov_hard"], "-", color=palette[3], lw=2.2,
                    label="Hard coverage")
    ln2 = ax1.plot(gaps, df["cov_soft"], "-", color=palette[0], lw=2.2,
                    label="Soft coverage")
    ax1.axhline(nom, ls="--", color="grey", lw=1)
    ax1.set_ylabel("Empirical coverage", color="black")
    ax1.set_ylim(max(0.60, df["cov_hard"].min() - 0.03), 1.0)

    # Instability on right axis
    ln3 = ax2.fill_between(gaps, 0, df["avg_pi_win"], alpha=0.18,
                            color=palette[4], label="π̂_win")
    ax2.plot(gaps, df["avg_pi_win"], "-", color=palette[4], lw=1, alpha=0.5)
    ax2.set_ylabel("Winner instability  π̂_win", color=palette[4])
    ax2.set_ylim(0, 0.55)

    ax1.set_xlabel("Quality gap  Δ")
    ax1.set_xlim(-0.01, gaps.max() + 0.01)
    ax1.set_title(f"Coverage & Instability vs Gap  (M={cfg.M}, K={cfg.K}, R={cfg.R})")

    # Merge legends
    from matplotlib.patches import Patch
    handles = [ln1[0], ln2[0], Patch(facecolor=palette[4], alpha=0.3, label="π̂_win")]
    ax1.legend(handles=handles,
               labels=["Hard coverage", "Soft coverage", "π̂_win"],
               loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp2_combined_diagnostic.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 5:  Std of point estimates (theta) — hard vs soft
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    hard_true = df["std_theta_hard"].values
    soft_true = df["std_theta_soft"].values
    hard_boot = (df["wid_hard"] / (2 * 1.96)).values
    soft_boot = (df["wid_soft"] / (2 * 1.96)).values

    # Shade the gap = missed variance for hard selection
    ax.fill_between(gaps, hard_boot, hard_true,
                     alpha=0.25, color=palette[3], label="Hard: MISSED variance")

    ax.plot(gaps, hard_true, "-", color=palette[3], lw=2.5,
            label="Hard argmax  TRUE std(θ̂)")
    ax.plot(gaps, hard_boot, "--", color=palette[3], lw=1.8,
            alpha=0.7, label="Hard: bootstrap thinks std is this")

    ax.plot(gaps, soft_true, "-", color=palette[0], lw=2.5,
            label="Soft (softmax)  TRUE std(θ̂)")
    ax.plot(gaps, soft_boot, "--", color=palette[0], lw=1.8,
            alpha=0.7, label="Soft: bootstrap thinks std is this")

    ax.set_xlabel("Quality gap  Δ", fontsize=13)
    ax.set_ylabel("Standard deviation of point estimate", fontsize=13)
    ax.set_title(f"True Variability vs Bootstrap Estimate  (M={cfg.M}, K={cfg.K}, R={cfg.R})\n"
                 f"Shaded area = variance bootstrap misses → undercoverage",
                 fontsize=13)
    ax.set_xlim(-0.01, gaps.max() + 0.01)
    ax.set_ylim(0, hard_true.max() * 1.25)  # start from 0 for full scale
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp2_std_theta.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 6:  Std of CI widths — hard vs soft
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(gaps, df["std_wid_hard"], "-", color=palette[3], lw=2, label="Hard argmax")
    ax.plot(gaps, df["std_wid_soft"], "-", color=palette[0], lw=2, label="Soft (softmax)")
    ax.set_xlabel("Quality gap  Δ")
    ax.set_ylabel("Std of CI width across trials")
    ax.set_title(f"CI Width Stability  (M={cfg.M}, K={cfg.K}, R={cfg.R})")
    ax.set_xlim(-0.01, gaps.max() + 0.01)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp2_std_width.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
#  Console summary
# ================================================================
def print_summary(df: pd.DataFrame, cfg: Exp2Config):
    nom = 1 - cfg.alpha
    print("\n" + "=" * 110)
    print("  RESULTS SUMMARY — Experiment 2: Near-Tie Hard vs Soft Selection")
    print("=" * 110)
    print(f"  {'Gap':>6s}  │ {'Cov_H':>7s}  {'Cov_S':>7s}  │ "
          f"{'Wid_H':>7s} {'Wid_S':>7s}  │ "
          f"{'Std_θ_H':>8s} {'Std_θ_S':>8s}  │ "
          f"{'Std_W_H':>8s} {'Std_W_S':>8s}  │ {'π̂_win':>6s}")
    print("  " + "─" * 104)

    for _, r in df.sort_values("gap").iterrows():
        flag_h = " ⚠" if r["cov_hard"] < nom - 2.5 * r["se_hard"] else "  "
        print(f"  {r['gap']:6.3f}  │ "
              f"{r['cov_hard']:5.3f}{flag_h} "
              f"{r['cov_soft']:7.3f}  │ "
              f"{r['wid_hard']:7.5f} {r['wid_soft']:7.5f}  │ "
              f"{r['std_theta_hard']:8.5f} {r['std_theta_soft']:8.5f}  │ "
              f"{r['std_wid_hard']:8.5f} {r['std_wid_soft']:8.5f}  │ "
              f"{r['avg_pi_win']:6.3f}")

    print("=" * 110)

    # Headline finding
    tie_row = df[df["gap"] == 0.0]
    worst_row = df.loc[df["cov_hard"].idxmin()]
    big_row = df[df["gap"] == df["gap"].max()]

    if len(tie_row) > 0:
        ch0 = tie_row.iloc[0]["cov_hard"]
        cs0 = tie_row.iloc[0]["cov_soft"]
        print(f"\n  At exact tie (Δ=0):   Hard={ch0:.4f}  Soft={cs0:.4f}")
        print(f"    Both are fine here — artifacts are identical, so misselection is harmless.")

    print(f"\n  Worst hard coverage:  Δ={worst_row['gap']:.3f}  "
          f"Hard={worst_row['cov_hard']:.4f}  Soft={worst_row['cov_soft']:.4f}")
    drop = nom - worst_row["cov_hard"]
    if drop > 0.02:
        print(f"    → Hard selection loses {drop:.1%} coverage in the 'near-tie' regime  ✓ Prop 3")
        print(f"    Mechanism: bootstrap misses selection-jump variance when winner changes")
    else:
        print(f"    → Coverage drop is modest ({drop:.1%}).  Full run with N_sim=2000 may sharpen.")

    if len(big_row) > 0:
        ch_big = big_row.iloc[0]["cov_hard"]
        pi_big = big_row.iloc[0]["avg_pi_win"]
        print(f"\n  At large gap (Δ={big_row.iloc[0]['gap']:.2f}):  "
              f"Hard={ch_big:.4f}  π̂_win={pi_big:.3f}")
        if ch_big > nom - 0.02:
            print(f"    → Hard recovers when winner is stable  ✓")
        else:
            print(f"    → Still some undercoverage — gap may not be large enough for full recovery")
    print()


# ================================================================
#  Main
# ================================================================
def main():
    quick = "--quick" in sys.argv
    if quick:
        print(">>> Running in QUICK mode <<<\n")
        cfg = quick_config()
    else:
        cfg = Exp2Config()

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    df = run_sweep(cfg)

    csv_path = os.path.join(cfg.results_dir, "exp2_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Results saved to {csv_path}")

    print_summary(df, cfg)

    print("  Generating figures ...")
    make_plots(df, cfg)
    print("\n  Done!  Check the figures/ directory.\n")


if __name__ == "__main__":
    main()
