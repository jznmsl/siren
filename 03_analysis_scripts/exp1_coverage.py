#!/usr/bin/env python
"""
Experiment 1: Bootstrap Coverage Validation
============================================
Validates finite-sample coverage of multiplier bootstrap CIs for the
repeated-split estimator with soft (softmax) selection.

Paper reference:
  - Theorem 1 (selection-preserving transfer theorem)
  - Section 4.6 (multiplier bootstrap)
  - Study A, Layer A (Section 5.3)

Usage:
  python exp1_coverage.py              # full run  (~10-25 min)
  python exp1_coverage.py --quick      # quick test (~1-3 min)

Required packages:
  pip install numpy scipy matplotlib seaborn pandas joblib tqdm
"""

import sys
import os
import time
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy.special import expit  # sigmoid function
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# Try tqdm for progress bars; fall back to a no-op wrapper
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw):
        total = kw.get("total", None)
        desc  = kw.get("desc", "")
        for i, x in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"  {desc} {i}/{total} ...", flush=True)
            yield x


# ================================================================
#  Configuration
# ================================================================
@dataclass
class ExpConfig:
    """All parameters for Experiment 1."""

    # --- DGP ---
    q_base: float = 0.0       # worst artifact quality
    delta_q: float = 0.3      # total quality range (best - worst ≈ 7 pct acc)
    diff_low: float = -2.0    # item difficulty lower bound
    diff_high: float = 2.0    # item difficulty upper bound

    # --- Sweep axes ---
    M_list: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000])
    K_list: List[int] = field(default_factory=lambda: [2, 5, 10, 20])
    R_list: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # --- Fixed pipeline parameters ---
    rho: float = 0.5          # dev / eval split fraction
    tau: float = 0.1          # softmax temperature
    n_bootstrap: int = 1000   # multiplier bootstrap draws
    alpha: float = 0.05       # 1 - confidence level

    # --- Simulation scale ---
    N_sim: int = 2000         # trials per configuration
    N_gt: int = 10_000        # Monte-Carlo runs for ground truth

    # --- Compute ---
    n_jobs: int = -1          # joblib parallelism (-1 = all cores)
    seed: int = 42

    # --- Output ---
    results_dir: str = "results"
    figures_dir: str = "figures"


def quick_config() -> ExpConfig:
    """Reduced config for fast debugging (~1-3 min)."""
    return ExpConfig(
        M_list=[100, 500, 2000],
        K_list=[2, 10],
        R_list=[1, 5],
        N_sim=500,
        N_gt=3_000,
    )


# ================================================================
#  DGP  –  Data Generating Process
# ================================================================
def make_qualities(K: int, q_base: float, delta_q: float) -> np.ndarray:
    """
    Create K artifact qualities evenly spaced from q_base to q_base+delta_q.
    Returns shape [K].
    """
    return np.linspace(q_base, q_base + delta_q, K)


def generate_scores(
    M: int,
    qualities: np.ndarray,
    rng: np.random.Generator,
    diff_low: float = -2.0,
    diff_high: float = 2.0,
) -> np.ndarray:
    """
    Sample M items and return a Bernoulli score matrix.

    scores[i, k] ~ Bernoulli( sigmoid(q_k - delta_i) )

    Parameters
    ----------
    M          : number of items
    qualities  : artifact quality vector, shape [K]
    rng        : numpy random Generator
    diff_low/high : Uniform range for item difficulties

    Returns
    -------
    scores : ndarray of shape [M, K], entries in {0, 1}
    """
    K = len(qualities)
    difficulties = rng.uniform(diff_low, diff_high, size=M)
    # logits[i, k] = q_k - delta_i
    logits = qualities[np.newaxis, :] - difficulties[:, np.newaxis]  # [M, K]
    probs = expit(logits)                                           # [M, K]
    scores = rng.binomial(1, probs).astype(np.float64)              # [M, K]
    return scores


# ================================================================
#  Selection rule
# ================================================================
def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    """
    Numerically stable softmax with temperature tau.
    Input  x : 1-D array [K].
    Output q : 1-D array [K], sums to 1.
    """
    z = x / tau
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ================================================================
#  Repeated-split pipeline
# ================================================================
def pipeline_fast(
    scores: np.ndarray,
    R: int,
    rho: float,
    tau: float,
    rng: np.random.Generator,
) -> float:
    """
    Lightweight pipeline that returns only theta_tilde.
    Used for ground-truth Monte-Carlo estimation where we do NOT need
    split details or influence contributions.
    """
    M, K = scores.shape
    n_dev = int(M * rho)

    Y_hats = np.empty(R)
    eval_sizes = np.empty(R, dtype=np.int64)

    for r in range(R):
        perm = rng.permutation(M)
        dev_idx = perm[:n_dev]
        eval_idx = perm[n_dev:]
        eval_sizes[r] = len(eval_idx)

        S_hat = scores[dev_idx].mean(axis=0)   # [K]
        T_hat = scores[eval_idx].mean(axis=0)  # [K]
        q_hat = softmax(S_hat, tau)            # [K]
        Y_hats[r] = q_hat @ T_hat

    weights = eval_sizes / eval_sizes.sum()
    return float(weights @ Y_hats)


def pipeline_full(
    scores: np.ndarray,
    R: int,
    rho: float,
    tau: float,
    rng: np.random.Generator,
) -> Tuple[float, list, np.ndarray]:
    """
    Full pipeline that also stores per-split info needed for the
    influence-function computation.

    Returns
    -------
    theta_tilde : float   – repeated-split point estimate
    split_info  : list of tuples (dev_idx, eval_idx, S_hat, T_hat, q_hat)
    weights     : ndarray [R] – normalised eval-size weights
    """
    M, K = scores.shape
    n_dev = int(M * rho)

    split_info = []
    Y_hats = np.empty(R)
    eval_sizes = np.empty(R, dtype=np.int64)

    for r in range(R):
        perm = rng.permutation(M)
        dev_idx = perm[:n_dev]
        eval_idx = perm[n_dev:]
        n_eval = len(eval_idx)
        eval_sizes[r] = n_eval

        S_hat = scores[dev_idx].mean(axis=0)   # [K]
        T_hat = scores[eval_idx].mean(axis=0)  # [K]
        q_hat = softmax(S_hat, tau)            # [K]
        Y_hats[r] = q_hat @ T_hat

        split_info.append((dev_idx, eval_idx, S_hat, T_hat, q_hat))

    weights = eval_sizes / eval_sizes.sum()
    theta_tilde = float(weights @ Y_hats)
    return theta_tilde, split_info, weights


# ================================================================
#  Influence function  (Theorem 1, item-level contributions)
# ================================================================
def compute_influence(
    scores: np.ndarray,
    split_info: list,
    weights: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Compute the item-level influence vector  psi_hat[M].

    From Theorem 1, the first-order expansion of the repeated-split
    estimator is:

        sqrt(M) * (theta_tilde - theta_split) ≈ (1/sqrt(M)) sum_i psi_i

    where each item's contribution has two parts:

        psi_i  =  sum_r  omega_r * [ eval_contrib_r(i) + dev_contrib_r(i) ]

    Eval contribution (item i in E_r):
        omega_r * (M / |E_r|) * q_hat_r^T (Z_i - T_hat_r)

        This is the ordinary held-out fluctuation that would remain
        even if the selector were non-adaptive.

    Dev contribution (item i in D_r):
        omega_r * (M / |D_r|) * c_r^T (Z_i - S_hat_r)

        where  c_{r,j} = q_hat_{r,j} * (T_hat_{r,j} - q_hat_r · T_hat_r) / tau

        This captures how item i's dev score perturbs the selection rule
        and thereby indirectly affects the held-out estimate.
        (Derived from the softmax Jacobian.)

    Returns
    -------
    psi : ndarray [M] – item-level influence contributions (centered, sums ≈ 0)
    """
    M, K = scores.shape
    R = len(split_info)
    psi = np.zeros(M)

    for r in range(R):
        dev_idx, eval_idx, S_hat, T_hat, q_hat = split_info[r]
        omega_r = weights[r]
        n_dev = len(dev_idx)
        n_eval = len(eval_idx)

        # ------ Eval contribution (items in E_r) ------
        # (M / n_eval) * q_hat^T (scores[i, :] - T_hat)   for i in E_r
        eval_resid = scores[eval_idx] - T_hat          # [n_eval, K]
        eval_contrib = eval_resid @ q_hat               # [n_eval]
        eval_contrib *= (M / n_eval)
        psi[eval_idx] += omega_r * eval_contrib

        # ------ Dev contribution (items in D_r) ------
        # c_j = q_j * (T_j - q^T T) / tau
        # (M / n_dev) * c^T (scores[i, :] - S_hat)   for i in D_r
        q_dot_T = q_hat @ T_hat                         # scalar
        c = q_hat * (T_hat - q_dot_T) / tau             # [K]
        dev_resid = scores[dev_idx] - S_hat              # [n_dev, K]
        dev_contrib = dev_resid @ c                      # [n_dev]
        dev_contrib *= (M / n_dev)
        psi[dev_idx] += omega_r * dev_contrib

    return psi


# ================================================================
#  Multiplier bootstrap  →  confidence interval
# ================================================================
def bootstrap_ci(
    psi: np.ndarray,
    M: int,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """
    Multiplier bootstrap for a symmetric CI.

    The bootstrap process:
        G*_b = (1/sqrt(M)) * sum_i  xi_{b,i} * psi_i
    approximates the distribution of  sqrt(M)*(theta_tilde - theta_split).

    We return the half-width:
        half_width = c_alpha / sqrt(M)
    where  c_alpha = quantile( |G*|, 1-alpha ).

    The 95% CI is then  [ theta_tilde - hw, theta_tilde + hw ].

    Vectorised: xi is [n_bootstrap, M], one matmul gives all G* values.
    """
    xi = rng.standard_normal((n_bootstrap, M))     # [B, M]
    G_star = xi @ psi / np.sqrt(M)                 # [B]
    c_alpha = np.quantile(np.abs(G_star), 1 - alpha)
    half_width = c_alpha / np.sqrt(M)
    return float(half_width)


# ================================================================
#  Ground-truth estimation  (Monte-Carlo)
# ================================================================
def compute_ground_truth(
    M: int,
    qualities: np.ndarray,
    R: int,
    rho: float,
    tau: float,
    N_gt: int,
    rng: np.random.Generator,
    diff_low: float,
    diff_high: float,
) -> Tuple[float, float]:
    """
    Estimate the true procedure performance via Monte-Carlo.

    theta_proc = E[ theta_tilde ]

    where the expectation is over random item draws, random splits,
    and Bernoulli score noise.

    Returns (mean, std) of the Monte-Carlo estimates.
    """
    estimates = np.empty(N_gt)
    for i in range(N_gt):
        scores = generate_scores(M, qualities, rng, diff_low, diff_high)
        estimates[i] = pipeline_fast(scores, R, rho, tau, rng)
    return float(estimates.mean()), float(estimates.std())


# ================================================================
#  Run ONE configuration  (M, K, R)
# ================================================================
def run_one_config(
    M: int,
    K: int,
    R: int,
    cfg: ExpConfig,
    config_seed: np.random.SeedSequence,
) -> dict:
    """
    For a single (M, K, R) setting:
      1. compute ground truth via Monte-Carlo
      2. run N_sim trials, each producing a bootstrap CI
      3. measure coverage and width

    Returns a dict with all results.
    """
    # Deterministic artifact qualities (shared across all trials)
    qualities = make_qualities(K, cfg.q_base, cfg.delta_q)

    # Two independent RNG streams: one for GT, one for main sim
    gt_stream, sim_stream = config_seed.spawn(2)
    rng_gt = np.random.default_rng(gt_stream)
    rng_sim = np.random.default_rng(sim_stream)

    # ---- Step 1: ground truth ----
    gt_mean, gt_std = compute_ground_truth(
        M, qualities, R, cfg.rho, cfg.tau,
        cfg.N_gt, rng_gt, cfg.diff_low, cfg.diff_high,
    )

    # ---- Step 2: main simulation ----
    covered = np.empty(cfg.N_sim)
    widths = np.empty(cfg.N_sim)

    for trial in range(cfg.N_sim):
        scores = generate_scores(
            M, qualities, rng_sim, cfg.diff_low, cfg.diff_high
        )
        theta_tilde, split_info, weights = pipeline_full(
            scores, R, cfg.rho, cfg.tau, rng_sim
        )
        psi = compute_influence(scores, split_info, weights, cfg.tau)

        # --- Sanity check (occasionally): influence sums to ≈ 0 ---
        # (Disabled in production for speed; uncomment to debug)
        # assert abs(psi.sum()) < 1e-10, f"psi sum = {psi.sum()}"

        hw = bootstrap_ci(psi, M, cfg.n_bootstrap, cfg.alpha, rng_sim)

        ci_lo = theta_tilde - hw
        ci_hi = theta_tilde + hw
        covered[trial] = float(ci_lo <= gt_mean <= ci_hi)
        widths[trial] = 2.0 * hw

    coverage_rate = covered.mean()
    avg_width = widths.mean()

    # Standard error of the measured coverage (binomial)
    se_coverage = np.sqrt(coverage_rate * (1 - coverage_rate) / cfg.N_sim)

    return {
        "M": M,
        "K": K,
        "R": R,
        "coverage": coverage_rate,
        "se_coverage": se_coverage,
        "avg_width": avg_width,
        "std_width": widths.std(),
        "gt_mean": gt_mean,
        "gt_std": gt_std,
    }


# ================================================================
#  Full sweep  (parallel across configurations)
# ================================================================
def run_sweep(cfg: ExpConfig) -> pd.DataFrame:
    """Run all (M, K, R) configurations in parallel."""

    configs = list(itertools.product(cfg.M_list, cfg.K_list, cfg.R_list))
    n_configs = len(configs)

    # Spawn independent seed sequences for each configuration
    master_ss = np.random.SeedSequence(cfg.seed)
    config_seeds = master_ss.spawn(n_configs)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Experiment 1: Bootstrap Coverage Validation     ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Configs : {n_configs:>4d}  (M×K×R combinations)        ║")
    print(f"║  N_sim   : {cfg.N_sim:>5d}  trials per config           ║")
    print(f"║  N_gt    : {cfg.N_gt:>5d}  MC runs for ground truth    ║")
    print(f"║  N_boot  : {cfg.n_bootstrap:>5d}  bootstrap draws            ║")
    print(f"║  Jobs    : {cfg.n_jobs:>5d}  parallel workers            ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    t0 = time.time()

    # --- Run one config first to estimate total time ---
    M0, K0, R0 = configs[0]
    print(f"  Timing pilot run (M={M0}, K={K0}, R={R0}) ...", flush=True)
    t_pilot = time.time()
    _ = run_one_config(M0, K0, R0, cfg, config_seeds[0])
    dt_pilot = time.time() - t_pilot
    est_total = dt_pilot * n_configs / max(1, os.cpu_count() if cfg.n_jobs == -1 else cfg.n_jobs)
    print(f"  Pilot: {dt_pilot:.1f}s  →  estimated total: {est_total:.0f}s "
          f"({est_total/60:.1f} min)\n", flush=True)

    # --- Parallel execution ---
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(run_one_config)(M, K, R, cfg, config_seeds[i])
        for i, (M, K, R) in enumerate(configs)
    )

    elapsed = time.time() - t0
    print(f"\n  Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    df = pd.DataFrame(results)
    return df


# ================================================================
#  Plotting
# ================================================================
def make_plots(df: pd.DataFrame, cfg: ExpConfig):
    """Generate the four publication figures."""

    os.makedirs(cfg.figures_dir, exist_ok=True)

    # Global style
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette = sns.color_palette("colorblind")

    # Nominal coverage line + MC noise band (± 1.96 * se at p=0.95)
    nom = 1 - cfg.alpha
    mc_se = np.sqrt(nom * (1 - nom) / cfg.N_sim)  # binomial SE of coverage estimate

    # Fix R at the recommended default for plots 1-3
    R_default = 5
    # Fall back to the largest available R if 5 not in data
    if R_default not in df["R"].values:
        R_default = df["R"].max()
    df_r = df[df["R"] == R_default].copy()

    # ------------------------------------------------------------------
    #  Plot 1: Coverage vs M  (lines by K)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, K in enumerate(sorted(df_r["K"].unique())):
        sub = df_r[df_r["K"] == K].sort_values("M")
        ax.plot(sub["M"], sub["coverage"], "o-", color=palette[i],
                label=f"K={K}", linewidth=2, markersize=7)
        # ± 1 SE error bars
        ax.fill_between(sub["M"],
                         sub["coverage"] - sub["se_coverage"],
                         sub["coverage"] + sub["se_coverage"],
                         alpha=0.15, color=palette[i])
    ax.axhline(nom, ls="--", color="grey", linewidth=1.2, label=f"Nominal {nom:.0%}")
    ax.axhspan(nom - 1.96 * mc_se, nom + 1.96 * mc_se,
               color="grey", alpha=0.10, label="MC noise band")
    ax.set_xlabel("Benchmark size  M")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Bootstrap CI Coverage vs Benchmark Size  (R={R_default}, soft selection)")
    ax.set_ylim(max(0.70, df_r["coverage"].min() - 0.03), 1.0)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp1_coverage_vs_M.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 2: Width vs M  (lines by K)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, K in enumerate(sorted(df_r["K"].unique())):
        sub = df_r[df_r["K"] == K].sort_values("M")
        ax.plot(sub["M"], sub["avg_width"], "s-", color=palette[i],
                label=f"K={K}", linewidth=2, markersize=7)
    ax.set_xlabel("Benchmark size  M")
    ax.set_ylabel("Average CI width")
    ax.set_title(f"Average CI Width vs Benchmark Size  (R={R_default}, soft selection)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp1_width_vs_M.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 3: Coverage heatmap  (rows=M, cols=K)
    # ------------------------------------------------------------------
    pivot = df_r.pivot_table(index="M", columns="K", values="coverage")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.80, vmax=1.0, linewidths=0.5,
        cbar_kws={"label": "Coverage"}, ax=ax,
    )
    ax.set_title(f"Coverage Heatmap  (R={R_default})")
    ax.set_ylabel("M  (benchmark size)")
    ax.set_xlabel("K  (shortlist size)")
    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp1_coverage_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    #  Plot 4: Impact of R  (fixed M, K)
    # ------------------------------------------------------------------
    # Pick the M, K pair with the most R values available
    best_pair = None
    best_count = 0
    for (m, k), grp in df.groupby(["M", "K"]):
        if len(grp) > best_count:
            best_count = len(grp)
            best_pair = (m, k)
    # Prefer M=500, K=10 if available with enough R values
    candidate = df[(df["M"] == 500) & (df["K"] == 10)]
    if len(candidate) >= 3:
        best_pair = (500, 10)

    M_fix, K_fix = best_pair
    df_rswp = df[(df["M"] == M_fix) & (df["K"] == K_fix)].sort_values("R")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: coverage vs R
    ax = axes[0]
    ax.bar(df_rswp["R"].astype(str), df_rswp["coverage"], color=palette[0],
           edgecolor="white", width=0.6)
    ax.axhline(nom, ls="--", color="grey", linewidth=1.2)
    ax.axhspan(nom - 1.96 * mc_se, nom + 1.96 * mc_se, color="grey", alpha=0.10)
    ax.set_xlabel("Number of repeated splits  R")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Coverage vs R  (M={M_fix}, K={K_fix})")
    ax.set_ylim(max(0.70, df_rswp["coverage"].min() - 0.05), 1.0)

    # Right: width vs R
    ax = axes[1]
    ax.bar(df_rswp["R"].astype(str), df_rswp["avg_width"], color=palette[1],
           edgecolor="white", width=0.6)
    ax.set_xlabel("Number of repeated splits  R")
    ax.set_ylabel("Average CI width")
    ax.set_title(f"CI Width vs R  (M={M_fix}, K={K_fix})")

    fig.tight_layout()
    path = os.path.join(cfg.figures_dir, "exp1_R_impact.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def print_summary_table(df: pd.DataFrame, cfg: ExpConfig):
    """Print a formatted summary to console."""

    nom = 1 - cfg.alpha
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'M':>5s}  {'K':>3s}  {'R':>3s}  │ {'Coverage':>9s}  {'±SE':>7s}  "
          f"{'Width':>8s}  {'GT mean':>8s}  {'Status':>8s}")
    print("  " + "─" * 66)

    for _, row in df.sort_values(["R", "M", "K"]).iterrows():
        status = "  ✓ OK" if abs(row["coverage"] - nom) < 2.5 * row["se_coverage"] else " ⚠ CHECK"
        print(f"  {row['M']:5.0f}  {row['K']:3.0f}  {row['R']:3.0f}  │ "
              f"{row['coverage']:9.4f}  ±{row['se_coverage']:.4f}  "
              f"{row['avg_width']:8.5f}  {row['gt_mean']:8.5f}  {status}")

    print("=" * 72)

    # Go / No-Go check
    R5 = df[df["R"] == (5 if 5 in df["R"].values else df["R"].max())]
    if len(R5) > 0:
        M500 = R5[R5["M"] >= 500]
        if len(M500) > 0:
            worst_cov = M500["coverage"].min()
            print(f"\n  Go/No-Go check (M≥500, R={R5['R'].iloc[0]}):")
            print(f"    Worst coverage among M≥500 configs: {worst_cov:.4f}")
            if worst_cov >= 0.92:
                print(f"    ✓ GO  — method looks reliable at practical benchmark sizes")
            elif worst_cov >= 0.88:
                print(f"    ~ MARGINAL — coverage slightly low, investigate bias / bootstrap")
            else:
                print(f"    ✗ NO-GO — coverage too low, check implementation before proceeding")
    print()


# ================================================================
#  Main entry point
# ================================================================
def main():
    quick = "--quick" in sys.argv

    if quick:
        print(">>> Running in QUICK mode (reduced scale) <<<\n")
        cfg = quick_config()
    else:
        cfg = ExpConfig()

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    # ---- Run ----
    df = run_sweep(cfg)

    # ---- Save raw results ----
    csv_path = os.path.join(cfg.results_dir, "exp1_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Results saved to {csv_path}")

    # ---- Summary ----
    print_summary_table(df, cfg)

    # ---- Plots ----
    print("  Generating figures ...")
    make_plots(df, cfg)

    print("\n  Done!  Check the figures/ directory for plots.\n")


if __name__ == "__main__":
    main()
