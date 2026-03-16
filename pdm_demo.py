"""
Pioneer Detection Method (PDM) - Bayesian Learning Benchmark
=============================================================

This script implements the simulation framework from the paper to benchmark
all pioneer detection methods against a known (but supervisor-unobservable)
true tail parameter alpha_t.

Model setup (following the paper)
---------------------------------
- Losses follow a Pareto distribution with tail parameter alpha_t.
- At t=0, alpha undergoes a structural break: it jumps from alpha_minus
  (well-learned) to alpha_plus (unknown to experts).
- m non-cooperative Bayesian experts each draw independent Pareto samples
  of size n_obs per period and update their posterior estimate of alpha.
- Expert 1 ("the pioneer") draws a larger sample per period, so it learns
  faster about the new alpha_plus.
- The supervisor S never observes alpha_t. S must pool expert estimates
  using one of the methods in pdm.py.
- We compare each method's pooled estimate against the true alpha_plus
  using cumulative Root Mean Square Error (RMSE).

This code corresponds to the approach introduced in:
    Vansteenberghe, Eric (2025),
    "Insurance Supervision under Climate Change: A Pioneer Detection Method,"
    The Geneva Papers on Risk and Insurance - Issues and Practice,
    https://doi.org/10.1057/s41288-025-00367-y
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pdm import (
    compute_pioneer_weights_angles,
    compute_pioneer_weights_distance,
    compute_granger_weights,
    compute_lagged_correlation_weights,
    compute_multivariate_regression_weights,
    compute_transfer_entropy_weights,
    compute_linear_pooling_weights,
    compute_median_pooling,
    pooled_forecast,
)


# ======================================================================
# Simulation: Bayesian experts learning a Pareto tail parameter
# ======================================================================

def simulate_bayesian_experts(
    alpha_minus: float = 3.0,
    alpha_plus: float = 1.5,
    n_experts: int = 3,
    T: int = 10,
    n_obs_base: int = 5,
    n_obs_pioneer: int = 7,
    seed: int = 4,
) -> tuple[pd.DataFrame, float]:
    """
    Simulate Bayesian experts learning a Pareto tail parameter after a
    structural break.

    The Pareto distribution has pdf f(x) = alpha / x^{alpha+1} for x >= 1
    with tail parameter alpha > 0.

    Each expert i maintains a Gamma posterior for alpha (conjugate prior for
    Pareto with known threshold x_min=1):
        alpha | data ~ Gamma(a_i, b_i)
    where a_i = a_prior + n_i and b_i = b_prior + sum(log(x_j)).

    The posterior mean is a_i / b_i.

    Expert 1 (the pioneer) receives n_obs_pioneer observations per period;
    all other experts receive n_obs_base.  This models the paper's assumption
    that some experts are exposed to more extreme events (larger private
    samples) and thus learn faster.

    Parameters
    ----------
    alpha_minus : float
        True tail parameter before the structural break (well-learned).
    alpha_plus : float
        True tail parameter after the break (to be learned).
    n_experts : int
        Number of experts (default 3).
    T : int
        Number of post-break time periods (default 30).
    n_obs_base : int
        Observations per period for non-pioneer experts (default 5).
    n_obs_pioneer : int
        Observations per period for the pioneer expert (default 20).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    forecasts : pd.DataFrame
        (T x n_experts) posterior means (expert estimates of alpha).
    alpha_true : float
        The true post-break tail parameter alpha_plus.
    """
    rng = np.random.default_rng(seed)

    # Gamma prior calibrated to the pre-break parameter (well-learned)
    # Prior: Gamma(a_prior, b_prior) with mean = alpha_minus
    a_prior = 50.0           # strong prior (experts confident in alpha_minus)
    b_prior = a_prior / alpha_minus

    n_obs = [n_obs_pioneer] + [n_obs_base] * (n_experts - 1)

    estimates = np.zeros((T, n_experts))

    for i in range(n_experts):
        a_post = a_prior
        b_post = b_prior

        for t in range(T):
            # Draw Pareto(alpha_plus) observations: x ~ Pareto(alpha_plus, x_min=1)
            samples = rng.pareto(alpha_plus, size=n_obs[i]) + 1.0

            # Bayesian update: Gamma conjugate for Pareto likelihood
            a_post += n_obs[i]
            b_post += np.sum(np.log(samples))

            # Posterior mean as point estimate
            estimates[t, i] = a_post / b_post

    cols = [f"exp{i+1}" for i in range(n_experts)]
    forecasts = pd.DataFrame(estimates, columns=cols)
    return forecasts, alpha_plus


# ======================================================================
# Run the simulation
# ======================================================================

print("=" * 70)
print("Bayesian Learning Benchmark: PDM and Alternative Methods")
print("=" * 70)

forecasts, alpha_true = simulate_bayesian_experts(
    alpha_minus=3.0,
    alpha_plus=1.5,
    n_experts=3,
    T=10,
    n_obs_base=5,
    n_obs_pioneer=7,
    seed=4,
)

print(f"\nTrue post-break alpha: {alpha_true}")
print(f"Expert 1 (pioneer): 20 obs/period  |  Experts 2-3: 5 obs/period")
print(f"\nExpert estimates (first 10 periods):")
print(forecasts.head(10).to_string(float_format="%.4f"))


# ======================================================================
# Apply all methods
# ======================================================================

results = {}

# --- PDM variants ---
w = compute_pioneer_weights_angles(forecasts)
results["PDM (angles)"] = pooled_forecast(forecasts, w)

w = compute_pioneer_weights_distance(forecasts)
results["PDM (distances)"] = pooled_forecast(forecasts, w)

# --- Alternative methods ---
w = compute_granger_weights(forecasts, maxlag=1)
results["Granger Causality"] = pooled_forecast(forecasts, w)

w = compute_lagged_correlation_weights(forecasts, lag=1)
results["Lagged Correlation"] = pooled_forecast(forecasts, w)

w = compute_multivariate_regression_weights(forecasts, lag=1)
results["Multivar. Regression"] = pooled_forecast(forecasts, w)

w = compute_transfer_entropy_weights(forecasts, n_bins=3, lag=1)
results["Transfer Entropy"] = pooled_forecast(forecasts, w)

# --- Traditional benchmarks ---
w = compute_linear_pooling_weights(forecasts)
results["Linear Pooling"] = pooled_forecast(forecasts, w)

results["Median Pooling"] = compute_median_pooling(forecasts)


# ======================================================================
# Compute RMSE relative to the true alpha_plus
# ======================================================================

print("\n" + "=" * 70)
print("RMSE relative to true alpha (lower is better)")
print("=" * 70)

rmse_table = {}
for name, pooled in results.items():
    se = (pooled - alpha_true) ** 2
    rmse = np.sqrt(se.mean())
    rmse_table[name] = rmse

# Normalise: PDM (angles) = 1.00
pdm_rmse = rmse_table["PDM (angles)"]
print(f"\n{'Method':30s}  {'RMSE':>8s}  {'Relative':>8s}")
print("-" * 50)
for name, rmse in sorted(rmse_table.items(), key=lambda x: x[1]):
    relative = rmse / pdm_rmse if pdm_rmse > 0 else float("nan")
    print(f"{name:30s}  {rmse:8.4f}  {relative:8.2f}")


# ======================================================================
# Compute cumulative RMSE over time (learning curve)
# ======================================================================

cumrmse = {}
for name, pooled in results.items():
    se = (pooled - alpha_true) ** 2
    cumrmse[name] = np.sqrt(se.expanding().mean())


# ======================================================================
# Plot 1: Expert estimates vs true alpha
# ======================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
t = forecasts.index
for col in forecasts.columns:
    ax.plot(t, forecasts[col], alpha=0.5, label=col)
ax.axhline(alpha_true, color="black", linestyle="--", linewidth=1.5, label=r"True $\alpha$")
ax.plot(t, results["PDM (angles)"], linewidth=2.5, color="tab:red", label="PDM (angles)")
ax.plot(t, results["Linear Pooling"], linewidth=1.5, linestyle=":", color="gray", label="Linear Pooling")
ax.set_xlabel("Period after structural break")
ax.set_ylabel(r"Estimate of $\alpha$")
ax.set_title("Expert estimates and pooled forecasts")
ax.legend(fontsize=8)

# ======================================================================
# Plot 2: Cumulative RMSE learning curves
# ======================================================================

ax = axes[1]
styles = {
    "PDM (angles)": dict(linewidth=2.5, color="tab:red"),
    "PDM (distances)": dict(linewidth=2, linestyle="--", color="tab:orange"),
    "Granger Causality": dict(linewidth=1.5, linestyle="-.", color="tab:blue"),
    "Lagged Correlation": dict(linewidth=1.5, linestyle="-.", color="tab:cyan"),
    "Multivar. Regression": dict(linewidth=1.5, linestyle=":", color="tab:purple"),
    "Transfer Entropy": dict(linewidth=1.5, linestyle=":", color="tab:brown"),
    "Linear Pooling": dict(linewidth=1.5, linestyle=":", color="gray"),
    "Median Pooling": dict(linewidth=1.5, linestyle="--", color="tab:olive"),
}
for name, cum in cumrmse.items():
    ax.plot(t, cum, label=name, **styles.get(name, {}))

ax.set_xlabel("Period after structural break")
ax.set_ylabel("Cumulative RMSE")
ax.set_title(r"Learning speed: cumulative RMSE vs true $\alpha$")
ax.legend(fontsize=7)

plt.tight_layout()
plt.show()


# ======================================================================
# Monte Carlo: average RMSE over many seeds
# ======================================================================

print("\n" + "=" * 70)
print("Monte Carlo: average RMSE over 100 simulations")
print("=" * 70)

N_MC = 100
mc_rmse = {name: [] for name in results.keys()}

for seed in range(N_MC):
    fc, at = simulate_bayesian_experts(
        alpha_minus=3.0, alpha_plus=1.5, n_experts=3, T=10,
        n_obs_base=5, n_obs_pioneer=6, seed=seed,
    )

    mc_results = {}
    w = compute_pioneer_weights_angles(fc)
    mc_results["PDM (angles)"] = pooled_forecast(fc, w)

    w = compute_pioneer_weights_distance(fc)
    mc_results["PDM (distances)"] = pooled_forecast(fc, w)

    w = compute_granger_weights(fc, maxlag=1)
    mc_results["Granger Causality"] = pooled_forecast(fc, w)

    w = compute_lagged_correlation_weights(fc, lag=1)
    mc_results["Lagged Correlation"] = pooled_forecast(fc, w)

    w = compute_multivariate_regression_weights(fc, lag=1)
    mc_results["Multivar. Regression"] = pooled_forecast(fc, w)

    w = compute_transfer_entropy_weights(fc, n_bins=3, lag=1)
    mc_results["Transfer Entropy"] = pooled_forecast(fc, w)

    w = compute_linear_pooling_weights(fc)
    mc_results["Linear Pooling"] = pooled_forecast(fc, w)

    mc_results["Median Pooling"] = compute_median_pooling(fc)

    for name, pooled in mc_results.items():
        rmse = np.sqrt(((pooled - at) ** 2).mean())
        mc_rmse[name].append(rmse)

# Print Monte Carlo results
pdm_mc = np.mean(mc_rmse["PDM (angles)"])
print(f"\n{'Method':30s}  {'Mean RMSE':>10s}  {'Std':>8s}  {'Relative':>8s}")
print("-" * 60)
for name in sorted(mc_rmse.keys(), key=lambda n: np.mean(mc_rmse[n])):
    mean_r = np.mean(mc_rmse[name])
    std_r = np.std(mc_rmse[name])
    rel = mean_r / pdm_mc if pdm_mc > 0 else float("nan")
    print(f"{name:30s}  {mean_r:10.4f}  {std_r:8.4f}  {rel:8.2f}")
