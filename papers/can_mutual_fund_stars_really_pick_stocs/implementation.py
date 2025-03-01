import time
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from pydantic import BaseModel
from statsmodels.regression.linear_model import OLS


def simulate_fund_data(n_months: int = 120, seed: int = 42):
    """
    Generates simulated mutual fund returns and factor data
    Returns:
        fund_returns (pd.DataFrame): Simulated raw returns for funds
        factors_df (pd.DataFrame): Simulated factor returns + risk-free rate
    """
    np.random.seed(seed)

    # Factor parameters (monthly)
    factor_means = {
        "RMRF": 0.005,  # Market excess return
        "SMB": 0.002,  # Small minus big
        "HML": 0.004,  # Value factor
        "PR1YR": 0.006,  # Momentum
    }

    factor_vols = {"RMRF": 0.04, "SMB": 0.03, "HML": 0.03, "PR1YR": 0.04}

    # Generate correlated factors
    factors = pd.DataFrame(
        {
            f: np.random.normal(mean, vol, n_months)
            for f, (mean, vol) in zip(
                factor_means.keys(), zip(factor_means.values(), factor_vols.values())
            )
        }
    )

    # Generate fund returns
    fund_returns = pd.DataFrame()
    # Random factor exposures
    betas = {
        "RMRF": np.random.uniform(0.8, 1.2),
        "SMB": np.random.uniform(-0.5, 0.5),
        "HML": np.random.uniform(-0.3, 0.3),
        "PR1YR": np.random.uniform(-0.2, 0.2),
    }

    # Residual risk (idiosyncratic volatility)
    sigma = np.random.uniform(0.01, 0.03)

    # Compute excess returns (alpha=0 under null)
    excess_return = (
        betas["RMRF"] * factors["RMRF"]
        + betas["SMB"] * factors["SMB"]
        + betas["HML"] * factors["HML"]
        + betas["PR1YR"] * factors["PR1YR"]
        + np.random.normal(0, sigma, n_months)
    )

    # Raw returns = risk-free rate + excess return
    fund_returns["Fund"] = excess_return

    return fund_returns, factors[["RMRF", "SMB", "HML", "PR1YR"]]


class BootstrapResult(BaseModel):
    actual_alpha: float
    actual_t_stat: float
    p_value_alpha: float
    p_value_t_stat: float


def bootstrap_skill_test(
    fund_excess_returns: np.ndarray, factors_returns: np.ndarray, n_iter: int = 10_000
) -> BootstrapResult:
    """
    Test if a fund's alpha is statistically significant using bootstrap.
    This is a single fund version of the bootstrap test.

    Parameters:
        fund_excess_returns (np.ndarray): Excess returns of the fund
        factors_returns (np.ndarray): Factor returns (e.g., RMRF, SMB, HML, PR1YR)
        n_iter (int): Number of bootstrap samples

    Returns:
        BootstrapResult: Actual alpha, p-value, and bootstrapped alpha distribution
    """
    if len(fund_excess_returns) != len(factors_returns):
        raise ValueError("Fund and factor returns must have the same length")

    # Step 1: Estimate alpha and residuals
    X = sm.add_constant(factors_returns)
    model = OLS(fund_excess_returns, X).fit()

    # Store actual results
    actual_alpha = model.params[0]
    actual_t_stat = model.tvalues[0]
    residuals = model.resid
    betas = model.params[1:]
    X_factors = factors_returns.copy()

    # Step 2: Bootstrap residuals under H0 (alpha=0)
    boot_alphas = []
    boot_t_stats = []
    n_obs = len(residuals)

    for _ in range(n_iter):
        # Resample residuals with replacement
        resampled_idx = np.random.choice(n_obs, size=n_obs, replace=True)
        resampled_residuals = residuals[resampled_idx]

        # Generate pseudo returns
        pseudo_returns = X_factors.dot(betas) + resampled_residuals

        # Retrieve bootstrap alpha
        pseudo_model = OLS(pseudo_returns, sm.add_constant(X_factors)).fit()
        boot_alphas.append(pseudo_model.params[0])
        boot_t_stats.append(pseudo_model.tvalues[0])

    # Step 3: Calculate p-value
    p_alpha = (np.sum(np.array(boot_alphas) >= actual_alpha) + 1) / (n_iter + 1)
    p_t_stat = (np.sum(np.array(boot_t_stats) >= actual_t_stat) + 1) / (n_iter + 1)

    return BootstrapResult(
        actual_alpha=actual_alpha,
        actual_t_stat=actual_t_stat,
        p_value_alpha=p_alpha,
        p_value_t_stat=p_t_stat,
    )


def bootstrap_skill_test_parallel(
    fund_returns: np.ndarray,
    factors: np.ndarray,
    n_iter: int = 10_000,
    n_jobs: int = -1,
) -> BootstrapResult:
    """
    Parallel implementation of bootstrap test for mutual fund performance

    Parameters:
        fund_returns (np.ndarray): Fund excess returns
        factors (np.ndarray): Factor returns matrix
        n_iter (int): Number of bootstrap iterations
        n_jobs (int): Number of parallel jobs (-1 uses all cores)

    Returns:
        BootstrapResult: Actual alpha, p-value, and bootstrapped alpha distribution
    """
    # Initial regression to get betas and residuals
    X = sm.add_constant(factors)
    model = sm.OLS(fund_returns, X).fit()

    def bootstrap_iteration(
        factors: np.ndarray,
        betas: np.ndarray,
        residuals: np.ndarray,
        seed: int,
    ):
        """Single bootstrap iteration with fixed parameters"""
        # Ensure that each parallel process has a different seed
        np.random.seed(seed)
        n_obs = len(factors)

        # Resample residuals with replacement
        resampled_idx = np.random.choice(n_obs, size=n_obs, replace=True)

        # Generate pseudo returns under H0 (alpha=0)
        pseudo_returns = factors.dot(betas) + residuals[resampled_idx]

        # Re-estimate alpha and t-stat
        X = sm.add_constant(factors)
        model = sm.OLS(pseudo_returns, X).fit()

        return model.params[0], model.tvalues[0]

    bootstrap_func = partial(
        bootstrap_iteration,
        factors=factors,
        betas=model.params[1:],
        residuals=model.resid,
    )
    # Run bootstrap iterations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_func)(seed=i) for i in range(n_iter)
    )

    # Unpack results
    boot_alphas, boot_t_stats = zip(*results)

    return BootstrapResult(
        actual_alpha=model.params[0],
        actual_t_stat=model.tvalues[0],
        p_value_alpha=(np.sum(np.array(boot_alphas) >= model.params[0]) + 1)
        / (n_iter + 1),
        p_value_t_stat=(np.sum(np.array(boot_t_stats) >= model.tvalues[0]) + 1)
        / (n_iter + 1),
    )


if __name__ == "__main__":
    fund_returns, factors = simulate_fund_data()
    excess_returns = fund_returns["Fund"].to_numpy().flatten()
    factor_returns = factors.to_numpy()
    n_iter = 100_000
    start_time = time.perf_counter()
    result = bootstrap_skill_test(excess_returns, factor_returns, n_iter=n_iter)
    end_time = time.perf_counter()
    print(f"Running bootstrap test with {n_iter} iterations")
    print(result.model_dump_json(indent=4))
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    start_time = time.perf_counter()
    result_parallel = bootstrap_skill_test_parallel(
        excess_returns, factor_returns, n_iter=n_iter
    )
    end_time = time.perf_counter()
    print(f"Running parallel bootstrap test with {n_iter} iterations")
    print(result_parallel.model_dump_json(indent=4))
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
