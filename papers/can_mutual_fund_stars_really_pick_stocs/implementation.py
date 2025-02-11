import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from pydantic import BaseModel


class BootstrapResult(BaseModel):
    actual_alpha: float
    actual_t_stat: float
    p_value_alpha: float
    p_value_t_stat: float
    bootstrapped_alphas: np.ndarray
    bootstrapped_t_stats: np.ndarray


def bootstrap_skill_test(
    fund_excess_returns: np.ndarray, factors_returns: np.ndarray, n_iter=10000
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
    # Step 1: Estimate alpha and residuals
    X = sm.add_constant(factors_returns)
    model = OLS(fund_excess_returns, X).fit()

    # Store actual results
    actual_alpha = model.params["const"]
    actual_t_stat = model.tvalues["const"]
    residuals = model.resid
    betas = model.params.drop("const")
    X_factors = factors_returns.copy()

    # Step 2: Bootstrap residuals under H0 (alpha=0)
    boot_alphas = []
    boot_t_stats = []
    n_obs = len(residuals)

    for _ in range(n_iter):
        # Resample residuals with replacement
        resampled_idx = np.random.choice(n_obs, size=n_obs, replace=True)
        resampled_residuals = residuals.iloc[resampled_idx]

        # Generate pseudo returns
        pseudo_returns = X_factors.dot(betas) + resampled_residuals.values

        # Retrieve bootstrap alpha
        pseudo_model = OLS(pseudo_returns, sm.add_constant(X_factors)).fit()
        boot_alphas.append(pseudo_model.params["const"])
        boot_t_stats.append(pseudo_model.tvalues["const"])

    # Step 3: Calculate p-value
    p_alpha = (np.sum(np.array(boot_alphas) >= actual_alpha) + 1) / (n_iter + 1)
    p_t_stat = (np.sum(np.array(boot_t_stats) >= actual_t_stat) + 1) / (n_iter + 1)

    return BootstrapResult(
        actual_alpha=actual_alpha,
        actual_t_stat=actual_t_stat,
        p_value_alpha=p_alpha,
        p_value_t_stat=p_t_stat,
        bootstrapped_alphas=np.array(boot_alphas),
        bootstrapped_t_stats=np.array(boot_t_stats),
    )
