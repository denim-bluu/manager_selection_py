import numpy as np
from scipy.optimize import minimize
from numpy import typing as npt


def portfolio_variance(w: npt.NDArray, cov: npt.NDArray) -> float:
    """
    Compute the portfolio variance given weights w and covariance matrix.
    """
    return np.dot(w, cov @ w)


def implied_returns(
    w: npt.NDArray, expected_returns: npt.NDArray, cov: npt.NDArray
) -> npt.NDArray:
    """
    Calculate the implied returns for each fund.
    """
    r_port = np.dot(w, expected_returns)
    v_p = portfolio_variance(w, cov)
    marg_contrib = 2 * cov @ w
    labmd = r_port / (2 * v_p)
    r_imp = labmd * marg_contrib
    return r_imp


def obj_func(w: npt.NDArray, expected_returns: npt.NDArray, cov: npt.NDArray) -> float:
    """
    Objective function: sum of squared differences between each fund's expected return
    and its implied return.
    """
    r_imp = implied_returns(w, expected_returns, cov)

    # Scaling to avoid numerical issues
    r_imp = r_imp * 100
    expected_returns = expected_returns * 100
    return np.sum((expected_returns - r_imp) ** 2)


def weight_constraint(w: npt.NDArray) -> float:
    """
    Constraint that the portfolio weights sum to 1.
    """
    return np.sum(w) - 1


if __name__ == "__main__":
    expected_returns = np.array([0.08, 0.04, 0.07])
    cov = np.array(
        [
            [0.0324, 0.00144, 0.01008],
            [0.00144, 0.0016, 0.00032],
            [0.01008, 0.00032, 0.0064],
        ]
    )
    init_w = np.array([0.6, 0.25, 0.15])

    bounds = [(0.0, 1.0) for _ in range(len(init_w))]
    constraints = {"type": "eq", "fun": weight_constraint}

    result = minimize(
        obj_func,
        init_w,
        args=(expected_returns, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        optimal_weights = result.x
        print(f"Initial weights: {[f'{weight:.2f}' for weight in init_w]}")
        print(f"Initial variance: {portfolio_variance(init_w, cov):.4f}")
        print(
            f"Initial expected portfolio return: {np.dot(init_w, expected_returns):.4f}"
        )
        print(
            f"Initial Implied returns: {[f'{ret:.2f}' for ret in implied_returns(init_w, expected_returns, cov)]}"
        )
        print(f"Optimal weights: {[f'{weight:.2f}' for weight in optimal_weights]}")
        r_imp_opt = implied_returns(optimal_weights, expected_returns, cov)
        print(f"Expected returns: {[f'{ret:.2f}' for ret in expected_returns]}")
        print(f"Implied returns: {[f'{ret:.2f}' for ret in r_imp_opt]}")
        print(f"Optimised variance: {portfolio_variance(optimal_weights, cov):.4f}")
        print(
            f"Optimised expected portfolio return: {np.dot(optimal_weights, expected_returns):.4f}"
        )
    else:
        print("Optimisation failed:", result.message)
