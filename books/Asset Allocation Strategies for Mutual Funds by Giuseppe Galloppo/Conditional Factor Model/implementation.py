from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


class ConditionalFactorModel:
    def __init__(
        self,
        factors: List[str] = ["MKT_RF", "SMB", "HML", "MOM"],
        state_vars: List[str] = ["TBILL", "DIV", "TERM", "QUAL"],
    ):
        self.factors = factors
        self.state_vars = state_vars
        self.results = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares dataset with time-varying components
        Returns design matrix X and target vector y
        """
        # Create lagged economic variables Z_{t-1}
        for z in self.state_vars:
            data[f"{z}_lag"] = data[z].shift(1)

        # Create multiplicative terms: F_j,t * Z_{k,t-1}
        mult_terms = []
        for factor in self.factors:
            for z in self.state_vars:
                term_name = f"{factor}_{z}"
                data[term_name] = data[factor] * data[f"{z}_lag"]
                mult_terms.append(term_name)

        # Construct the design matrix
        X = pd.concat(
            [
                data[[f"{z}_lag" for z in self.state_vars]],  # Alpha terms (γ)
                data[self.factors],  # Base factors (β0)
                data[mult_terms],  # Time-varying betas (δ)
            ],
            axis=1,
        )

        X = pd.DataFrame(sm.add_constant(X.dropna()))
        y = data["EXCESS_RETURN"].loc[X.index]

        return X, y

    def estimate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Estimates model with HAC standard errors.

        HAC standard errors are used to account for potential homoskedasticity
        & autocorrelation in residuals, which is common in financial time-series data.

        """
        model = sm.OLS(y, X)
        self.results = model.fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    def get_summary(self) -> Dict:
        if not self.results:
            raise ValueError("Model not estimated. Call estimate() first")

        return {
            "alpha_0": self.results.params["const"],
            "gamma": self.results.params[
                [f"{z}_lag" for z in self.state_vars]
            ].to_dict(),
            "base_betas": self.results.params[self.factors].to_dict(),
            "delta_coeffs": self.results.params.drop(
                ["const"] + [f"{z}_lag" for z in self.state_vars] + self.factors
            ).to_dict(),
            "rsquared": self.results.rsquared,
            "rsquared_adj": self.results.rsquared_adj,
        }


def generate_sample_data(periods: int = 500) -> pd.DataFrame:
    """GPT generated sample data for Conditional Factor Model"""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            # Factor exposures
            "MKT_RF": np.random.normal(0.08 / 12, 0.15 / np.sqrt(12), periods),
            "SMB": np.random.normal(0.03 / 12, 0.10 / np.sqrt(12), periods),
            "HML": np.random.normal(0.04 / 12, 0.10 / np.sqrt(12), periods),
            "MOM": np.random.normal(0.02 / 12, 0.12 / np.sqrt(12), periods),
            "RF": np.random.normal(0.02 / 12, 0.01 / np.sqrt(12), periods),
            # Economic state variables
            "TBILL": np.random.normal(0.02, 0.01, periods).cumsum(),
            "DIV": np.random.normal(0.02, 0.005, periods).cumsum(),
            "TERM": np.random.normal(0.01, 0.005, periods).cumsum(),
            "QUAL": np.random.normal(0.01, 0.003, periods).cumsum(),
        }
    )

    # Generate time-varying parameters
    alpha = 0.02 / 12 + 0.5 * data["TBILL"].shift(1) + 0.3 * data["DIV"].shift(1)
    beta_mkt = 1.0 + 0.2 * data["TERM"].shift(1) + 0.1 * data["QUAL"].shift(1)

    # Generate fund returns
    data["FUND_RETURN"] = (
        alpha
        + beta_mkt * data["MKT_RF"]
        + 0.4 * data["SMB"]
        + 0.2 * data["HML"]
        + 0.1 * data["MOM"]
        + np.random.normal(0, 0.02 / np.sqrt(12), periods)
    )

    data["EXCESS_RETURN"] = data["FUND_RETURN"] - data["RF"]
    return data.dropna()


if __name__ == "__main__":
    sample_data = generate_sample_data()

    model = ConditionalFactorModel()
    X, y = model.prepare_data(sample_data)
    model.estimate(X, y)

    # Print formatted results
    summary = model.get_summary()
    print(f"Base Alpha (α0): {summary['alpha_0']:.4f}")
    print("\nGamma Coefficients (Time-Varying Alpha):")
    for k, v in summary["gamma"].items():
        print(f"{k}: {v:.4f}")

    print("\nBase Factor Loadings:")
    for k, v in summary["base_betas"].items():
        print(f"{k}: {v:.4f}")

    print("\nDelta Coefficients (Time-Varying Betas):")
    for k, v in summary["delta_coeffs"].items():
        print(f"{k}: {v:.4f}")

    print("\nModel Fit:")
    print(f"Adj. R-squared: {summary['rsquared_adj']:.4f}")
