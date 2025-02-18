import numpy as np
import pandas as pd
import statsmodels.api as sm


class TreynorMazuyModel:
    """
    Implementation of the Treynor-Mazuy model, which extends the CAPM framework
    to assess market timing ability by including a quadratic term in the regression.
    """

    def __init__(self, portfolio_returns: pd.Series, market_returns: pd.Series):
        self.portfolio_returns = portfolio_returns
        self.market_returns = market_returns

    def fit(self):
        """
        Fits the Treynor-Mazuy quadratic regression model.
        """
        # Construct the design matrix with a constant, market returns, and quadratic market returns
        X = pd.DataFrame(
            {
                "market_returns": self.market_returns,
                "market_returns_squared": self.market_returns**2,
            }
        )
        X = sm.add_constant(X)
        y = self.portfolio_returns
        self.model = sm.OLS(y, X).fit()
        return self.model.summary()


class HenrikssonMertonModel:
    """
    Implementation of the Henriksson-Merton model, which uses a dual-beta regression
    to test for market timing ability by incorporating a dummy for up-market returns.
    """

    def __init__(self, portfolio_returns: pd.Series, market_returns: pd.Series):
        self.portfolio_returns = portfolio_returns
        self.market_returns = market_returns

    def fit(self):
        """
        Fits the Henriksson-Merton dual-beta regression model.
        """
        # Construct the design matrix with a constant, market returns, and the up-market dummy variable
        X = pd.DataFrame(
            {
                "market_returns": self.market_returns,
                "up_market": np.maximum(0, self.market_returns),
            }
        )
        X = sm.add_constant(X)
        y = self.portfolio_returns

        self.model = sm.OLS(y, X).fit()
        return self.model.summary()


if __name__ == "__main__":
    np.random.seed(42)

    # Generate sample data
    portfolio_returns = pd.Series(np.random.normal(0.01, 0.02, 100))
    market_returns = pd.Series(np.random.normal(0.01, 0.015, 100))

    treynor_mazuy = TreynorMazuyModel(portfolio_returns, market_returns)
    print("Treynor-Mazuy Model Summary:")
    print(treynor_mazuy.fit())

    henriksson_merton = HenrikssonMertonModel(portfolio_returns, market_returns)
    print("\nHenriksson-Merton Model Summary:")
    print(henriksson_merton.fit())
