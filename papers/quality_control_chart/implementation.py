from typing import Tuple

import numpy as np
import plotly.graph_objects as go


class QualityControlChart:
    def __init__(self, data: np.ndarray, confidence_level: float = 0.95):
        self.data = data
        self.ci = confidence_level
        self.std = np.std(self.data)
        self.time_periods = len(self.data)
        self.z_score = 1.96  # For 95% confidence = 1.96 z-score

    def calculate_bounds(self, t: float) -> Tuple[float, float]:
        """Calculate control limits at time t"""
        sigma = self.std / np.sqrt(t)
        ucl = self.z_score * sigma
        lcl = self.z_score * sigma
        return ucl, lcl

    def plot(self, title: str = "Quality Control Chart"):
        """Generate the chart"""
        x_axis = np.arange(1, self.time_periods + 1, dtype=int)
        cumul_returns = np.cumsum(self.data) / np.arange(1, len(self.data) + 1)
        ucls = [self.calculate_bounds(i)[0] for i in x_axis]
        lcls = [self.calculate_bounds(i)[1] for i in x_axis]

        fig = go.Figure()

        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([ucls, lcls[::-1]]),
                fill="toself",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Band",
            )
        )

        # Add benchmark line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=[0] * len(x_axis),
                line=dict(color="black", dash="dash"),
                name="Benchmark",
            )
        )

        # Add cumulative returns
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=cumul_returns,
                line=dict(color="blue"),
                name="Cumulative Excess Returns",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Cumulative Excess Returns",
            showlegend=True,
            hovermode="x unified",
        )

        return fig


if __name__ == "__main__":
    # Example 1: Manager without skills (Random Walk)
    np.random.seed(42)
    unskilled_returns = np.random.normal(0, 0.02, 36)  # zero drift

    # Example 2: Manager with skills (Consistent Outperformance)
    skilled_returns = np.random.normal(0.01, 0.02, 36)  # Positive drift of 1% per month

    unskilled_qcc = QualityControlChart(unskilled_returns)
    skilled_qcc = QualityControlChart(skilled_returns)

    fig_unskilled = unskilled_qcc.plot(
        "Unskilled Manager: Returns Within Confidence Band"
    )
    fig_skilled = skilled_qcc.plot("Skilled Manager: Returns Outside Confidence Band")

    fig_unskilled.show()
    fig_skilled.show()
