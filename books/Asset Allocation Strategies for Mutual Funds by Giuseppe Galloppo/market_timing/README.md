# Chapter 2.5 Market Timing

Market timing measures managers' ability to dynamically shift capital by increasing beta before market rallies and reducing beta before downturns. Successful timing enhances returns by aligning portfolio sensitivity (β) with anticipated market trends.

## Model Specifications  

### Treynor-Mazuy Quadratic Model  

Extended the CAPM by adding a quadratic market return term to capture nonlinear relationships caused by timing decisions. The equation is:

$$
r_{p,t} = \alpha_p + \beta_p r_{m,t} + \gamma_p r_{m,t}^2 + \varepsilon_{p,t}
$$

- $r_{p,t}$: Excess return of the portfolio over the risk-free rate at time t
- $r_{m,t}$: Excess return of the market portfolio  
- $\alpha_p$: Selectivity skill (persistent outperformance unrelated to market movements)  
- $\beta_p$: Baseline market exposure  
- $\gamma_p$: Timing ability coefficient. A positive $\gamma_p$ indicates successful timing: managers raise beta in rising markets and lower it in falling markets.  

The quadratic term ($r_{m,t}^2$) isolates timing effects: larger absolute market returns amplify gains for skilled timers.  

### Henriksson-Merton Dual-Beta Model  

Proposes a dual-beta framework where managers adjust beta based on bullish/bearish forecasts. The model uses a dummy variable to distinguish market phases:  

$$
r_{p,t} = \alpha_p + \beta_p r_{m,t} + \gamma_p \max(0, r_{m,t}) + \varepsilon_{p,t}
$$

- $\gamma_p$: Difference between "up-market" and "down-market" betas. A positive $\gamma_p$ implies higher beta during bull markets.  

This approach tests whether managers systematically alter risk exposure in anticipation of market shifts.
