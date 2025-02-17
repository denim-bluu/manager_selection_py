
# **Chapter 4.3 Conditional Factor Model: Theory & Intuition**

## **Core Objective**

This model evaluates fund performance while accounting for **time-varying risk exposures** influenced by macroeconomic conditions. It extends classic unconditional factor models (CAPM, Fama-French, Carhart) by allowing both **alpha (α)** and **factor loadings (β)** to dynamically adjust to economic states.
Simply put, the betas and alphas are not fixed but change with the economic environment over time.

## **Model Specification**

The general form is:

$$
R_{it} - R_{ft} = \alpha_{it} + \beta_{i0}(R_{mt} - R_{ft}) + \mathbf{B}_i' \cdot \mathbf{Z}_{t-1} \cdot \mathbf{F}_t + \epsilon_{it}
$$

### **Where**

1. **Time-Varying Alpha**:
   - Represents manager skill adjusted for macroeconomic conditions
   - Specified as (Christopherson et al. (1998)):
     $$
     \alpha_{it} = \alpha_{i0} + \sum_{k=1}^4 \gamma_{ik} Z_{k,t-1}
     $$
   - **Interpretation**:  
     - $\alpha_{i0}$: Baseline skill (persistent alpha)
     - $\gamma_{ik}$: Sensitivity of alpha to lagged macroeconomic variable $Z_{k,t-1}$

2. **Time-Varying Factor Loadings**:
   - For each factor $F_j$ (Market, SMB, HML, Momentum):  
     $$
     \beta_{it}^{F_j} = \beta_{i0}^{F_j} + \sum_{k=1}^4 \delta_{jk} Z_{k,t-1}
     $$
   - **Interpretation**:  
     - $\beta_{i0}^{F_j}$: Baseline exposure to factor $F_j$.  
     - $\delta_{jk}$: How factor sensitivity changes with $Z_{k,t-1}$.

3. **Macroeconomic State Variables Z**:
   - **1-month T-bill rate**: Proxy for monetary policy tightness
   - **Dividend yield**: Signals market valuation levels
   - **Term spread** (Long-term - Short-term rates): Predicts economic growth
   - **Quality spread** (Corporate - Gov bond yields): Reflects credit risk appetite

## **Key Innovations**

1. **Dynamic Risk Adjustment**  
   - Unlike unconditional model, this recognises that funds alter their market/SMB/HML/momentum exposures in response to economic shifts (e.g., recession vs expansion).

2. **Skill Isolation**  
   - Separates persistent manager skill ($\alpha_{i0}$) from temporary alpha fluctuations driven by macroeconomic regimes ($\sum \gamma_{ik} Z_{k,t-1}$).
