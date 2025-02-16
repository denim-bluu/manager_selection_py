# Paper Summary - Can Mutual Fund ‘Stars’ Really Pick Stocks? New Evidence from a Bootstrap Analysis

## Implementation & Details

1. **Model Selection**
   Use the Carhart 4-factor model (or another benchmark relevant to the fund’s strategy):
   $$r_t - r_f = \alpha + \beta_1(RMRF_t) + \beta_2(SMB_t) + \beta_3(HML_t) + \beta_4(PR1YR_t) + \epsilon_t$$

3. **Estimate Parameters**
   - Run an OLS regression on manager's ex-post returns to obtain:
     - Factor loadings $\hat{\beta}$ ($\beta_1, \beta_2, \beta_3, \beta_4$)  
     - Residuals ($\epsilon_t$)  
     - Estimated alpha ($\alpha$) and its $t$-statistic  

4. **Bootstrap**
   - **H0 Null Hypothesis**: True $\alpha = 0$ (no skill).
   - **Pseudo Returns**:
      $$r_t^{\text{pseudo}} = \hat{\beta}\text{Factors} + \epsilon_t^{\text{resampled}}$$
   - Resample residuals **with replacement** (use block-bootstrapping if residuals show autocorrelation, in the paper they found no significant difference in terms of results between block and standard bootstrapping)

5. **Simulation Process**
   For $B = 10,000$ iterations:
   - Generate pseudo returns using resampled residuals
   - Re-run the factor model on pseudo returns to estimate $\alpha^*_b$
     - $$r_t^{\text{pseudo}} = \alpha^*_b + \beta_1^*(RMRF_t) + \beta_2^*(SMB_t) + \beta_3^*(HML_t) + \beta_4^*(PR1YR_t) + \epsilon_t^*$$
   - Store all $\alpha^*_b$ values

6. **Statistical Inference**
   - **p-value**:
      $$p_{\text{alpha}} = \frac{\text{Number of } \alpha^*_b \geq \alpha_{\text{actual}} + 1}{B + 1}$$
      $$p_{\text{t-stat}} = \frac{\text{Number of } t_{\alpha^*_b} \geq t_{\alpha_{\text{actual}}}}{B + 1}$$
   - **Interpretation**:
     - $p < 0.05$: Significant evidence of skill ($\alpha$ unlikely under "luck").
     - $p \geq 0.05$: Performance consistent with random chance.

## **Key Considerations**

- **Factor Model Integrity**: Ensure the factor model of choice captures the fund’s risk exposures.
- **Residual Properties**:
  - Check for heteroskedasticity/autocorrelation (use Newey-West standard errors).
  - For time-dependent residuals, use a **block bootstrap** (e.g., 6-month blocks).
- **Economic vs. Statistical Significance**: Even with $p < 0.05$, assess if $\alpha$ exceeds fees/trading costs.
- **Using t-statistic instead of raw alpha**: More robust to non-normality in $\alpha$ distribution.
