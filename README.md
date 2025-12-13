# Credit-Risk-Model (BNPL)

## Credit Scoring Business Understanding

**Basel II & interpretability**  
Basel II forces banks to justify model inputs, coefficients and monitoring procedures to supervisors.  
An interpretable model (logistic regression with WoE) allows direct mapping of each feature to a probability of default (PD) and therefore satisfies Pillar 1 (minimum capital) and Pillar 3 (disclosure).

**Proxy variable necessity**  
We lack a “default” label. A proxy (here: disengaged cluster from RFM) is therefore mandatory to cast the problem as supervised learning.  
Business risk: if the proxy is mis-aligned with true default, the bank will mis-price loans → higher unexpected losses → regulatory capital breach.

**Simple vs. complex trade-off**  
|               | Interpretable (Logistic+WoE) | Complex (XGBoost) |
|---------------|------------------------------|-------------------|
| Explainability| High – coefficients & WoE    | Low – SHAP only   |
| Capital rule  | Easy to map to PD            | Needs full validation |
| Regulation    | Preferred under FIRB/AIRB    | Acceptable if validated |
| Performance   | May lag 1-2 % AUC            | Usually higher AUC |
For a regulated financial product we **start** with the interpretable model and **benchmark** against the complex one.