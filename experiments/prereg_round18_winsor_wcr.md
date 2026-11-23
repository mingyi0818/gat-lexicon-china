# Preregistration — composite-scale winsorised wild cluster bootstrap

- Date written: 2026-11-15, before the run.
- Script to be executed after this file: `code/round18_wcr_winsorised.py`.
- Question: On the operative four-statistic composite, what is the restricted wild-cluster-bootstrap p-value of the lagged GAT × emerging interaction when tourism-growth is winsorised at the 1st and 99th percentiles?
- Design (fixed before the run): exact two-way within transformation; regressors are lagged composite GAT and its interaction with the median-split emerging indicator; growth clipped at the quantiles used in `results/round16_composite_rebase.json`; CR1 standard errors; restricted WCR with the null imposed on the interaction; Webb six-point weights; B = 9,999; seed = 20260930 (same seed as `code/round16_wcr_bootstrap.py`).
- Gate: the point estimate must match the archived composite winsorised coefficient (0.08978515753532024) before any p-value is written into the manuscript. If it does not match, stop and do not update the text.
- Reporting rule: whatever p-value the bootstrap returns is reported in Section 4.2. It does not enter the abstract, and it does not change the Holm, ex-ante maturity, or equal-weight conclusions. A significant bootstrap p-value is not a licence to describe the interaction as established.
