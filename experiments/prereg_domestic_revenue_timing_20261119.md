# Preregistration — domestic tourism revenue timing (07_policy_gat)

Date: 2026-11-19. Written before estimation.

## Question
Does the same provincial lexicon GAT that is jointly null for inbound foreign-exchange receipt growth also fail to predict growth in domestic tourism revenue from provincial statistical bulletins?

## Data
- GAT and calendar keys: `results/merged_panel_calendar_safe.csv` (exact-calendar GAT_L0, GAT_L1, GAT_L2).
- Domestic revenue: `domestic_tourism_revenue_yi` (亿元) from `results/merged_panel_corrected_with_domestic.csv`, parse_status == ok, value > 0. Do not use `total_tourism_revenue_yi` (it includes inbound spending).
- Sample: the 27 provinces already in the calendar-safe panel, years 2003–2019. No interpolation of missing bulletin cells.

## Estimator
Two-way fixed effects by iterative demeaning. Standard errors clustered by province.
Outcome: consecutive-calendar-year change in log domestic tourism revenue.
Separate regressions at k = 0, 1, 2, plus one joint regression of the three lags on the common sample.
Joint Wald test that all three coefficients are zero is the only confirmatory test.

## What will be reported
All four fits, observation counts, and an 80% minimum detectable coefficient using t critical values with G−1 degrees of freedom on the one-year lag. No heterogeneity search. No term will be dropped after seeing coefficients.

## Interpretation (fixed now)
- A positive joint test at 5% would mean the domestic-revenue margin differs from the foreign-exchange margin. It would still not be a causal effect.
- A null is the expected outcome under the paper's existing foreign-exchange evidence. It will be reported as a null, with the minimum detectable coefficient. It does not by itself turn the paper into a high-acceptance Applied Economics article.
