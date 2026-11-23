# Preregistration — Round 20 integrity remediation

- Written: 2026-11-15, before the Round 20 scripts are created or run.
- Purpose: address the evidence-backed issues in `reviewer_feedback.md` without searching for significance.
- Governing rule: every planned result is retained in the archived output. Null, sign-reversed, imprecise, or validation-failing results are reportable and may not be discarded because they weaken the paper.
- Seed: 20261115 for any resampling. Deterministic analyses record the seed but do not use it.

## Pre-run integrity amendment

Before any Round 20 analysis completed successfully, inspection of the downloaded source workbook established that legacy script `04_attempt2_fixed_merge.py` had mislabelled column 968. Its exact header is `国际旅游外汇收入总额(百万美元)` (international tourism foreign-exchange receipts, US$ million), not domestic tourism spending. The workbook contains no domestic-tourism-revenue column. External checks for Qinghai and Hunan confirmed that the legacy values match inbound-receipt magnitudes and do not match official domestic-tourism revenue.

Accordingly, all references below to “domestic tourism spending” are amended to “international/inbound tourism receipts in US$ million,” and all log growth outcomes are growth in those receipts. No numeric outcome value is changed. Round 20 reads the source-traced `merged_panel_corrected.csv`; the legacy file is retained only for audit. This amendment changes the semantic estimand, not the planned models or decision rules.

## R20-1: H1 timing estimand

### Question

Does pooled Government Attention to Tourism (GAT), without an emerging-destination interaction, predict tourism growth contemporaneously, one year later, or two years later?

### Fixed design

- Panel: the canonical 27-province `merged_panel_v2.csv`.
- Outcome: annual change in log domestic tourism spending.
- Exposure: four-statistic GAT composite at lags 0, 1, and 2.
- Estimator: exact two-way fixed effects; province-clustered CR1 standard errors.
- Separate models: one exposure lag per model.
- Joint distributed-lag model: lags 0, 1, and 2 entered together on their common complete-case sample.
- Wald tests in the joint model: lag 0 = lag 1; lag 0 = lag 2; all three GAT coefficients = 0.
- The coefficient on lagged GAT in the emerging-interaction model is not an average effect; it will be labelled as the effect for the mature/reference group.

### Decision rule

H1 is supported only if a lagged pooled coefficient is distinguishable from zero and stronger than the contemporaneous coefficient under the preregistered Wald comparison. Otherwise H1 is not supported. A nominal result in only one of the three lag models is exploratory and must be interpreted with the three-lag family in view.

## R20-2: maturity-classification timing

### Question

Does the end-2011 5A-stock moderator produce the same interaction when the outcome window is restricted to years after classification?

### Fixed design

- Emerging status: the already archived below-median end-2011 cumulative 5A stock.
- Restricted outcome years: 2012–2019.
- Exposure: one-year lagged four-statistic GAT composite.
- Outcome: annual change in log domestic tourism spending.
- Estimator: exact two-way fixed effects; province-clustered CR1 standard errors.
- The moderator will be called “outcome-distinct but potentially endogenous,” not exogenous.
- The third-party designation list's URL, coverage, and limitations will be retained in the output and disclosed.

### Decision rule

The restricted estimate is descriptive. No result from this short post-classification panel can establish exogeneity or causality, irrespective of its p-value.

## R20-3: province-level H3

### Question

Is province-mean GAT negatively associated with province-level tourism maturity when the effective unit of variation is correctly treated as the province?

### Fixed design

- One row per province (N = 27).
- Dependent variable: province-mean four-statistic GAT composite over 2003–2019.
- Focal regressor A: province-mean log domestic tourism spending over 2003–2019 (descriptive long-run maturity).
- Focal regressor B: province-mean log domestic tourism spending over 2003–2007 (timing sensitivity).
- Scale control: province-mean log GDP over the matching window.
- Estimator: OLS with HC3 standard errors.
- Diagnostics: Cook's distance; leave-one-province-out coefficient range and sign count.

### Decision rule

H3 is not supported unless the coefficient is negative and robust in direction to the pre-period definition and leave-one-out analysis. Because N = 27 and maturity remains observational, all estimates are descriptive.

## R20-4: index scaling and dictionary sensitivity

### Questions

1. Does restricting component standardisation to the 2003–2019 estimation window change the main H2 result or equal-weight two-layer sensitivity?
2. Does the H2 interaction survive reasonable, fixed dictionary perturbations?

### Fixed scaling variants

- Published/canonical composite: components z-scored over 2003–2024.
- Estimation-window composite: the same four raw components z-scored over 2003–2019 only.
- Equal-weight two-layer sensitivity: estimation-window composite z-score averaged with TF–IDF z-scored over the matched 2003–2019 panel.

### Fixed dictionary variants

The 37 terms are partitioned before the run:

- Generic sector terms: 旅游, 文旅, 文化旅, 游客, 景区, 景点, 接待.
- Tourism formats: 全域旅游, 智慧旅游, 乡村旅游, 红色旅游, 生态旅游, 康养旅游, 冰雪旅游, 入境旅游, 出境旅游, 夜间文旅.
- Designation/accommodation instruments: 5A, 4A, A级景区, 民宿, 酒店, 精品线路.
- Industry/outcome phrases: all remaining lexicon terms.

Planned variants:

- remove potentially ambiguous generic terms 接待, 酒店, 民宿;
- leave out each of the four term groups in turn;
- retain all terms but standardise components only on 2003–2019.

Each variant recomputes all four lexical components from the source reports, forms an equal-weight component-z composite on 2003–2019, merges to the canonical panel, and estimates the baseline one-year GAT × outcome-derived-emerging interaction using exact TWFE and province-clustered CR1 inference.

### Decision rule

The lexical measure is considered dictionary-robust only if the interaction direction and substantive magnitude are not governed by one term group. Statistical significance is not required and will not be used to choose a preferred dictionary. These automated sensitivity checks do not substitute for independent human validation. No AI-generated labels will be represented as human coding.

## R20-5: outcome-cell sensitivity

### Question

Do the headline level and growth conclusions depend on the observations already flagged as low or extreme?

### Fixed design

- Flagged low-spending cells: Ningxia 2003–2012 and Qinghai 2003–2004.
- Flagged extreme-growth years: every canonical observation with absolute annual log growth above 0.70.
- Sensitivity A: exclude flagged low-spending cells and any growth observations whose current or lagged spending cell is flagged.
- Sensitivity B: exclude all absolute-growth-above-0.70 observations.
- Sensitivity C: exclude the union.
- Re-estimate the contemporaneous level model, pooled lag-0/1/2 growth models, and baseline H2 interaction.
- Verification status for each cell will be recorded separately. Absence of independent verification is reported as “unverifiable,” not “incorrect.”

### Decision rule

No exclusion specification replaces the canonical data without source evidence of an error. It is a sensitivity analysis only. Any material sign or inference change must enter the manuscript.

## R20-6: coefficient figure

- A new figure will plot pooled lag-0/1/2 growth coefficients and H2 interaction estimates under baseline, bottom-tercile, pre-sample spending, early-inbound, end-2011 5A, and equal-weight two-layer definitions.
- Every point and confidence interval must be generated from an archived Round 16/19/20 result or recomputed in the Round 20 canonical script.
- The figure is descriptive where the underlying specification is exploratory.

## Human validation limitation

The review recommends two independent human coders. This cannot be completed honestly by the current automated agent. Round 20 will prepare a reproducible sampling frame and coding protocol only if source-text access permits, but will not fabricate coder identities, agreement, precision, or recall. Until genuine independent coding occurs, the manuscript must state that direct human content validation remains absent and narrow the construct claim accordingly.

## Planned outputs

- `code/round20_integrity_remediation.py`
- `results/round20_integrity_remediation.json`
- `results/round20_dictionary_sensitivity.csv`
- `results/round20_outcome_cell_audit.csv`
- `figures/fig3_lag_heterogeneity_coefficients.png`

