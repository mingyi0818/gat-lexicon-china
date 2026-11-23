# Preregistration — Round 21 calendar-integrity correction

- Written: 2026-11-15, before creating or running the Round 21 analysis.
- Trigger: an independent provenance check showed that row-based `groupby().diff()` and `shift()` crossed missing report years. Shanxi 2005 was therefore calculated against 2003, not 2004.
- Governing rule: calendar-corrected results replace Round 20 dynamic results regardless of sign, magnitude, or statistical significance. Every planned output is retained.
- Seed: 20261115 for the restricted wild cluster bootstrap.

## Source-cell correction

The mirrored workbook reports Ningxia 2011 international tourism receipts as 6.00 million USD. China Statistical Yearbook 2012, Table 18-23, reports 6.20 million USD. The calendar-corrected canonical panel will use the official 6.20 value and retain a mirror-value sensitivity using 6.00.

Official source: https://www.stats.gov.cn/sj/ndsj/2012/html/Q1823C.HTM

## Calendar-safe panel

- Outcome base: all 27 study provinces and every calendar year 2003–2019 from the yearbook outcome panel.
- Annual receipt growth at year t: `ln(receipts_t) - ln(receipts_t-1)` only when `year_t - year_t-1 = 1`.
- GDP growth follows the same consecutive-year rule.
- GAT lag k is joined by the exact key `(province, year-k)`, never by row position.
- A one-year lag model may use year t when the report at t-1 exists; a current-year report is not additionally required.
- A contemporaneous model requires the report at t. A two-year model requires the report at t-2.
- The joint lag model requires reports at t, t-1, and t-2.
- Receipt-scale classifications use the complete outcome panel and therefore do not depend on report missingness.

## Models and decision rules

1. Re-estimate pooled H1 lag 0, 1, and 2 models and their joint Wald tests.
2. Re-estimate the six primary H2 specifications: baseline median split, bottom tercile, early-period receipt split, GDP-growth control, lagged-receipt convergence control, and equal-weight two-layer index. Recompute the six-test Holm adjustment.
3. Re-estimate early-arrival, official end-2011 5A, post-2011 5A, region-cluster, dictionary, scaling, winsorisation, and extreme-growth deletion checks on calendar-safe variables.
4. Re-estimate the baseline restricted wild cluster bootstrap with 9,999 Webb and Rademacher draws.
5. Re-estimate H3 on province means from the complete outcome panel with HC3 and leave-one-out diagnostics.
6. Rebuild Tables 2–5 inputs and Figures 3–4 where affected.

No old p-value remains canonical after this correction. Null, adverse, or sign-reversed results are reported without qualification shopping.

## Planned outputs

- `code/round21_calendar_integrity.py`
- `results/merged_panel_calendar_safe.csv`
- `results/round21_calendar_integrity.json`
- `results/round21_dictionary_sensitivity.csv`
- `results/round21_outcome_cell_audit.csv`
- `figures/fig3_lag_heterogeneity_coefficients.png`
- updated manuscript and reviewer dossier
