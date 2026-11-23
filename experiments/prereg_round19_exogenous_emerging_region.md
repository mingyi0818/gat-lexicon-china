# Preregistration — exogenous emerging splits and region-level clustering

- Date written: 2026-11-15, before the analytical runs below.
- Scripts to be executed after this file:
  - `code/round19_exogenous_emerging_region.py` (B2 inbound tercile; A region aggregation)
  - `code/round19_build_5a_and_emerging.py` then the same analysis entry for B3
- Seed: 20260930 (shared with Round-16 WCR scripts). No randomisation enters the emerging splits; the seed is recorded only for any bootstrap that is re-used.

## Questions

1. **B2.** If “emerging” is redefined by the bottom tercile of province-mean inbound tourist arrivals over 2003–2007 (a series that is not the domestic-spending outcome), does the lagged composite GAT × emerging interaction remain positive and cluster-robust significant?
2. **A.** If standard errors (and, if feasible, the clustering unit) are taken at the seven-region level rather than the province level, what happens to the baseline interaction’s precision? Cameron–Gelbach–Miller (2008) style few-cluster caution applies: G ≈ 7 is more conservative than G = 27.
3. **B3.** If “emerging” is redefined by below-median cumulative national 5A scenic-spot stock as of end-2011 (resource endowment, not the outcome), does the interaction survive?

## Design (fixed before the run)

- Outcome and exposure unchanged: Δln(domestic tourism spending); lagged four-statistic GAT composite; exact two-way within transform; CR1 cluster-robust inference.
- Baseline emerging (outcome median) and pre-sample spending emerging remain archived reference points; they are not re-estimated as “new discoveries.”
- **B2 emerging:** province-mean `inbound_tourists` over 2003–2007; bottom tercile = emerging. Time-invariant within province.
- **A regions:** fixed map — North China (河北, 山西, 内蒙古); Northeast (辽宁, 吉林, 黑龙江); East (江苏, 浙江, 安徽, 福建, 江西, 山东); Central (河南, 湖北, 湖南); South (广东, 广西, 海南); Southwest (四川, 贵州, 云南, 西藏); Northwest (陕西, 甘肃, 青海, 宁夏, 新疆). Cluster SEs by region (G = 7). Also report province-clustered point estimate unchanged with region-clustered SEs only.
- **B3 emerging:** build province-year cumulative 5A counts from a designation list (source recorded in the result JSON); take the cumulative stock at year = 2011; provinces below the cross-section median are emerging. Years before the 5A scheme (pre-2007) contribute zero designations.
- Optional restricted WCR (Webb, B = 9,999) only for specifications whose cluster-robust p < 0.10, to avoid fishing.

## Decision rules

- Failure is reportable. A null or attenuated interaction under B2/B3 strengthens the cautionary reading and must enter the manuscript as such.
- No specification may be promoted into the abstract unless it also survives Holm adjustment over the enlarged family that includes the new splits (or a clearly labelled reduced independent family). Default plan: report in §7.3 / robustness text; do not rewrite the abstract headline unless the user later authorises an abstract change after seeing the numbers.
- Do not drop provinces to manufacture balance beyond the existing 27-unit corpus constraint.

## Outputs

- `results/round19_exogenous_emerging_region.json`
- `results/provincial_5a_stock.csv` and `results/round19_5a_emerging.json` (B3)
