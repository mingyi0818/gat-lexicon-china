# Preregistration — Round 22: 31-Province Extension (add 4 municipalities)

Date: 2026-11-22
Status: preregistered before execution

## 1. Motivation

The canonical panel (Round 21 and earlier) covers 27 province-level units (22
provinces + 5 autonomous regions) and **omits the 4 municipalities** (北京/天津/上海/重庆).
This preregistration extends the panel to all 31 province-level administrative
divisions to test whether the core results are robust to including municipalities.

## 2. Data scope and sources (all local, no new scraping)

- **Corpus**: `E:\dataset\lawtext_gov\`
  - 27 units: `政府工作报告-省自治区\` (unchanged, same 714 report files as canonical)
  - 4 municipalities: `政府工作报告-主要城市\{北京市,天津市,上海市,重庆市}\`
    (北京市 1998–2026; 天津/上海/重庆 2003–2026; regression window 2003–2019 fully covered)
- **Outcome**: `results/provincial_inbound_receipts_panel.csv` already contains all 31
  units × 17 years (2003–2019) = 527 rows. No change needed.
- **5A stock**: `results/official_mct_5a_20250311.html` already lists 北京/天津/上海/重庆.
  The canonical parse simply excluded them from the province set; Round 22 adds them back.

## 3. What changes vs canonical (and what does not)

Changed:
1. GAT panel extended 27 → 31 units; the 4 component z-scores are recomputed on the
   **full 31-unit panel** (ddof=0), so GAT_composite values for the original 27 units
   also shift slightly. GAT for municipalities is NOT a naive append of 27-unit z-values.
2. TF-IDF tourism weight recomputed on 31 units (same TfidfVectorizer settings).
3. End-2011 5A stock adds 北京/天津/上海/重庆 (same "first designation year ≤ 2011" rule).
4. REGION map gains 北京→North, 天津→North, 上海→East, 重庆→Southwest
   (standard NBS regional classification).
5. `em_baseline` / maturity splits are recomputed on the 31-unit outcome; because all 4
   municipalities are high-maturity markets, the median shifts up and some original units
   may be reclassified. This is a **sample-dependent consequence**, reported as-is.

Unchanged (frozen, integrity-first):
- TOURISM_LEXICON (37 terms) — identical.
- GAT computation formula (lex_density / seg_ratio / char_density / term_breadth).
- Two-way FE specification; exact (province, year−k) calendar joins for lagged GAT;
  no row-position shift/diff for dynamic variables.
- Ningxia 2011 official correction (6.20) and all pre-existing outcome cells.
- Human coding coverage: the stratified human-validation sample was drawn from the 27-unit
  corpus only; it does **not** cover municipalities, and this limitation will be stated.

## 4. Judgment rules (honest reporting)

- Every number is written to `results/round22_31province_*.json` / `*_31.csv` before any
  prose references it. No hard-coded figures.
- Results are reported whether or not they match the 27-unit canonical. Null, adverse, and
  sign-flip outcomes are retained verbatim.
- No dictionary / classification / control / sample is modified to obtain significance.
- If municipalities cannot enter any analysis (e.g., missing input), that analysis is
  reported as FAIL with the reason, never silently dropped.

## 5. Planned outputs

- `results/provincial_gat_panel_31.csv`
- `results/provincial_gat_tfidf_31.csv`
- `results/provincial_5a_stock_official_31.csv`
- `results/merged_panel_calendar_safe_31.csv`
- `results/round22_31province.json`
- `results/round22_31province_dictionary_sensitivity.csv`
- `results/round22_31province_outcome_cell_audit.csv`
- `figures/fig3_lag_heterogeneity_coefficients_31.png`

## 6. Comparison basis

Canonical 27-unit results are those in `results/round21_calendar_integrity.json`
(winsorised / WCR / dictionary / outcome sensitivity as documented in iterMemo).
