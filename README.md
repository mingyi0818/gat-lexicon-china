# GAT Lexicon China

Replication package for the paper:

> Government Attention to Tourism: A Transparent Lexicon Index and a Cautionary
> Assessment of Inbound Tourism Growth in China

The paper builds a 37-term lexicon index of Government Attention to Tourism
(GAT) from 671 provincial government work reports (2003–2024), links it to
international tourism foreign-exchange receipts for 31 Chinese provinces over
2003–2019 with exact calendar-year lags, and estimates two-way fixed-effects
timing models under conservative few-cluster inference. The pooled estimates
are jointly null; an exploratory lower-receipt interaction fails the wild
cluster bootstrap, Holm adjustment, and every influence screen.

## Repository layout

```
code/          analysis and figure scripts (Python)
experiments/   preregistration notes, written before each remediation round
results/       derived panels and archived outputs (JSON)
```

## Data sources

| Input | Source |
|---|---|
| Provincial work reports | The open lawtext/gov corpus (not redistributed here; 340 MB). Set `GAT_CORPUS` to the directory containing `政府工作报告-省自治区` and `政府工作报告-主要城市`. |
| Inbound receipts (US$ million), GDP, arrivals | China Statistical Yearbook / China Tourism Statistical Yearbook (NBS, 2012), provincial pages. |
| 5A scenic-area ratings | Ministry of Culture and Tourism list, archived 2025-03-11. |
| Comparison GAT index | Public portal released with Yang et al. (2026). |

## Reproduction

**Level 1 — tables and statistics from the archived panels** (no external
downloads needed; the derived panels in `results/` are committed):

```bash
python code/round22_analysis_31province.py     # H1/H2/H3, WCR, sensitivity family
python code/round23_31province_extras.py       # multi-outcome, MDE, validation tables
python code/round35_fig12_colour.py            # Figures 1-2
```

**Level 2 — full rebuild from the corpus** (requires the lawtext/gov
download):

```bash
export GAT_CORPUS=/path/to/lawtext_gov
python code/round22_build_31province.py
```

Environment: Python 3.13 with pandas, numpy, statsmodels, scikit-learn,
matplotlib, jieba, beautifulsoup4. The WCR bootstrap uses 9,999 replications
and the figure scripts write into `figures/` (created on demand).

The 27-province baseline (before the municipality extension) is reproduced by
`code/round21_calendar_integrity.py`; its archived outputs are
`results/round21_calendar_integrity.json` and
`results/merged_panel_calendar_safe.csv` (the latter is not committed; the
31-province panel supersedes it for all reported estimates).

## Script-to-exhibit map

| Manuscript item | Script | Archived output |
|---|---|---|
| Table 1 (summary statistics) | round22_analysis_31province.py | round22_31province.json |
| Table 2 (multi-outcome triangulation) | round23_31province_extras.py | round23_31province_extras.json |
| Table 3 (panel FE regressions) | round22_analysis_31province.py | round22_31province.json |
| Table 4 (interaction robustness family) | round22_analysis_31province.py | round22_31province.json |
| Table 5 (external-criteria correlations) | round23_31province_extras.py | round23_31province_extras.json |
| Table 6 (index-to-index convergence) | round23_31province_extras.py | round23_31province_extras.json |
| Table 7 (human coding) | manual coding, adjudicated | human_coding_agreement_20261117.json, human_coding_adjudicated_20261117.json |
| Figure 1 (provincial GAT trends) | round35_fig12_colour.py | round35_fig12_colour.json |
| Figure 2 (GAT heatmap) | round35_fig12_colour.py | round35_fig12_colour.json |
| Figure 3 (lag estimates, classification sensitivity) | round22_analysis_31province.py | round22_31province.json |
| Figure 4 (between/within decomposition) | round23_31province_extras.py | round23_31province_extras.json |
| MDE calculations (Sections 5, 6.1) | round23_31province_extras.py | round23_31province_extras.json |

## Notes on design decisions

`experiments/` holds the preregistration note for each remediation round,
written before the corresponding script was run. Two design points documented
there matter for interpretation:

- All lags are joined by exact `(province, year-k)` keys. Earlier drafts used
  row-position shifts, which misdate lags when a province-year report is
  missing; those estimates are superseded.
- The TF–IDF layer is estimated on the full 815-document corpus (1979–2026)
  while the composite index exists on the 671 units of 2003–2024. Section 6.4
  of the paper reports the window-restricted re-estimation.

## License

MIT — see `LICENSE`.

## Citation

If you use the lexicon or the replication code, please cite the paper and
this repository (citation details are provided in the paper).
