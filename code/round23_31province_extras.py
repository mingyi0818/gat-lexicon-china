"""Supplementary 31-province analyses reported alongside the main estimates.

Companion to round22_analysis_31province.py. Computes everything the manuscript
reports that round22 does not: multi-outcome triangulation, MDE, classification
thresholds, between-province validation (Table 4), Yang et al. index-to-index
convergence (Table 5), between/within decomposition (Figure 4), domestic-revenue
timing, 5A stock agreement, and manuscript-facing extractions from the round22
JSON. All results are written to results/round23_31province_extras.json before
any prose references them.

Preregistered under experiments/prereg_round22_31province_20261122.md (same
judgment rules: honest reporting, no tuning for significance).
"""
# 补充分析都在这个脚本里：三结果三角、收敛效度、省间/省内分解、回溯预测。
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 22,
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PANEL = RESULTS / "merged_panel_calendar_safe_31.csv"
GAT31 = RESULTS / "provincial_gat_panel_31.csv"
R22 = json.loads((RESULTS / "round22_31province.json").read_text(encoding="utf-8"))

EN = {
    "安徽": "Anhui", "福建": "Fujian", "甘肃": "Gansu", "广东": "Guangdong",
    "广西": "Guangxi", "贵州": "Guizhou", "海南": "Hainan", "河北": "Hebei",
    "河南": "Henan", "黑龙江": "Heilongjiang", "湖北": "Hubei", "湖南": "Hunan",
    "吉林": "Jilin", "江苏": "Jiangsu", "江西": "Jiangxi", "辽宁": "Liaoning",
    "内蒙古": "Inner Mongolia", "宁夏": "Ningxia", "青海": "Qinghai", "山东": "Shandong",
    "山西": "Shanxi", "陕西": "Shaanxi", "四川": "Sichuan", "西藏": "Xizang",
    "新疆": "Xinjiang", "云南": "Yunnan", "浙江": "Zhejiang",
    "北京": "Beijing", "天津": "Tianjin", "上海": "Shanghai", "重庆": "Chongqing",
}

out: dict = {}


# 双向去均值，跟主分析同一套写法，保证数字能对上。
def two_way_demean(df, cols, entity="province", time="year", tol=1e-13, maxit=5000):
    m = df[cols].astype(float).copy()
    for _ in range(maxit):
        prev = m.to_numpy().copy()
        m = m - m.groupby(df[entity]).transform("mean")
        m = m - m.groupby(df[time]).transform("mean")
        if np.max(np.abs(m.to_numpy() - prev)) < tol:
            break
    return m


def twfe(panel, dep, indep, entity="province"):
    sub = panel.dropna(subset=[dep] + indep + [entity, "year"]).copy()
    if len(sub) < 30:
        return None, sub
    dm = two_way_demean(sub, [dep] + indep)
    keep = [c for c in indep if dm[c].std(ddof=0) > 1e-12]
    if not keep:
        return None, sub
    m = sm.OLS(dm[dep], sm.add_constant(dm[keep], has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": sub[entity]}
    )
    return m, sub


# 日历滞后，老规矩：按（省, 年份-k）取，不按行号 shift。
def calendar_lag(df, col, k):
    s = df.set_index(["province", "year"])[col]
    idx = pd.MultiIndex.from_arrays([df["province"], df["year"] - k])
    return idx.map(s)


# Fisher z 变换算相关系数的置信区间，小样本比正态近似稳一些。
def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)
    return [float(lo), float(hi)]


# 偏相关：先把控制变量回归掉，再算残差之间的相关。
def partial_corr(x, y, c):
    df = pd.concat([x, y, c], axis=1).dropna()
    rx = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 2])).fit().resid
    ry = sm.OLS(df.iloc[:, 1], sm.add_constant(df.iloc[:, 2])).fit().resid
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def main():
    df = pd.read_csv(PANEL)
    df = df.sort_values(["province", "year"]).copy()

    # ---------- 1. corpus and panel counts ----------
    gat = pd.read_csv(GAT31)
    win_reports = gat[gat["year"].between(2003, 2019)]
    out["counts"] = {
        "corpus_total_documents_31units": 815,
        "corpus_27units_all_years": 714,
        "corpus_municipal_all_years": 101,
        "index_construction_units_2003_2024": int(len(gat)),
        "reports_in_estimation_window_2003_2019": int(len(win_reports)),
        "calendar_cells_31x17": 527,
        "calendar_cells_without_report": int(527 - len(win_reports)),
        "level_estimation_sample": int(
            df.dropna(subset=["ln_tourism", "GAT_composite", "ln_gdp"]).shape[0]
        ),
        "municipal_coverage_note": (
            "Beijing 1998-2026 (29 docs); Tianjin/Shanghai/Chongqing 2003-2026 "
            "(24 docs each); all four fully cover 2003-2024."
        ),
    }

        # 换两个结果（入境人次、住宿餐饮业 GDP）再跑同一套时序，看零结果是不是只在收入上出现
    # ---------- 2. multi-outcome triangulation ----------
    df["ln_arrivals"] = np.log(df["inbound_tourist_arrivals_10k"].where(df["inbound_tourist_arrivals_10k"] > 0))
    df["ln_accgdp"] = np.log(df["accommodation_catering_gdp_cny_100m"].where(df["accommodation_catering_gdp_cny_100m"] > 0))
    for col, dcol in [("ln_arrivals", "dln_arrivals"), ("ln_accgdp", "dln_accgdp")]:
        df[f"{col}_L1"] = calendar_lag(df, col, 1)
        df[dcol] = df[col] - df[f"{col}_L1"]
    df["dln_receipts"] = df["tourism_growth"]

    multi_rows, joints = [], {}
    for dep, olab in [("dln_receipts", "receipts_growth"),
                      ("dln_arrivals", "arrivals_growth"),
                      ("dln_accgdp", "accgdp_growth")]:
        cells = []
        for k in (0, 1, 2):
            x = "GAT_L0" if k == 0 else f"GAT_L{k}"
            m, sub = twfe(df, dep, [x])
            cells.append({
                "spec": f"GAT_k{k}",
                "coef": None if m is None else float(m.params[x]),
                "se": None if m is None else float(m.bse[x]),
                "p": None if m is None else float(m.pvalues[x]),
                "nobs": 0 if m is None else int(m.nobs),
                "n_clusters": int(sub["province"].nunique()),
            })
        multi_rows.append({"outcome": olab, "cells": cells})
        jm, jsub = twfe(df, dep, ["GAT_L0", "GAT_L1", "GAT_L2"])
        R = np.eye(len(jm.params))[1:, :]
        w = jm.wald_test(R)
        joints[olab] = {
            "joint_stat": float(np.asarray(w.statistic).reshape(-1)[0]),
            "joint_p": float(np.asarray(w.pvalue).reshape(-1)[0]),
            "nobs": int(jm.nobs),
            "n_clusters": int(jsub["province"].nunique()),
            "coefs": {n: float(jm.params[n]) for n in ["GAT_L0", "GAT_L1", "GAT_L2"]},
        }
    ps = [c["p"] for r in multi_rows for c in r["cells"] if c["p"] is not None]
    order = np.argsort(ps)
    running, holm = 0.0, [0.0] * len(ps)
    for rank, idx in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[idx])
        holm[idx] = min(running, 1.0)
    i = 0
    for r in multi_rows:
        for c in r["cells"]:
            if c["p"] is not None:
                c["p_holm9"] = float(holm[i]); i += 1
    out["multi_outcome"] = {"rows": multi_rows, "joint_wald": joints}

    # ---------- 3. MDE (80% power, two-sided 5%, df = G-1 = 30) ----------
    h1 = R22["h1"]["separate_models"]
    mde = {}
    for lag in ("lag0", "lag1", "lag2"):
        se = h1[lag]["focal"]["se"]
        G = h1[lag]["n_provinces"]
        tc = float(stats.t.ppf(0.975, G - 1) + stats.t.ppf(0.80, G - 1))
        mde[lag] = {"se": float(se), "n_clusters": int(G), "mde": float(tc * se)}
    out["mde_timing"] = mde

    # ---------- 4. classification details ----------
    maturity = df.groupby("province")["ln_tourism"].mean()
    early = df[df["year"].between(2003, 2007)].groupby("province")["ln_tourism"].mean()
    arrivals_early = df[df["year"].between(2003, 2007)].groupby("province")["inbound_tourists"].mean()
    stock = pd.read_csv(RESULTS / "provincial_5a_stock_official_31.csv")
    third = pd.read_csv(RESULTS / "official_vs_third_party_5a_2011.csv")
    n_lower = int((df.groupby("province")["em_baseline"].first() == 1).sum())
    n_t33 = int((df.groupby("province")["em_t33"].first() == 1).sum())
    n_early = int((df.groupby("province")["em_early_receipts"].first() == 1).sum())
    n_arr = int((df.groupby("province")["em_early_arrivals"].first() == 1).sum())
    n_5a = int((stock["n_5a_2011_official"] < stock["n_5a_2011_official"].median()).sum())
    lower_obs = int(df["em_baseline"].sum())
    agree = int((third["difference"] == 0).sum())
    out["classification"] = {
        "baseline_median_threshold_ln": float(maturity.median()),
        "baseline_n_lower": n_lower,
        "baseline_lower_obs": lower_obs,
        "t33_threshold_ln": float(maturity.quantile(0.33)),
        "t33_n_lower": n_t33,
        "early_threshold_ln": float(early.median()),
        "early_n_lower_of": int(early.notna().sum()),
        "early_n_lower": n_early,
        "arrivals_tercile_n": n_arr,
        "arrivals_tercile_threshold": float(arrivals_early.quantile(1 / 3)),
        "five_a_median": float(stock["n_5a_2011_official"].median()),
        "five_a_n_below": n_5a,
        "five_a_municipal_stock": {
            p: int(v) for p, v in stock.set_index("province")["n_5a_2011_official"]
            .reindex(["北京", "天津", "上海", "重庆"]).items()
        },
        "five_a_agreement_official_vs_third_party": {
            "agree_of": int(len(third)), "agree": agree,
            "note": "third-party stock exists for the 27 provincial units only",
        },
    }

        # 省均值之间的相关，只能算效标验证的一半，负相关也别急着解释成政策反向
    # ---------- 5. Table 4: between-province validation correlations ----------
    s = df.dropna(subset=["ln_tourism", "GAT_composite", "ln_gdp"]).copy()
    pm = s.groupby("province").agg(
        GAT=("GAT_composite", "mean"),
        two_layer=("GAT_two_layer", "mean"),
        ln_arr=("ln_arrivals", "mean"),
        ln_acc=("ln_accgdp", "mean"),
        ln_gdp=("ln_gdp", "mean"),
    ).dropna()
    table4 = {}
    for measure in ("GAT", "two_layer"):
        for crit in ("ln_arr", "ln_acc"):
            sub = pm[[measure, crit, "ln_gdp"]].dropna()
            r, p = stats.pearsonr(sub[measure], sub[crit])
            table4[f"{measure}_vs_{crit}"] = {
                "n": int(len(sub)), "r": float(r), "p": float(p),
            }
    out["table4_between_validation"] = table4

        # 跟外部 LLM 指数对齐；相关高不高先看剔掉海南还剩多少
    # ---------- 6. Table 5: Yang et al. index-to-index ----------
    y = pd.read_csv(RESULTS / "yang2026_gat_database_raw.csv")
    sys.path.insert(0, str(ROOT))
    from utils import normalize_province
    yprov = y[y["level"] == "province"].copy()
    yprov["prov"] = yprov["region"].map(normalize_province)
    yprov = yprov.dropna(subset=["prov"])
    our_full = gat[gat["year"].between(2003, 2024)].groupby("province")["GAT_composite"].mean()
    our_win = gat[gat["year"].between(2003, 2019)].groupby("province")["GAT_composite"].mean()
    tf = pd.read_csv(RESULTS / "provincial_gat_tfidf_31.csv")
    tf_win = tf[tf["year"].between(2003, 2019)]
    zt = (tf_win["GAT_tfidf"] - tf_win["GAT_tfidf"].mean()) / tf_win["GAT_tfidf"].std(ddof=1)
    tf_win = tf_win.assign(z_tfidf=zt)
    two_win = tf_win.merge(gat[["province", "year", "GAT_composite"]], on=["province", "year"])
    two_win["two"] = (two_win["GAT_composite"] + two_win["z_tfidf"]) / 2
    our_two = two_win.groupby("province")["two"].mean()

    yang_full = yprov[yprov["year"].between(2002, 2023)].groupby("prov")["GAT"].mean()
    yang_win = yprov[yprov["year"].between(2003, 2019)].groupby("prov")["GAT"].mean()
    gdp_mean = s.groupby("province")["ln_gdp"].mean()

    table5 = {}
    for ylab, yser in [("full_2002_2023", yang_full), ("window_2003_2019", yang_win)]:
        for mlab, mser in [("composite", our_full if ylab.startswith("full") else our_win),
                           ("two_layer", our_two)]:
            d = pd.concat([yser, mser, gdp_mean], axis=1, keys=["yang", "ours", "gdp"]).dropna()
            r, p = stats.pearsonr(d["yang"], d["ours"])
            rho, rp = stats.spearmanr(d["yang"], d["ours"])
            pr, pp = partial_corr(d["yang"], d["ours"], d["gdp"])
            table5[f"{ylab}_{mlab}"] = {
                "n": int(len(d)), "pearson_r": float(r), "pearson_p": float(p),
                "pearson_ci95": fisher_ci(r, len(d)),
                "spearman_rho": float(rho), "spearman_p": float(rp),
                "partial_r_given_ln_gdp": float(pr), "partial_p": float(pp),
            }
    # Hainan exclusion and LOO range on the full-window composite
    d = pd.concat([yang_full, our_full, gdp_mean], axis=1, keys=["yang", "ours", "gdp"]).dropna()
    dh = d.drop(index="海南", errors="ignore")
    r_h, p_h = stats.pearsonr(dh["yang"], dh["ours"])
    rho_h, _ = stats.spearmanr(dh["yang"], dh["ours"])
    loo = []
    for prov in d.index:
        dd = d.drop(index=prov)
        rr, _ = stats.pearsonr(dd["yang"], dd["ours"])
        loo.append({"province": prov, "r": float(rr)})
    d2 = pd.concat([yang_full, our_two, gdp_mean], axis=1, keys=["yang", "ours", "gdp"]).dropna()
    r_h2, p_h2 = stats.pearsonr(d2.drop(index="海南")["yang"], d2.drop(index="海南")["ours"])
    out["table5_yang_convergence"] = {
        "cells": table5,
        "matched_units": int(len(d)),
        "yang_provincial_units": int(yprov["prov"].nunique()),
        "hainan_excluded": {
            "composite_r": float(r_h), "composite_p": float(p_h),
            "composite_ci95": fisher_ci(r_h, len(dh)),
            "composite_spearman": float(rho_h), "n": int(len(dh)),
            "two_layer_r": float(r_h2), "two_layer_p": float(p_h2),
        },
        "loo_pearson_range": {
            "min": float(min(x["r"] for x in loo)),
            "max": float(max(x["r"] for x in loo)),
        },
    }

        # 图 4：省间那条线是省均值回归，省内是去均值以后再看，两个符号可以不一样
    # ---------- 7. Figure 4: between/within decomposition ----------
    r_raw, p_raw = stats.pearsonr(pm["GAT"], pm["ln_receipts".replace("ln_receipts", "GAT")]) if False else (None, None)
    lv = df.dropna(subset=["ln_tourism", "GAT_composite", "ln_gdp"]).copy()
    pm4 = lv.groupby("province").agg(
        GAT=("GAT_composite", "mean"), ln_rec=("ln_tourism", "mean"), ln_gdp=("ln_gdp", "mean")
    ).reset_index()
    r_between, p_between = stats.pearsonr(pm4["GAT"], pm4["ln_rec"])
    mb = sm.OLS(pm4["ln_rec"], sm.add_constant(pm4[["GAT", "ln_gdp"]])).fit(cov_type="HC1")
    cooks = mb.get_influence().cooks_distance[0]
    cook_top = sorted(
        [{"province": EN.get(p, p), "cooks_d": float(c)} for p, c in zip(pm4["province"], cooks)],
        key=lambda x: -x["cooks_d"],
    )[:3]
    subw = lv.copy()
    for col in ["ln_tourism", "GAT_composite", "ln_gdp"]:
        subw[f"{col}_w"] = subw[col] - subw.groupby("province")[col].transform("mean")
    mw = sm.OLS(subw["ln_tourism_w"], sm.add_constant(subw[["GAT_composite_w", "ln_gdp_w"]])).fit(
        cov_type="cluster", cov_kwds={"groups": subw["province"]}
    )
    sd_gat = float(lv["GAT_composite"].std(ddof=1))
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5))
    sns.regplot(data=pm4, x="GAT", y="ln_rec", ax=axes[0], color="black",
                scatter_kws={"s": 75, "facecolors": "white", "edgecolors": "black"},
                line_kws={"color": "black", "linestyle": "--"})
    axes[0].set_xlabel("Province-mean GAT composite")
    axes[0].set_ylabel("Province-mean log inbound receipts")
    axes[0].set_title(f"Between-province association\n(r = {r_between:.3f}, p = {p_between:.3f})")
    coef_b, coef_w = float(mb.params["GAT"]), float(mw.params["GAT_composite_w"])
    axes[1].bar(["Between", "Within"], [coef_b, coef_w], color="white",
                edgecolor="black", hatch=["//", "\\\\"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("GAT composite coefficient")
    axes[1].set_title("GAT coefficient on log inbound receipts")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_between_within_31.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out["figure4_between_within"] = {
        "n_provinces": int(len(pm4)),
        "raw_between_r": float(r_between), "raw_between_p": float(p_between),
        "between_coef_hc1": coef_b, "between_p_hc1": float(mb.pvalues["GAT"]),
        "between_per_sd": coef_b * sd_gat,
        "within_coef_cluster": coef_w, "within_p_cluster": float(mw.pvalues["GAT_composite_w"]),
        "within_per_sd": coef_w * sd_gat,
        "gat_sd": sd_gat,
        "cooks_top3": cook_top,
        "figure": "figures/fig4_between_within_31.png",
    }

    # ---------- 8. domestic revenue timing (bulletin-parsed provinces only) ----------
    dom_src = pd.read_csv(RESULTS / "merged_panel_corrected_with_domestic.csv")
    dom = dom_src.loc[
        dom_src["parse_status"].eq("ok") & dom_src["domestic_tourism_revenue_yi"].gt(0),
        ["year", "province", "domestic_tourism_revenue_yi"],
    ].copy()
    dd = df.merge(dom, on=["year", "province"], how="left").sort_values(["province", "year"])
    dd["ln_dom"] = np.log(dd["domestic_tourism_revenue_yi"])
    dd["ln_dom_lag"] = dd.groupby("province")["ln_dom"].shift(1)
    dd["year_lag"] = dd.groupby("province")["year"].shift(1)
    dd["dln_dom"] = np.where(dd["year"] - dd["year_lag"] == 1, dd["ln_dom"] - dd["ln_dom_lag"], np.nan)
    dom_rows = []
    for k, col in [("k0", "GAT_L0"), ("k1", "GAT_L1"), ("k2", "GAT_L2")]:
        m, sub = twfe(dd, "dln_dom", [col])
        dom_rows.append({
            "spec": k, "coef": float(m.params[col]), "se": float(m.bse[col]),
            "p": float(m.pvalues[col]), "n": int(m.nobs),
            "clusters": int(sub["province"].nunique()),
        })
    jm, jsub = twfe(dd, "dln_dom", ["GAT_L0", "GAT_L1", "GAT_L2"])
    R = np.zeros((3, len(jm.params)))
    names = list(jm.params.index)
    for i, name in enumerate(["GAT_L0", "GAT_L1", "GAT_L2"]):
        R[i, names.index(name)] = 1
    w = jm.wald_test(R, scalar=True)
    G = int(jsub["province"].nunique())
    tc = float(stats.t.ppf(0.975, G - 1) + stats.t.ppf(0.80, G - 1))
    out["domestic_revenue_timing"] = {
        "coverage_note": (
            "Provincial statistical bulletins parsed successfully for 26 units; "
            "the four municipalities are not in the parsed bulletin set, so this "
            "check runs on the 26 covered units with 31-province GAT exposures."
        ),
        "n_cells_ok": int(dd["domestic_tourism_revenue_yi"].notna().sum()),
        "separate": dom_rows,
        "joint": {
            "wald_stat": float(np.asarray(w.statistic).reshape(-1)[0]),
            "wald_p": float(np.asarray(w.pvalue).reshape(-1)[0]),
            "n": int(jm.nobs), "clusters": G,
        },
        "mean_growth": float(dd["dln_dom"].mean()),
        "mde_one_year": float(tc * dom_rows[1]["se"]),
    }

    # ---------- 9. extractions from round22 JSON (manuscript-facing) ----------
    h2b = R22["h2"]["specifications"]["baseline"]
    conv = R22["h2"]["specifications"]["convergence_control"]
    out["extracts"] = {
        "h2_baseline": {
            "reference_coef": float(h2b["params"]["GAT_L1"]),
            "reference_p": float(h2b["pvalues"]["GAT_L1"]),
            "interaction": h2b["focal"],
            "total_lower_effect": h2b["total_lower_effect"],
            "r2_within": float(h2b["r2_within"]),
            "nobs": int(h2b["nobs"]),
        },
        "convergence_lagged_level": {
            "coef": float(conv["params"]["ln_tourism_L1"]),
            "p": float(conv["pvalues"]["ln_tourism_L1"]),
        },
        "h1_joint": {
            "coefs": {k: float(v) for k, v in R22["h1"]["joint_model"]["params"].items()},
            "nobs": int(R22["h1"]["joint_model"]["nobs"]),
            "wald_stat": float(R22["h1"]["wald"]["all_three_zero"]["statistic"]),
            "wald_p": float(R22["h1"]["wald"]["all_three_zero"]["p"]),
            "eq_01_p": float(R22["h1"]["wald"]["lag0_equals_lag1"]["p"]),
            "eq_02_p": float(R22["h1"]["wald"]["lag0_equals_lag2"]["p"]),
        },
        "level_model": {
            lag: {
                "coef": float(R22["level_model"][lag]["params"]["GAT_L0" if lag == "lag0" else ("GAT_L1" if lag == "lag1" else "GAT_L2")]),
                "p": float(R22["level_model"][lag]["pvalues"]["GAT_L0" if lag == "lag0" else ("GAT_L1" if lag == "lag1" else "GAT_L2")]),
                "nobs": int(R22["level_model"][lag]["nobs"]),
                "r2_within": float(R22["level_model"][lag]["r2_within"]),
            } for lag in ("lag0", "lag1", "lag2")
        },
        "region_cluster_p": {
            k: float(v["focal"]["p"]) for k, v in R22["h2"]["region_cluster"].items()
        },
        "winsor_bounds": R22["outcome_sensitivity"]["winsor_bounds"],
        "wcr": {"t_obs": R22["wcr"]["webb"]["t_obs"], "webb_p": R22["wcr"]["webb"]["p_wcr"],
                "rademacher_p": R22["wcr"]["rademacher"]["p_wcr"]},
        "winsor_focal": R22["outcome_sensitivity"]["winsor"]["focal"],
        "winsor_wcr_p": R22["outcome_sensitivity"]["winsor"]["wcr_webb"]["p_wcr"],
        "dictionary_variants": {
            k: {"coef": v["focal"]["coef"], "se": v["focal"]["se"], "p": v["focal"]["p"],
                "n_terms": v["n_terms"]}
            for k, v in R22["dictionary"]["variants"].items()
        },
        "dictionary_two_layer_window_scaled": R22["dictionary"]["two_layer_window_scaled"]["focal"],
        "h3": {
            "long_run_coef": float(R22["h3"]["long_run"]["params_hc3"]["maturity_long"]),
            "long_run_se": float(R22["h3"]["long_run"]["bse_hc3"]["maturity_long"]),
            "long_run_p": float(R22["h3"]["long_run"]["pvalues_hc3"]["maturity_long"]),
            "long_run_n": int(R22["h3"]["long_run"]["n_provinces"]),
            "early_coef": float(R22["h3"]["early_period"]["params_hc3"]["maturity_early"]),
            "early_se": float(R22["h3"]["early_period"]["bse_hc3"]["maturity_early"]),
            "early_p": float(R22["h3"]["early_period"]["pvalues_hc3"]["maturity_early"]),
            "early_n": int(R22["h3"]["early_period"]["n_provinces"]),
            "loo_min": float(R22["h3"]["long_run"]["loo_coef_min"]),
            "loo_max": float(R22["h3"]["long_run"]["loo_coef_max"]),
            "loo_negative": int(R22["h3"]["long_run"]["loo_negative_count"]),
        },
        "oos": R22["oos_prediction"]["variants"],
        "mirror_h2_baseline": R22["ningxia_2011_mirror_sensitivity"]["h2_baseline"]["focal"],
    }

    # which province drops from H3 early (N=30)?
    early_prov = set(early.index)
    all_prov = set(maturity.index)
    out["extracts"]["h3_early_missing_province"] = sorted(all_prov - early_prov)

    # ---------- 10. hybrid correlation and TF-IDF scale ----------
    hybrid_r = float(stats.pearsonr(lv["GAT_composite"], lv["GAT_half_scale_hybrid"])[0])
    out["measurement_scales"] = {
        "hybrid_vs_composite_r": hybrid_r,
        "tfidf_raw_sd_window": float(lv["GAT_tfidf"].std(ddof=1)),
        "tfidf_raw_mean_window": float(lv["GAT_tfidf"].mean()),
        "composite_sd": float(lv["GAT_composite"].std(ddof=1)),
    }

    # ---------- 11. extreme growth cells ----------
    ext = df[df["tourism_growth"].abs() > 0.70][["province", "year", "tourism_growth"]]
    ext["province_en"] = ext["province"].map(EN)
    out["extreme_growth_cells"] = {
        "n": int(len(ext)),
        "cells": ext.to_dict(orient="records"),
    }

    dest = RESULTS / "round23_31province_extras.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)
    print(json.dumps({
        "multi_joints": {k: round(v["joint_p"], 4) for k, v in joints.items()},
        "mde": {k: round(v["mde"], 4) for k, v in mde.items()},
        "table5_full_composite": {k: round(v, 4) for k, v in table5["full_2002_2023_composite"].items() if isinstance(v, (int, float))},
        "hainan_excluded_r": round(r_h, 4),
        "between_r": round(r_between, 4), "between_p": round(p_between, 4),
        "within_coef": round(coef_w, 4), "within_p": round(float(mw.pvalues["GAT_composite_w"]), 4),
        "extreme_n": len(ext),
        "h3_early_missing": out["extracts"]["h3_early_missing_province"],
    }, indent=2))


if __name__ == "__main__":
    main()
