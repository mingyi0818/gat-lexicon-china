"""Calendar-safe dynamic estimates and the official-cell correction.

Preregistered in experiments/prereg_round21_calendar_integrity.md.
All lagged variables are joined by exact province-year keys. No row-position
diff/shift is used for dynamic variables.
"""
# 这一版就干一件事：滞后按（省, 年份-k）精确取，绝不能拿上一行数据顶上。
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT / "code"))

import round20_integrity_remediation as r20  # noqa: E402
from round16_wcr_bootstrap import wcr_pvalue  # noqa: E402

SEED = 20261115
REGION = {
    "河北": "North", "山西": "North", "内蒙古": "North",
    "辽宁": "Northeast", "吉林": "Northeast", "黑龙江": "Northeast",
    "江苏": "East", "浙江": "East", "安徽": "East", "福建": "East",
    "江西": "East", "山东": "East",
    "河南": "Central", "湖北": "Central", "湖南": "Central",
    "广东": "South", "广西": "South", "海南": "South",
    "四川": "Southwest", "贵州": "Southwest", "云南": "Southwest", "西藏": "Southwest",
    "陕西": "Northwest", "甘肃": "Northwest", "青海": "Northwest",
    "宁夏": "Northwest", "新疆": "Northwest",
}


# 按日历键取滞后值；那一年没报告就留空，不要用相邻年份凑。
def exact_lag(source: pd.DataFrame, value: str, lag: int, output: str) -> pd.DataFrame:
    result = source[["province", "year", value]].copy()
    result["year"] = result["year"] + lag
    return result.rename(columns={value: output})


# 只有相邻两年都有数才算增长，中间断年直接置空，不然会凭空造出增长率。
def consecutive_change(frame: pd.DataFrame, value: str) -> pd.Series:
    previous_value = frame.groupby("province")[value].shift(1)
    previous_year = frame.groupby("province")["year"].shift(1)
    return (frame[value] - previous_value).where(frame["year"] - previous_year == 1)


# 面板拼装：指数、年鉴结果、5A 存量三块按省-年对齐，缺的留 NaN 不填。
def prepare_panel() -> tuple[pd.DataFrame, dict]:
    outcome = pd.read_csv(RESULTS / "provincial_inbound_receipts_panel.csv")
    gat = pd.read_csv(RESULTS / "provincial_gat_panel.csv")
    provinces = sorted(gat["province"].dropna().unique())
    outcome = outcome[
        outcome["province"].isin(provinces) & outcome["year"].between(2003, 2019)
    ].copy()
    outcome = outcome.sort_values(["province", "year"]).reset_index(drop=True)
    if len(outcome) != 27 * 17:
        raise RuntimeError(f"Expected complete 27x17 outcome panel, found {len(outcome)}")

    mirror_ningxia = float(outcome.loc[
        (outcome["province"] == "宁夏") & (outcome["year"] == 2011),
        "inbound_tourism_receipts_usd_million",
    ].iloc[0])
    outcome.loc[
        (outcome["province"] == "宁夏") & (outcome["year"] == 2011),
        "inbound_tourism_receipts_usd_million",
    ] = 6.20

    outcome["ln_tourism"] = np.log(outcome["inbound_tourism_receipts_usd_million"])
    outcome["ln_gdp"] = np.log(outcome["gdp_cny_100m"])
    outcome["tourism_growth"] = consecutive_change(outcome, "ln_tourism")
    outcome["dln_gdp"] = consecutive_change(outcome, "ln_gdp")
    outcome = outcome.merge(
        exact_lag(outcome, "ln_tourism", 1, "ln_tourism_L1"),
        on=["province", "year"], how="left", validate="one_to_one",
    )
    outcome["inbound_tourists"] = outcome["inbound_tourist_arrivals_10k"]

    gat_window = gat[gat["year"].between(2003, 2019)].copy()
    current_columns = [
        "province", "year", "GAT_composite", "lex_density", "seg_ratio",
        "char_density", "term_breadth",
    ]
    panel = outcome.merge(
        gat_window[current_columns], on=["province", "year"],
        how="left", validate="one_to_one",
    )
    panel = panel.merge(
        exact_lag(gat_window, "GAT_composite", 1, "GAT_L1"),
        on=["province", "year"], how="left", validate="one_to_one",
    ).merge(
        exact_lag(gat_window, "GAT_composite", 2, "GAT_L2"),
        on=["province", "year"], how="left", validate="one_to_one",
    )
    panel["GAT_L0"] = panel["GAT_composite"]

    maturity = outcome.groupby("province")["ln_tourism"].mean()
    panel["maturity_long"] = panel["province"].map(maturity)
    panel["em_baseline"] = (panel["maturity_long"] < maturity.median()).astype(float)
    panel["em_t33"] = (
        panel["maturity_long"] <= maturity.quantile(0.33)
    ).astype(float)
    early = outcome[outcome["year"].between(2003, 2007)]
    maturity_early = early.groupby("province")["ln_tourism"].mean()
    arrival_early = early.groupby("province")["inbound_tourists"].mean()
    panel["em_early_receipts"] = panel["province"].map(
        (maturity_early < maturity_early.median()).astype(float)
    )
    panel["em_early_arrivals"] = panel["province"].map(
        (arrival_early <= arrival_early.quantile(1 / 3)).astype(float)
    )

    stock = pd.read_csv(RESULTS / "provincial_5a_stock_official.csv").rename(
        columns={"n_5a_2011_official": "n_5a_2011"}
    )
    median_5a = float(stock["n_5a_2011"].median())
    stock["em_5a"] = (stock["n_5a_2011"] < median_5a).astype(float)
    panel = panel.merge(
        stock[["province", "n_5a_2011", "em_5a"]],
        on="province", how="left", validate="many_to_one",
    )
    panel["region"] = panel["province"].map(REGION)

    tfidf = pd.read_csv(RESULTS / "provincial_gat_tfidf.csv")
    report_measure = gat_window[["province", "year", "GAT_composite"]].merge(
        tfidf[["province", "year", "GAT_tfidf"]],
        on=["province", "year"], how="left", validate="one_to_one",
    )
    report_measure["z_tfidf"] = (
        report_measure["GAT_tfidf"] - report_measure["GAT_tfidf"].mean()
    ) / report_measure["GAT_tfidf"].std(ddof=1)
    report_measure["GAT_two_layer"] = (
        report_measure["GAT_composite"] + report_measure["z_tfidf"]
    ) / 2
    report_measure["GAT_half_scale_hybrid"] = (
        report_measure["GAT_composite"] + report_measure["GAT_tfidf"]
    ) / 2
    panel = panel.merge(
        report_measure[[
            "province", "year", "GAT_tfidf", "z_tfidf",
            "GAT_two_layer", "GAT_half_scale_hybrid",
        ]],
        on=["province", "year"], how="left", validate="one_to_one",
    )
    panel = panel.merge(
        exact_lag(report_measure, "GAT_two_layer", 1, "two_layer_L1"),
        on=["province", "year"], how="left", validate="one_to_one",
    ).merge(
        exact_lag(
            report_measure, "GAT_half_scale_hybrid", 1, "half_scale_hybrid_L1"
        ),
        on=["province", "year"], how="left", validate="one_to_one",
    )

    panel.to_csv(
        RESULTS / "merged_panel_calendar_safe.csv",
        index=False, encoding="utf-8-sig",
    )
    audit = {
        "panel_rows": int(len(panel)),
        "n_provinces": int(panel["province"].nunique()),
        "calendar_complete": bool(
            panel.groupby("province")["year"].nunique().eq(17).all()
        ),
        "mirror_ningxia_2011": mirror_ningxia,
        "official_ningxia_2011": 6.20,
        "official_source": "https://www.stats.gov.cn/sj/ndsj/2012/html/Q1823C.HTM",
        "dynamic_rule": "exact province-year joins; consecutive-year outcome changes only",
    }
    return panel, audit


# 双向固定效应用组内变换实现，比塞一堆虚拟变量省内存。
def twfe(panel, dependent, independent, cluster="province", mask=None):
    required = [dependent] + independent + ["province", "year", cluster]
    sub = panel.dropna(subset=required).copy()
    if mask is not None:
        sub = sub.loc[~mask.reindex(sub.index, fill_value=False)].copy()
    dm = r20.two_way_demean(sub, [dependent] + independent)
    model = sm.OLS(dm[dependent], sm.add_constant(dm[independent])).fit(
        cov_type="cluster", cov_kwds={"groups": sub[cluster]}
    )
    return model, sub


def block(model, sub, focal=None):
    return r20.model_block(model, sub, focal=focal)


def interaction(panel, exposure, moderator, controls=None, cluster="province", mask=None):
    controls = controls or []
    work = panel.copy()
    work["gat_x_lower"] = work[exposure] * work[moderator]
    indep = [exposure, "gat_x_lower"] + controls
    model, sub = twfe(work, "tourism_growth", indep, cluster=cluster, mask=mask)
    result = block(model, sub, focal="gat_x_lower")
    names = list(model.params.index)
    contrast = np.zeros((1, len(names)))
    contrast[0, names.index(exposure)] = 1
    contrast[0, names.index("gat_x_lower")] = 1
    test = model.t_test(contrast)
    result["total_lower_effect"] = {
        "coef": float(np.asarray(test.effect).item()),
        "se": float(np.asarray(test.sd).item()),
        "p": float(np.asarray(test.pvalue).item()),
    }
    return result


# Holm 逐步校正：六个设定算一个家族，先排 p 值再逐个收紧阈值。
def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * raw[key])
        adjusted[key] = min(running, 1.0)
    return adjusted


def run_h1(panel):
    separate = {}
    for lag in range(3):
        key = f"GAT_L{lag}"
        model, sub = twfe(panel, "tourism_growth", [key])
        separate[f"lag{lag}"] = block(model, sub, focal=key)
    joint, sub = twfe(panel, "tourism_growth", ["GAT_L0", "GAT_L1", "GAT_L2"])
    names = list(joint.params.index)
    restrictions = {}
    for label, pairs in {
        "lag0_equals_lag1": [("GAT_L0", 1), ("GAT_L1", -1)],
        "lag0_equals_lag2": [("GAT_L0", 1), ("GAT_L2", -1)],
    }.items():
        matrix = np.zeros((1, len(names)))
        for name, value in pairs:
            matrix[0, names.index(name)] = value
        restrictions[label] = r20.wald_result(joint, matrix)
    all_zero = np.zeros((3, len(names)))
    for row, name in enumerate(["GAT_L0", "GAT_L1", "GAT_L2"]):
        all_zero[row, names.index(name)] = 1
    restrictions["all_three_zero"] = r20.wald_result(joint, all_zero)
    return {
        "separate_models": separate,
        "joint_model": block(joint, sub),
        "wald": restrictions,
    }


def run_level(panel):
    results = {}
    for lag in range(3):
        exposure = f"GAT_L{lag}"
        model, sub = twfe(
            panel, "ln_tourism", [exposure, "ln_gdp"]
        )
        results[f"lag{lag}"] = block(model, sub, focal=exposure)
    return results


def run_h2(panel):
    specs = {
        "baseline": interaction(panel, "GAT_L1", "em_baseline"),
        "bottom_tercile": interaction(panel, "GAT_L1", "em_t33"),
        "early_receipts": interaction(panel, "GAT_L1", "em_early_receipts"),
        "gdp_growth_control": interaction(
            panel, "GAT_L1", "em_baseline", controls=["dln_gdp"]
        ),
        "convergence_control": interaction(
            panel, "GAT_L1", "em_baseline", controls=["ln_tourism_L1"]
        ),
        "equal_weight_two_layer": interaction(
            panel, "two_layer_L1", "em_baseline"
        ),
        "early_arrivals": interaction(panel, "GAT_L1", "em_early_arrivals"),
        "official_5a_full": interaction(panel, "GAT_L1", "em_5a"),
        "official_5a_2012_2019": interaction(
            panel, "GAT_L1", "em_5a", mask=panel["year"] < 2012
        ),
    }
    primary = [
        "baseline", "bottom_tercile", "early_receipts",
        "gdp_growth_control", "convergence_control", "equal_weight_two_layer",
    ]
    raw = {key: specs[key]["focal"]["p"] for key in primary}
    return {
        "specifications": specs,
        "primary_family": primary,
        "holm_raw": raw,
        "holm_adjusted": holm(raw),
        "region_cluster": {
            key: interaction(panel, "GAT_L1", moderator, cluster="region")
            for key, moderator in {
                "baseline": "em_baseline",
                "early_receipts": "em_early_receipts",
                "early_arrivals": "em_early_arrivals",
            }.items()
        },
    }


# 野聚类自助法，交互项的主推断用它，聚类稳健 p 值并排报出来做对照。
def run_wcr(panel):
    work = panel.copy()
    work["gat_x_lower"] = work["GAT_L1"] * work["em_baseline"]
    sub = work.dropna(
        subset=["tourism_growth", "GAT_L1", "gat_x_lower", "province", "year"]
    ).copy()
    dm = r20.two_way_demean(sub, ["tourism_growth", "GAT_L1", "gat_x_lower"])
    x = np.column_stack([
        np.ones(len(sub)), dm["GAT_L1"].to_numpy(), dm["gat_x_lower"].to_numpy()
    ])
    y = dm["tourism_growth"].to_numpy()
    groups = sub["province"].to_numpy()
    return {
        "webb": wcr_pvalue(x, y, groups, 2, B=9999, weights="webb", seed=SEED),
        "rademacher": wcr_pvalue(
            x, y, groups, 2, B=9999, weights="rademacher", seed=SEED
        ),
    }


def run_h3(panel):
    gat_mean = panel.groupby("province", as_index=False).agg(
        gat_mean=("GAT_composite", "mean")
    )
    outcome = panel.groupby("province", as_index=False).agg(
        maturity_long=("ln_tourism", "mean"), lngdp_long=("ln_gdp", "mean")
    )
    early = panel[panel["year"].between(2003, 2007)].groupby(
        "province", as_index=False
    ).agg(maturity_early=("ln_tourism", "mean"), lngdp_early=("ln_gdp", "mean"))
    cross = gat_mean.merge(outcome, on="province").merge(early, on="province")
    return {
        "long_run": r20.fit_h3_cross_section(cross, "maturity_long", "lngdp_long"),
        "early_period": r20.fit_h3_cross_section(
            cross, "maturity_early", "lngdp_early"
        ),
    }


def lagged_variant(panel, measure, value, output):
    return panel.merge(
        exact_lag(measure, value, 1, output),
        on=["province", "year"], how="left", validate="one_to_one",
    )


def run_dictionary(panel):
    cached, _ = r20.load_report_cache()
    variants = {
        "full_37_window_scaled": list(r20.TOURISM_LEXICON),
        "remove_ambiguous_reception_hotel_homestay": [
            term for term in r20.TOURISM_LEXICON
            if term not in {"接待", "酒店", "民宿"}
        ],
    }
    for group, terms in r20.TERM_GROUPS.items():
        variants[f"leave_out_{group}"] = [
            term for term in r20.TOURISM_LEXICON if term not in set(terms)
        ]
    rows, blocks = [], {}
    full_measure = None
    for label, terms in variants.items():
        measure = r20.metrics_for_terms(cached, terms)
        merged = lagged_variant(panel, measure, "GAT_variant", "variant_L1")
        result = interaction(merged, "variant_L1", "em_baseline")
        result["n_terms"] = len(terms)
        blocks[label] = result
        rows.append({
            "variant": label, "n_terms": len(terms),
            "coef": result["focal"]["coef"], "se": result["focal"]["se"],
            "p": result["focal"]["p"], "nobs": result["nobs"],
        })
        if label == "full_37_window_scaled":
            full_measure = measure[["province", "year", "GAT_variant"]]

    tfidf = pd.read_csv(RESULTS / "provincial_gat_tfidf.csv")
    two = full_measure.merge(
        tfidf[["province", "year", "GAT_tfidf"]],
        on=["province", "year"], how="left", validate="one_to_one",
    )
    two["z_tfidf"] = (
        two["GAT_tfidf"] - two["GAT_tfidf"].mean()
    ) / two["GAT_tfidf"].std(ddof=1)
    two["two_layer"] = (two["GAT_variant"] + two["z_tfidf"]) / 2
    merged = lagged_variant(panel, two, "two_layer", "two_layer_window_L1")
    scaled = interaction(merged, "two_layer_window_L1", "em_baseline")
    pd.DataFrame(rows).to_csv(
        RESULTS / "round21_dictionary_sensitivity.csv",
        index=False, encoding="utf-8-sig",
    )
    return {"variants": blocks, "two_layer_window_scaled": scaled}


def run_outcome_sensitivity(panel):
    low = (
        ((panel["province"] == "宁夏") & panel["year"].between(2003, 2012))
        | ((panel["province"] == "青海") & panel["year"].between(2003, 2004))
    )
    low_keys = panel.loc[low, ["province", "year"]].copy()
    low_keys["year"] += 1
    low_lag = pd.MultiIndex.from_frame(panel[["province", "year"]]).isin(
        pd.MultiIndex.from_frame(low_keys)
    )
    low_affected = low | low_lag
    extreme = panel["tourism_growth"].abs() > 0.70
    rows = panel.loc[low | extreme, [
        "province", "year", "inbound_tourism_receipts_usd_million",
        "tourism_growth",
    ]].copy()
    rows["flag_low"] = low[low | extreme].to_numpy()
    rows["flag_extreme"] = extreme[low | extreme].to_numpy()
    rows.to_csv(
        RESULTS / "round21_outcome_cell_audit.csv",
        index=False, encoding="utf-8-sig",
    )
    results = {}
    for label, mask in {
        "canonical": pd.Series(False, index=panel.index),
        "exclude_low": low_affected,
        "exclude_extreme": extreme,
        "exclude_union": low_affected | extreme,
    }.items():
        results[label] = interaction(
            panel, "GAT_L1", "em_baseline", mask=mask
        )
    work = panel.copy()
    valid = work["tourism_growth"].dropna()
    q01, q99 = valid.quantile([0.01, 0.99])
    work["growth_winsor"] = work["tourism_growth"].clip(q01, q99)
    work["gat_x_lower"] = work["GAT_L1"] * work["em_baseline"]
    model, sub = twfe(
        work, "growth_winsor", ["GAT_L1", "gat_x_lower"]
    )
    winsor = block(model, sub, focal="gat_x_lower")
    dm = r20.two_way_demean(
        sub, ["growth_winsor", "GAT_L1", "gat_x_lower"]
    )
    x = np.column_stack([
        np.ones(len(sub)), dm["GAT_L1"], dm["gat_x_lower"]
    ])
    winsor["wcr_webb"] = wcr_pvalue(
        x, dm["growth_winsor"].to_numpy(), sub["province"].to_numpy(),
        2, B=9999, weights="webb", seed=SEED,
    )
    return {
        "n_extreme": int(extreme.sum()),
        "specifications": results,
        "winsor_bounds": [float(q01), float(q99)],
        "winsor": winsor,
    }


def run_mirror_value_sensitivity(panel):
    """Retain the mirrored Ningxia 2011 value as a documented sensitivity."""
    mirror = panel.copy().sort_values(["province", "year"]).reset_index(drop=True)
    cell = (mirror["province"] == "宁夏") & (mirror["year"] == 2011)
    mirror.loc[cell, "inbound_tourism_receipts_usd_million"] = 6.00
    mirror.loc[cell, "ln_tourism"] = np.log(6.00)
    mirror["tourism_growth"] = consecutive_change(mirror, "ln_tourism")
    lagged = exact_lag(mirror, "ln_tourism", 1, "ln_tourism_L1_mirror")
    mirror = mirror.drop(columns=["ln_tourism_L1"]).merge(
        lagged, on=["province", "year"], how="left", validate="one_to_one"
    ).rename(columns={"ln_tourism_L1_mirror": "ln_tourism_L1"})
    maturity = mirror.groupby("province")["ln_tourism"].transform("mean")
    mirror["em_baseline"] = (maturity < maturity.groupby(mirror["province"]).first().median()).astype(float)
    return {
        "h1": run_h1(mirror),
        "h2_baseline": interaction(mirror, "GAT_L1", "em_baseline"),
        "guardrail": "Uses the mirrored 6.00 value only; canonical official value is 6.20.",
    }


def within_fit_predict(train, test, exposure=None):
    means = train.groupby("province")["tourism_growth"].mean()
    baseline = test["province"].map(means).fillna(train["tourism_growth"].mean())
    if exposure is None:
        return baseline.to_numpy()
    work = train.copy()
    work["yw"] = work["tourism_growth"] - work.groupby("province")[
        "tourism_growth"
    ].transform("mean")
    work["xw"] = work[exposure] - work.groupby("province")[exposure].transform("mean")
    denominator = float(np.dot(work["xw"], work["xw"]))
    beta = 0.0 if denominator == 0 else float(
        np.dot(work["xw"], work["yw"]) / denominator
    )
    return (baseline + beta * test[exposure]).to_numpy()


# 回溯式样本外：只用 t 年以前的数据估模型，再拿来预测第 t 年。
def run_oos(panel):
    output = {}
    for label, exposure in {
        "composite_lag": "GAT_L1",
        "half_scale_hybrid_lag": "half_scale_hybrid_L1",
        "two_layer_lag": "two_layer_L1",
    }.items():
        data = panel.dropna(subset=["tourism_growth", exposure]).copy()
        errors_no, errors_gat = [], []
        per_year = []
        for year in range(2010, 2020):
            train = data[data["year"] < year]
            test = data[data["year"] == year]
            y = test["tourism_growth"].to_numpy()
            pred0 = within_fit_predict(train, test)
            pred1 = within_fit_predict(train, test, exposure)
            e0, e1 = y - pred0, y - pred1
            errors_no.extend(e0)
            errors_gat.extend(e1)
            per_year.append({"year": year, "n": int(len(test))})
        e0, e1 = np.asarray(errors_no), np.asarray(errors_gat)
        loss_diff = e1 ** 2 - e0 ** 2
        test = stats.ttest_1samp(loss_diff, 0.0)
        output[label] = {
            "n_predictions": int(len(e0)),
            "rmse_no_gat": float(np.sqrt(np.mean(e0 ** 2))),
            "rmse_with_gat": float(np.sqrt(np.mean(e1 ** 2))),
            "mae_no_gat": float(np.mean(np.abs(e0))),
            "mae_with_gat": float(np.mean(np.abs(e1))),
            "dm_t": float(test.statistic),
            "dm_p": float(test.pvalue),
            "per_year": per_year,
        }
    return {
        "design": (
            "Rolling origin 2010-2019; exact calendar lag; paired common sample "
            "within each exposure; descriptive t-test on squared-error differences."
        ),
        "variants": output,
    }


def descriptives(panel):
    level = panel.dropna(
        subset=[
            "GAT_composite", "inbound_tourism_receipts_usd_million", "ln_gdp"
        ]
    ).copy()
    variables = {
        "GAT_composite": level["GAT_composite"],
        "GAT_half_scale_hybrid": level["GAT_half_scale_hybrid"],
        "GAT_two_layer": level["GAT_two_layer"],
        "receipts": level["inbound_tourism_receipts_usd_million"],
        "ln_gdp": level["ln_gdp"],
        "growth": panel["tourism_growth"].dropna(),
        "lower_receipt": level["em_baseline"],
    }
    return {
        key: {
            "n": int(series.count()), "mean": float(series.mean()),
            "sd": float(series.std(ddof=1)), "min": float(series.min()),
            "max": float(series.max()),
        }
        for key, series in variables.items()
    }


def plot_coefficients(h1, h2):
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 22,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    lag_rows = [h1["separate_models"][f"lag{x}"]["focal"] for x in range(3)]
    y = np.arange(3)
    axes[0].errorbar(
        [x["coef"] for x in lag_rows], y,
        xerr=[1.96 * x["se"] for x in lag_rows],
        fmt="o", color="black", capsize=6,
    )
    axes[0].set_yticks(y, ["Contemporaneous", "One-year lag", "Two-year lag"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Coefficient on pooled GAT")
    axes[0].set_title("(a) Pooled timing estimates")

    order = [
        ("baseline", "Full-sample receipt median"),
        ("bottom_tercile", "Receipt bottom tercile"),
        ("early_receipts", "2003–2007 receipt scale"),
        ("early_arrivals", "2003–2007 arrival scale"),
        ("official_5a_full", "Official end-2011 5A stock"),
        ("equal_weight_two_layer", "Equal-weight two-layer"),
    ]
    rows = [h2["specifications"][key]["focal"] for key, _ in order]
    y2 = np.arange(len(rows))
    axes[1].errorbar(
        [x["coef"] for x in rows], y2,
        xerr=[1.96 * x["se"] for x in rows],
        fmt="s", color="black", capsize=6,
    )
    axes[1].set_yticks(y2, [label for _, label in order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("GAT × lower-scale coefficient")
    axes[1].set_title("(b) Classification sensitivity")
    for axis in axes:
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.grid(axis="x", color="0.85")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "fig3_lag_heterogeneity_coefficients.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def main():
    panel, source_audit = prepare_panel()
    h1 = run_h1(panel)
    level = run_level(panel)
    h2 = run_h2(panel)
    wcr = run_wcr(panel)
    h3 = run_h3(panel)
    dictionary = run_dictionary(panel)
    outcome = run_outcome_sensitivity(panel)
    mirror_value = run_mirror_value_sensitivity(panel)
    oos = run_oos(panel)
    desc = descriptives(panel)
    plot_coefficients(h1, h2)
    output = {
        "note": "calendar-integrity correction; all results retained.",
        "prereg": "experiments/prereg_round21_calendar_integrity.md",
        "source_audit": source_audit,
        "descriptives": desc,
        "h1": h1,
        "level_model": level,
        "h2": h2,
        "wcr": wcr,
        "h3": h3,
        "dictionary": dictionary,
        "outcome_sensitivity": outcome,
        "ningxia_2011_mirror_sensitivity": mirror_value,
        "oos_prediction": oos,
        "figure": "figures/fig3_lag_heterogeneity_coefficients.png",
    }
    destination = RESULTS / "round21_calendar_integrity.json"
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("H1:")
    for key, value in h1["separate_models"].items():
        f = value["focal"]
        print(key, f["coef"], f["se"], f["p"], value["nobs"])
    print("H1 joint p", h1["wald"]["all_three_zero"]["p"])
    print("H2:")
    for key, value in h2["specifications"].items():
        f = value["focal"]
        print(key, f["coef"], f["se"], f["p"], value["nobs"])
    print("Holm", h2["holm_adjusted"])
    print("WCR", wcr)
    print("Wrote", destination)


if __name__ == "__main__":
    main()
