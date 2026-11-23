"""Round-R16 revision for Paper 07 (R15 reviewer Majors M1, M3, M5, M6).

Supersedes the wild-cluster-bootstrap block in 09_revision_analysis.py, which
was invalid: the bootstrap draws were generated from the *unrestricted* fit
(`yb = Xm @ beta + resid * wv`) and the bootstrap statistic was not centred
(`t_boot[b] = betab[2] / seb[2]`, no `- beta[2]`), so the bootstrap t
distribution was centred near t_obs and returned p ~= 0.5 mechanically.

This script implements a restricted wild cluster bootstrap (WCR):
  1. fit the restricted model (headline interaction dropped), keep resid_r;
  2. for each draw b: y_b = X_r @ beta_r + resid_r * w_g, w_g a cluster-level
     Rademacher (+-1) or Webb six-point weight;
  3. re-fit the full model, recompute the CR1 cluster-robust SE, form
     t_b = beta_b[j] / se_b[j];
  4. p = (1 + #{|t_b| >= |t_obs|}) / (B + 1).

The design is an exact two-way FE design obtained by iterative alternating
projections (the previous one-pass demeaning is exact only for balanced
panels; the growth sample is unbalanced). CR1 small-sample correction uses
G = number of clusters, N = obs, K = number of *non-absorbed* parameters, so
the reported t matches the manuscript's Table 3 ratio.

Also produced here:
  M5  headline growth estimate re-based on the composite index (with WCR p);
  M3  leave-Hainan-out / robust correlations for the external validation;
  M6  outcome-data audit of extreme province-years + winsorised re-estimation.

Outputs: results/round16_wcr_bootstrap.json, results/round16_outcome_audit.json,
         results/round16_leverage.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as st

RESULTS = str(Path(__file__).resolve().parent.parent / "results")
SEED = 20260930


# ---------------------------------------------------------------- design ----
def two_way_demean(df, cols, entity="province", time="year", tol=1e-13, maxit=5000):
    """Exact two-way within transformation via alternating projections."""
    out = df[cols].astype(float).copy()
    for _ in range(maxit):
        prev = out.values.copy()
        out = out - out.groupby(df[entity]).transform("mean")
        out = out - out.groupby(df[time]).transform("mean")
        if np.max(np.abs(out.values - prev)) < tol:
            break
    return out


def cr1_se(X, resid, groups, n_params):
    """Cluster-robust (CR1) standard errors. X: (N,K) numpy, groups: (N,)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    k = X.shape[1]
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        xg, eg = X[idx], resid[idx]
        s = xg.T @ eg
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    G = len(np.unique(groups))
    N = X.shape[0]
    c = (G / (G - 1.0)) * ((N - 1.0) / (N - n_params))
    return np.sqrt(np.diag(c * V))


def wcr_pvalue(X, y, groups, j, B=9999, weights="webb", seed=SEED, n_params=None):
    """Restricted wild cluster bootstrap p-value for coefficient j of X.

    Null imposed on column j (dropped in the restricted fit). Statistic is the
    centred CR1 t ratio; two-sided p.
    """
    N, K = X.shape
    if n_params is None:
        n_params = K
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = cr1_se(X, resid, groups, n_params)
    t_obs = beta[j] / se[j]

    keep = [c for c in range(K) if c != j]
    Xr = X[:, keep]
    beta_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
    resid_r = y - Xr @ beta_r

    uniq = np.unique(groups)
    G = len(uniq)
    gidx = {g: i for i, g in enumerate(uniq)}
    obs_g = np.array([gidx[g] for g in groups])

    rng = np.random.default_rng(seed)
    if weights == "rademacher":
        wmat = rng.choice([-1.0, 1.0], size=(B, G))
    elif weights == "webb":
        vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                         np.sqrt(0.5), 1.0, np.sqrt(1.5)])
        wmat = rng.choice(vals, size=(B, G))
    else:
        raise ValueError(weights)

    t_boot = np.empty(B)
    for b in range(B):
        wv = wmat[b][obs_g]
        yb = Xr @ beta_r + resid_r * wv
        bb = np.linalg.lstsq(X, yb, rcond=None)[0]
        eb = yb - X @ bb
        sb = cr1_se(X, eb, groups, n_params)
        t_boot[b] = bb[j] / sb[j] if sb[j] > 0 else np.nan
    t_boot = t_boot[~np.isnan(t_boot)]
    p = float((1 + np.sum(np.abs(t_boot) >= abs(t_obs))) / (len(t_boot) + 1))
    return {
        "t_obs": float(t_obs), "coef": float(beta[j]), "se_cr1": float(se[j]),
        "p_wcr": p, "B": int(len(t_boot)), "weights": weights,
        "t_boot_q": {q: float(np.quantile(t_boot, q)) for q in (0.025, 0.5, 0.975)},
    }


def main():
    panel = pd.read_csv(RESULTS + r"\merged_panel_v2.csv")
    tfidf = pd.read_csv(RESULTS + r"\provincial_gat_tfidf.csv")
    panel = panel.merge(tfidf[["year", "province", "GAT_tfidf"]], on=["year", "province"], how="left")
    panel["GAT_hybrid"] = panel[["GAT_composite", "GAT_tfidf"]].mean(axis=1)
    panel["ln_tourism"] = np.log(panel["domestic_tourism_spending"].replace(0, np.nan))
    panel["tourism_growth"] = panel.groupby("province")["ln_tourism"].diff()
    panel["ln_gdp"] = np.log(panel["gdp_billion"].replace(0, np.nan))
    panel["tourism_maturity"] = panel.groupby("province")["ln_tourism"].transform("mean")
    panel["emerging"] = (panel["tourism_maturity"] < panel["tourism_maturity"].median()).astype(float)
    panel["GAT_lag1"] = panel.groupby("province")["GAT_hybrid"].shift(1)
    panel["GAT_lag1_x_emerging"] = panel["GAT_lag1"] * panel["emerging"]
    panel["GATc_lag1"] = panel.groupby("province")["GAT_composite"].shift(1)
    panel["GATc_lag1_x_emerging"] = panel["GATc_lag1"] * panel["emerging"]

    out = {"note": "Round-R16 WCR (null imposed, centred) on exact two-way FE design"}

    # ---- check: exact demeaning vs dummy-variable OLS -------------------
    sub = panel.dropna(subset=["tourism_growth", "GAT_lag1", "GAT_lag1_x_emerging"]).copy()
    dm = two_way_demean(sub, ["tourism_growth", "GAT_lag1", "GAT_lag1_x_emerging"])
    X = np.column_stack([np.ones(len(sub)), dm["GAT_lag1"].values,
                         dm["GAT_lag1_x_emerging"].values])
    y = dm["tourism_growth"].values
    groups = sub["province"].values
    N, K = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se_cr1 = cr1_se(X, resid, groups, K)

    D = pd.get_dummies(sub["province"], drop_first=True).astype(float).values
    T = pd.get_dummies(sub["year"], drop_first=True).astype(float).values
    Xd = np.column_stack([X, D, T])
    beta_d = np.linalg.lstsq(Xd, sub["tourism_growth"].values, rcond=None)[0]
    out["demean_check"] = {
        "coef_interaction_demeaned": float(beta[2]),
        "coef_interaction_dummies": float(beta_d[2]),
        "abs_diff": float(abs(beta[2] - beta_d[2])),
    }

    # ---- M1: corrected WCR on the hybrid (manuscript baseline) ---------
    out["M1_hybrid_baseline"] = {
        "coef": float(beta[2]), "se_cr1": float(se_cr1[2]),
        "t_cr1": float(beta[2] / se_cr1[2]),
        "p_t": float(2 * st.t.sf(abs(beta[2] / se_cr1[2]), df=len(np.unique(groups)) - 1)),
        "nobs": int(N), "n_clusters": int(len(np.unique(groups))),
    }
    out["M1_wcr_webb"] = wcr_pvalue(X, y, groups, 2, B=9999, weights="webb")
    out["M1_wcr_rademacher"] = wcr_pvalue(X, y, groups, 2, B=9999, weights="rademacher")

    # ---- M5: headline re-based on the composite index ------------------
    subc = panel.dropna(subset=["tourism_growth", "GATc_lag1", "GATc_lag1_x_emerging"]).copy()
    dmc = two_way_demean(subc, ["tourism_growth", "GATc_lag1", "GATc_lag1_x_emerging"])
    Xc = np.column_stack([np.ones(len(subc)), dmc["GATc_lag1"].values,
                          dmc["GATc_lag1_x_emerging"].values])
    yc = dmc["tourism_growth"].values
    gc = subc["province"].values
    betac = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    residc = yc - Xc @ betac
    se_c = cr1_se(Xc, residc, gc, 3)
    out["M5_composite_baseline"] = {
        "coef": float(betac[2]), "se_cr1": float(se_c[2]),
        "t_cr1": float(betac[2] / se_c[2]),
        "p_t": float(2 * st.t.sf(abs(betac[2] / se_c[2]), df=len(np.unique(gc)) - 1)),
        "nobs": int(len(subc)), "n_clusters": int(len(np.unique(gc))),
    }
    out["M5_wcr_webb"] = wcr_pvalue(Xc, yc, gc, 2, B=9999, weights="webb")

    with open(RESULTS + r"\round16_wcr_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- M3: leverage of the external validation ----------------------
    lev = {"note": "leave-one-out and robust correlations of the Table 6 cells"}
    with open(RESULTS + r"\round8_external_validation.json", encoding="utf-8") as f:
        ext = json.load(f)
    pp = pd.DataFrame(ext["per_province"])

    def corr_block(x, yv, prov, label):
        m = x.notna() & yv.notna()
        x, yv, prov = x[m].values, yv[m].values, prov[m].values
        r, p = st.pearsonr(x, yv)
        rho, prho = st.spearmanr(x, yv)
        tau, ptau = st.kendalltau(x, yv)
        # Fisher z CI
        n = len(x)
        z = np.arctanh(r)
        sez = 1 / np.sqrt(n - 3)
        ci = (np.tanh(z - 1.96 * sez), np.tanh(z + 1.96 * sez))
        # leave-one-out range
        loos = [st.pearsonr(np.delete(x, i), np.delete(yv, i))[0] for i in range(n)]
        return {"label": label, "n": int(n), "pearson_r": float(r), "pearson_p": float(p),
                "pearson_ci95": [float(ci[0]), float(ci[1])],
                "spearman_rho": float(rho), "spearman_p": float(prho),
                "kendall_tau": float(tau), "kendall_p": float(ptau),
                "loo_r_min": float(np.min(loos)), "loo_r_max": float(np.max(loos)),
                "loo_r_argmin_province": str(prov[int(np.argmin(loos))])}

    for col, name in [("gat_composite", "composite"), ("gat_hybrid_z", "hybrid_z")]:
        lev[f"{name}_full"] = corr_block(pp[col], pp["gat_yang_full"], pp["province"],
                                         f"{name} vs yang_full (N=27)")
        m = (pp["province"] != "海南").values
        lev[f"{name}_ex_Hainan"] = corr_block(pp.loc[m, col], pp.loc[m, "gat_yang_full"],
                                              pp.loc[m, "province"],
                                              f"{name} vs yang_full, excl. Hainan (N=26)")
    lev["Hainan"] = pp[pp["province"] == "海南"].to_dict("records")
    with open(RESULTS + r"\round16_leverage.json", "w", encoding="utf-8") as f:
        json.dump(lev, f, indent=2, ensure_ascii=False)

    # ---- M6: outcome-data audit ---------------------------------------
    aud = {"note": "extreme province-years in domestic_tourism_spending and growth"}
    p2 = panel.copy()
    lo = p2[p2["domestic_tourism_spending"] < 10]
    aud["spending_lt_10"] = lo[["year", "province", "domestic_tourism_spending"]].to_dict("records")
    hi = p2[p2["tourism_growth"].abs() > 0.7]
    aud["abs_growth_gt_0.7"] = hi[["year", "province", "domestic_tourism_spending",
                                   "tourism_growth"]].to_dict("records")
    # winsorised re-estimation (1st/99th pct) of the hybrid baseline
    sub2 = p2.dropna(subset=["tourism_growth", "GAT_lag1", "GAT_lag1_x_emerging"]).copy()
    q1, q99 = np.quantile(sub2["tourism_growth"], [0.01, 0.99])
    sub2["growth_w"] = sub2["tourism_growth"].clip(q1, q99)
    dm2 = two_way_demean(sub2, ["growth_w", "GAT_lag1", "GAT_lag1_x_emerging"])
    X2 = np.column_stack([np.ones(len(sub2)), dm2["GAT_lag1"].values,
                          dm2["GAT_lag1_x_emerging"].values])
    y2 = dm2["growth_w"].values
    g2 = sub2["province"].values
    b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    r2 = y2 - X2 @ b2
    s2 = cr1_se(X2, r2, g2, 3)
    aud["winsorised"] = {"q1": float(q1), "q99": float(q99),
                         "coef": float(b2[2]), "se_cr1": float(s2[2]),
                         "t": float(b2[2] / s2[2])}
    aud["winsorised_wcr_webb"] = wcr_pvalue(X2, y2, g2, 2, B=9999, weights="webb")
    with open(RESULTS + r"\round16_outcome_audit.json", "w", encoding="utf-8") as f:
        json.dump(aud, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in out.items()}, indent=2, ensure_ascii=False))
    print("saved: round16_wcr_bootstrap.json / round16_leverage.json / round16_outcome_audit.json")


if __name__ == "__main__":
    main()