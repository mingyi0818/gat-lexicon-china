"""Round 35: recolour Figures 1 and 2 (31-province panel) for submission.

Mirrors code/round23_fig12_31province.py exactly in data handling and window
(2003-2019, 671-unit panel restricted to 516 rows). Only visual encoding
changes: Figure 1 lines move from all-black to a warm/cool two-group palette,
Figure 2 moves from Greys to YlGnBu. No index value, z-score, selection rule,
or row/column order is recomputed or altered.

Outputs: figures/fig1_gat_trends_31.png (colour)
         figures/fig2_gat_heatmap_31.png (colour)
         results/round35_fig12_colour.json
"""
from __future__ import annotations

import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

PANEL = os.path.join(RES, "provincial_gat_panel_31.csv")
YEAR_LO, YEAR_HI = 2003, 2019

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 22,
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

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

WARM = ["#B2182B", "#D6604D", "#E58267", "#F4A582"]
COOL = ["#053061", "#2166AC", "#35978F", "#8073AC"]


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> None:
    df = pd.read_csv(PANEL)
    win = df[(df["year"] >= YEAR_LO) & (df["year"] <= YEAR_HI)].copy()
    win["province_en"] = win["province"].map(EN)
    missing = sorted(win.loc[win["province_en"].isna(), "province"].unique())
    if missing:
        raise SystemExit(f"unmapped province names: {missing}")

    n_rows, n_prov = len(win), win["province"].nunique()

    means = win.groupby("province_en")["GAT_composite"].mean().sort_values(ascending=False)
    top4 = list(means.index[:4])
    bottom4 = list(means.index[-4:])
    selected = top4 + bottom4

    fig, ax = plt.subplots(figsize=(19, 11.5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for idx, name in enumerate(selected):
        s = win[win["province_en"] == name].sort_values("year")
        colour = WARM[idx] if idx < 4 else COOL[idx - 4]
        ax.plot(
            s["year"], s["GAT_composite"], color=colour,
            marker=markers[idx], markersize=8, linewidth=2.2,
            linestyle=styles[idx], label=name,
        )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("GAT composite (z-score)")
    ax.legend(ncol=4, fontsize=22, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    f1 = os.path.join(FIG, "fig1_gat_trends_31.png")
    fig.savefig(f1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    piv = win.pivot_table(index="province_en", columns="year", values="GAT_composite", aggfunc="mean")
    piv = piv.reindex(means.index)
    order = list(piv.index)
    mat = piv.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(19, 14))
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=90, fontsize=22)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=22)
    ax.set_xlabel("Year")
    ax.set_ylabel("Provincial unit")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("GAT composite (z-score)")
    fig.tight_layout()
    f2 = os.path.join(FIG, "fig2_gat_heatmap_31.png")
    fig.savefig(f2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out = {
        "direction": "07_policy_gat",
        "round": "35-fig12-colour",
        "date": "2026-11-23",
        "purpose": ("recolour Figures 1-2 for submission; visual encoding only, "
                    "no index value, selection rule, or ordering recomputed"),
        "input": "results/provincial_gat_panel_31.csv",
        "window": [YEAR_LO, YEAR_HI],
        "rows_in_window": int(n_rows),
        "provinces_in_window": int(n_prov),
        "province_mean_gat": {k: float(v) for k, v in means.items()},
        "selection_rule": "four highest and four lowest province-mean GAT composite, 2003-2019",
        "selected_for_figure1": selected,
        "figure2_row_order": order,
        "missing_cells_figure2": int(np.isnan(mat).sum()),
        "palette": {"figure1_warm_top4": WARM, "figure1_cool_bottom4": COOL,
                    "figure2_cmap": "YlGnBu"},
        "outputs": {
            "figures/fig1_gat_trends_31.png": {"md5": md5(f1), "bytes": os.path.getsize(f1)},
            "figures/fig2_gat_heatmap_31.png": {"md5": md5(f2), "bytes": os.path.getsize(f2)},
        },
        "note": "data layer identical to round23_fig12_31province.json; colour only",
    }
    dest = os.path.join(RES, "round35_fig12_colour.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in
                      ["rows_in_window", "provinces_in_window", "selected_for_figure1",
                       "missing_cells_figure2"]}, indent=2, ensure_ascii=False))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
