"""Build end-2011 provincial 5A stock from the official MCT data page.

Source archived at results/official_mct_5a_20250311.html.
The page states a data cutoff of 2025-03-11 and lists designation years.
For entries with a slash year (e.g. 2007/2018), the first designation year is
used for cumulative stock at end-2011.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS = ROOT / "results"
HTML = RESULTS / "official_mct_5a_20250311.html"
SOURCE_URL = "https://sjfw.mct.gov.cn/site/dataservice/rural?type=10"

PROVINCES = {
    "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽",
    "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "四川",
    "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
}


def main():
    soup = BeautifulSoup(HTML.read_text(encoding="utf-8"), "html.parser")
    container = soup.select_one("#province")
    if container is None:
        raise RuntimeError("Official MCT province container not found")

    records = []
    for block in container.select(":scope > .x-wrap > .list > .li"):
        title = block.select_one(".tit")
        if title is None:
            continue
        province = title.get_text(strip=True)
        if province not in PROVINCES:
            continue
        for anchor in block.select(".box .span a"):
            text = anchor.get_text(" ", strip=True)
            years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", text)]
            if not years:
                continue
            records.append({
                "province": province,
                "attraction": text,
                "designation_year_first": min(years),
                "years_in_label": "/".join(map(str, years)),
            })

    attractions = pd.DataFrame(records)
    if attractions.empty or attractions["province"].nunique() != 27:
        raise RuntimeError(
            f"Unexpected official parse: {len(attractions)} rows, "
            f"{attractions['province'].nunique() if len(attractions) else 0} provinces"
        )
    attractions.to_csv(
        RESULTS / "official_mct_5a_attractions.csv",
        index=False, encoding="utf-8-sig"
    )

    stock = (
        attractions.assign(
            by_end_2011=attractions["designation_year_first"] <= 2011
        )
        .groupby("province", as_index=False)["by_end_2011"]
        .sum()
        .rename(columns={"by_end_2011": "n_5a_2011_official"})
    )
    stock["n_5a_2011_official"] = stock["n_5a_2011_official"].astype(int)
    median = float(stock["n_5a_2011_official"].median())
    stock["em_5a_official"] = (
        stock["n_5a_2011_official"] < median
    ).astype(float)
    stock.to_csv(
        RESULTS / "provincial_5a_stock_official.csv",
        index=False, encoding="utf-8-sig"
    )

    third_party = pd.read_csv(RESULTS / "provincial_5a_stock.csv")
    third_2011 = (
        third_party[third_party["year"] == 2011][["province", "n_5a_cum"]]
        .rename(columns={"n_5a_cum": "n_5a_2011_third_party"})
    )
    comparison = stock.merge(third_2011, on="province", how="left")
    comparison["difference"] = (
        comparison["n_5a_2011_official"]
        - comparison["n_5a_2011_third_party"]
    )
    comparison.to_csv(
        RESULTS / "official_vs_third_party_5a_2011.csv",
        index=False, encoding="utf-8-sig"
    )
    audit = {
        "source_url": SOURCE_URL,
        "archived_html": str(HTML),
        "source_cutoff": "2025-03-11",
        "n_attractions_current_page": int(len(attractions)),
        "n_sample_provinces": int(stock["province"].nunique()),
        "median_end2011": median,
        "n_emerging": int(stock["em_5a_official"].sum()),
        "official_vs_third_party": {
            "n_equal": int((comparison["difference"] == 0).sum()),
            "n_different": int((comparison["difference"] != 0).sum()),
            "max_abs_difference": int(comparison["difference"].abs().max()),
        },
        "guardrail": (
            "Official end-2011 stock is outcome-distinct but potentially endogenous "
            "and post-baseline for 2003-2011 observations."
        ),
    }
    (RESULTS / "round20_official_5a_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
