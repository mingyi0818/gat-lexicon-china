"""Build the 31-province panels (27 provinces plus the four municipalities).

Preregistered in experiments/prereg_round22_31province_20261122.md.
Reads 27 province-level units from 政府工作报告-省自治区 plus 4 municipalities
(北京市/天津市/上海市/重庆市) from 政府工作报告-主要城市, recomputes GAT,
TF-IDF tourism weight, and end-2011 5A stock on all 31 units.
"""
# 建面板：政府工作报告 → 37 词词典的四个统计量 → TF-IDF 层 → 5A 存量。
from __future__ import annotations

import os

import re
import sys
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from utils import normalize_province  # noqa: E402

CORPUS = Path(os.environ.get("GAT_CORPUS", "corpus/lawtext_gov"))
PROV_DIR = CORPUS / "政府工作报告-省自治区"
CITY_DIR = CORPUS / "政府工作报告-主要城市"
MUNICIPALITIES = ["北京市", "天津市", "上海市", "重庆市"]

# Frozen lexicon — must match config.TOURISM_LEXICON exactly (37 terms).
TOURISM_LEXICON = [
    "旅游", "文旅", "文化旅", "全域旅游", "智慧旅游", "乡村旅游", "红色旅游",
    "生态旅游", "康养旅游", "冰雪旅游", "入境旅游", "出境旅游", "游客",
    "景区", "景点", "5A", "4A", "A级景区", "民宿", "酒店", "接待",
    "文旅融合", "旅游收入", "旅游消费", "旅游经济", "旅游旺", "旅游强",
    "精品线路", "旅游品牌", "旅游合作", "旅游扶贫", "旅游恢复",
    "夜间文旅", "旅游业态", "旅游服务", "旅游质量", "旅游安全",
]


# 清文本：去 HTML 标签、全角空格、页码那些噪声，正文不能吃掉。
def _clean_text(text: str) -> str:
    text = re.sub(r"^---.*?---", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*.*?\*\*", "", text)
    text = re.sub(r"\s+", "", text)
    return text


# 按句末标点切子句；少于 8 个字的碎片丢掉，太短的多半是标题残片。
def _segment(text: str) -> list[str]:
    parts = re.split(r"[。！？\n；;]", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


# 四个统计量各自 z 分数后等权平均：词典密度、子句比例、字符密度、词项广度。
def compute_gat_for_text(text: str) -> dict:
    text = _clean_text(text)
    chars = max(len(text), 1)
    words = list(jieba.cut(text))
    word_n = max(len(words), 1)
    lex_hits = sum(1 for w in words if any(kw in w for kw in TOURISM_LEXICON))
    lex_density = lex_hits / word_n
    segs = _segment(text)
    seg_n = max(len(segs), 1)
    tourism_segs = [s for s in segs if any(kw in s for kw in TOURISM_LEXICON)]
    seg_ratio = len(tourism_segs) / seg_n
    char_hits = sum(text.count(kw) for kw in TOURISM_LEXICON)
    char_density = char_hits / chars * 1000
    unique_terms = sum(1 for kw in TOURISM_LEXICON if kw in text)
    term_breadth = unique_terms / len(TOURISM_LEXICON)
    return {
        "text_length": chars,
        "word_count": word_n,
        "segment_count": seg_n,
        "tourism_segment_count": len(tourism_segs),
        "lex_density": lex_density,
        "seg_ratio": seg_ratio,
        "char_density": char_density,
        "term_breadth": term_breadth,
    }


def iter_report_dirs() -> list[Path]:
    dirs = [d for d in PROV_DIR.iterdir() if d.is_dir()]
    for name in MUNICIPALITIES:
        p = CITY_DIR / name
        if p.is_dir():
            dirs.append(p)
    return dirs


def build_gat(year_min=2003, year_max=2024) -> pd.DataFrame:
    records = []
    for d in iter_report_dirs():
        province = normalize_province(d.name)
        if province is None:
            continue
        for md in sorted(d.glob("*.md")):
            if not md.stem.isdigit():
                continue
            year = int(md.stem)
            if not (year_min <= year <= year_max):
                continue
            text = md.read_text(encoding="utf-8", errors="ignore")
            metrics = compute_gat_for_text(text)
            records.append({"year": year, "province": province, "province_dir": d.name, **metrics})
    gat = pd.DataFrame(records)
    gat = gat.drop_duplicates(subset=["year", "province"], keep="last")
    gat = gat.sort_values(["province", "year"]).reset_index(drop=True)
    for col in ["lex_density", "seg_ratio", "char_density", "term_breadth"]:
        mu, sd = gat[col].mean(), gat[col].std(ddof=0)
        gat[f"z_{col}"] = (gat[col] - mu) / sd if sd > 0 else 0.0
    gat["GAT_composite"] = gat[
        [f"z_{c}" for c in ["lex_density", "seg_ratio", "char_density", "term_breadth"]]
    ].mean(axis=1)
    gat["GAT_lag"] = gat.groupby("province")["GAT_composite"].shift(1)
    gat["GAT_change"] = gat["GAT_composite"] - gat["GAT_lag"]
    gat.to_csv(RESULTS / "provincial_gat_panel_31.csv", index=False, encoding="utf-8-sig")
    return gat


# TF-IDF 必须用全部 815 份文档估，只用估计窗口的话权重会跟着窗口漂。
def build_tfidf() -> pd.DataFrame:
    """TF-IDF over the FULL corpus (all years, 815 documents), matching the
    canonical methodology (06_attempt4_between_determinants.compute_tfidf_gat
    uses no year filter)."""
    records = []
    for d in iter_report_dirs():
        province = normalize_province(d.name)
        if province is None:
            continue
        for md in sorted(d.glob("*.md")):
            if not md.stem.isdigit():
                continue
            year = int(md.stem)
            text = md.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"\s+", " ", text)
            records.append({"year": year, "province": province, "text": text[:80000]})
    df = pd.DataFrame(records)
    vec = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b", stop_words=None)
    tfidf = vec.fit_transform(df["text"])
    vocab = vec.get_feature_names_out()
    tour_idx = [i for i, w in enumerate(vocab) if any(k in w for k in TOURISM_LEXICON)]
    if not tour_idx:
        tour_idx = [i for i, w in enumerate(vocab) if "旅" in w or "游" in w]
    df["GAT_tfidf"] = np.asarray(tfidf[:, tour_idx].mean(axis=1)).ravel()
    df = df.drop(columns=["text"])
    df.to_csv(RESULTS / "provincial_gat_tfidf_31.csv", index=False, encoding="utf-8-sig")
    return df


# 5A 按授予年份累加，取 2011 年底的值；年份写两个的按头一个算。
def build_5a_stock() -> pd.DataFrame:
    HTML = RESULTS / "official_mct_5a_20250311.html"
    PROVINCES = {
        "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
        "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
        "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
        "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
    }
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
            years = [int(v) for v in re.findall(r"(?:19|20)\d{2}", text)]
            if not years:
                continue
            records.append({
                "province": province,
                "attraction": text,
                "designation_year_first": min(years),
            })
    att = pd.DataFrame(records)
    stock = (
        att.assign(by_end_2011=att["designation_year_first"] <= 2011)
        .groupby("province", as_index=False)["by_end_2011"]
        .sum()
        .rename(columns={"by_end_2011": "n_5a_2011_official"})
    )
    stock["n_5a_2011_official"] = stock["n_5a_2011_official"].astype(int)
    stock.to_csv(RESULTS / "provincial_5a_stock_official_31.csv", index=False, encoding="utf-8-sig")
    return stock


if __name__ == "__main__":
    gat = build_gat()
    tfidf = build_tfidf()
    stock = build_5a_stock()
    print(f"GAT_31: {gat.shape}, provinces={gat['province'].nunique()}")
    print(f"TFIDF_31: {tfidf.shape}, provinces={tfidf['province'].nunique()}, "
          f"years={tfidf['year'].min()}-{tfidf['year'].max()}")
    print(f"5A_31: {stock.shape}, provinces={stock['province'].nunique()}")
    print("Municipality GAT years (2003-2019):")
    muni = gat[gat["province"].isin(["北京", "天津", "上海", "重庆"])]
    print(muni.groupby("province")["year"].agg(["min", "max", "count"]).to_string())
    print("5A stock (31 units):")
    print(stock.sort_values("n_5a_2011_official", ascending=False).to_string(index=False))
