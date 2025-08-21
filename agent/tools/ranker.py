# rank_from_csv_llm.py
import os, json
import pandas as pd
import numpy as np
from typing import Optional, Dict
from openai import OpenAI

# ---- Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ID_COLS = {"ticker", "name", "sector"}  # not used for scoring

# ---- LLM: map free text -> weights & directions (no hardcoded keywords)
def prefs_from_text_llm(user_text: str, df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Ask the LLM to pick *only* from the DataFrame's columns and return a spec like:
    {"col": {"weight": 0.4, "direction": "positive"|"negative"}, ...}
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build schema of numeric columns the model is allowed to use
    numeric_cols = [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]
    # Provide a tiny auto-description to help the model infer meaning/direction
    descriptions = {}
    for c in numeric_cols:
        lc = c.lower()
        if "beta" in lc or "vol" in lc or "risk" in lc:
            descriptions[c] = "Risk-related metric; lower is generally better."
        elif "cap" in lc or "mcap" in lc or "market" in lc:
            descriptions[c] = "Size/scale; direction depends on user intent."
        elif "environment" in lc:
            descriptions[c] = "Environmental risk/score; lower is generally better."
        elif "social" in lc or "divers" in lc:
            descriptions[c] = "Social/diversity risk/score; lower is generally better."
        elif "governance" in lc:
            descriptions[c] = "Governance risk/score; lower is generally better."
        elif "esg" in lc:
            descriptions[c] = "Overall ESG risk/score; lower is generally better."
        else:
            descriptions[c] = "Numeric feature; direction depends on user intent."

    system = (
        "You map an investor's natural-language request to a scoring spec over a table of companies.\n"
        "- Use ONLY the provided numeric columns.\n"
        "- Choose columns relevant to the request; assign weights in [0,1] (they don't have to sum to 1; we'll normalize).\n"
        "- For each chosen column, set direction: 'positive' (higher is better) or 'negative' (lower is better).\n"
        "- Prefer intuitive directions: risk or '..._risk' → 'negative', growth/quality → 'positive', unless the user says otherwise.\n"
        "- Return STRICT JSON: {\"preferences\": {\"col\": {\"weight\": <float>, \"direction\": \"positive\"|\"negative\"}, ...}}\n"
        "- Do not include any columns not in the allowed list."
    )

    user = (
        f"Allowed numeric columns (with short hints):\n"
        + json.dumps(descriptions, indent=2)
        + "\n\nUser request:\n"
        + user_text
        + "\n\nReturn ONLY the JSON object described."
    )

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=1
    )

    data = json.loads(resp.choices[0].message.content)
    prefs = data.get("preferences", {}) if isinstance(data, dict) else {}
    # Keep only valid columns; sanitize values
    clean = {}
    for col, cfg in prefs.items():
        if col in numeric_cols and isinstance(cfg, dict):
            d = str(cfg.get("direction", "positive")).lower()
            d = d if d in ("positive","negative") else "positive"
            try:
                w = float(cfg.get("weight", 0.0))
            except Exception:
                w = 0.0
            clean[col] = {"weight": max(0.0, w), "direction": d}
    return clean

# ---- Scoring core (pure pandas)
def _dirnorm(series: pd.Series, direction: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.median())
    x = -s if direction == "negative" else s
    lo, hi = np.nanmin(x.values), np.nanmax(x.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)  # neutral if constant
    return (x - lo) / (hi - lo)

def rank_from_csv(csv_path: str, user_text: str, top_n: int = 20,
                  sector_neutral: bool = False, out_path: Optional[str] = None):
    df = pd.read_csv(csv_path)

    for col in ("ticker","name"):
        if col not in df.columns:
            raise ValueError("CSV must include 'ticker' and 'name' columns.")

    prefs = prefs_from_text_llm(user_text, df)
    if not prefs:
        raise RuntimeError("LLM returned no usable preferences. Try a clearer request.")

    # Normalize weights to sum to 1
    total_w = sum(v["weight"] for v in prefs.values()) or 1.0
    for k in prefs:
        prefs[k]["weight"] = prefs[k]["weight"] / total_w

    work = df.copy()

    # Sector-neutralize (z-score within sector before normalization)
    mapped = {}
    if sector_neutral and "sector" in work.columns:
        for col in prefs.keys():
            if col in work.columns:
                zcol = f"{col}__sector_z"
                grp = work.groupby("sector")[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
                work[zcol] = grp
                mapped[col] = zcol
    for col in prefs.keys():
        mapped.setdefault(col, col)

    comp_cols = []
    total = None
    for col, cfg in prefs.items():
        src = mapped[col]
        if src not in work.columns:
            continue
        comp = _dirnorm(work[src], cfg["direction"])
        cname = f"score__{col}"
        work[cname] = comp
        comp_cols.append(cname)
        total = comp * cfg["weight"] if total is None else total + comp * cfg["weight"]

    work["score_total"] = total if total is not None else 0.0
    work["rank"] = work["score_total"].rank(ascending=False, method="min").astype(int)

    front = ["rank","ticker","name"] + (["sector"] if "sector" in work.columns else []) + ["score_total"] + comp_cols
    front = [c for c in front if c in work.columns]
    ordered = work.sort_values(["rank","score_total"]).reset_index(drop=True)
    ordered = ordered[front + [c for c in ordered.columns if c not in front]]

    if out_path:
        ordered.to_csv(out_path, index=False)

    return ordered.head(top_n), prefs