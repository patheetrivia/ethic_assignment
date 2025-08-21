# esg_scores_playwright_pillars.py
import re
import time
import argparse
import pandas as pd
from playwright.sync_api import sync_playwright

SUST_URL = "https://finance.yahoo.com/quote/{ticker}/sustainability"

# Simple number finder (e.g., "23.6", "12")
NUM = re.compile(r"\d+(?:\.\d+)?")

LABELS = {
    "environmental_risk": ["environment risk score", "environmental risk score"],
    "social_risk":        ["social risk score"],
    "governance_risk":    ["governance risk score"],
    "esg_total":          ["esg risk rating", "total esg risk", "overall esg risk"],
}

def extract_pillars_from_page(page_text):
    """Scan visible text once and pull the closest number next to known labels."""
    text = " ".join(page_text.split())  # collapse whitespace
    out = {"environmental_risk": None, "social_risk": None, "governance_risk": None, "esg_total": None}

    def find_first(label_variants):
        for lab in label_variants:
            i = text.lower().find(lab)
            if i != -1:
                # search a small window after the label for the first number
                window = text[i:i+200].lower()
                m = NUM.search(window)
                if m:
                    try:
                        return float(m.group())
                    except Exception:
                        pass
        return None

    for key, variants in LABELS.items():
        out[key] = find_first(variants)

    return out

def fetch_scores_with_playwright(ticker, wait_ms=1200):
    """Open rendered sustainability page and extract pillar + total scores."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(locale="en-US")
        page = ctx.new_page()
        page.goto(SUST_URL.format(ticker=ticker), wait_until="domcontentloaded")
        # Give the client scripts a moment to render numbers
        page.wait_for_timeout(wait_ms)
        # Grab full visible text (robust across layout changes)
        txt = page.locator("body").inner_text()
        browser.close()
    return extract_pillars_from_page(txt)

def main(csv_path, limit=None, sleep=0.05, wait_ms=1200):
    df = pd.read_csv(csv_path)
    for c in ["environmental_risk","social_risk","governance_risk","esg_total"]:
        if c not in df.columns:
            df[c] = None

    tickers = df["ticker"].astype(str).str.upper().tolist()
    if limit:
        tickers = tickers[:limit]

    for i, t in enumerate(tickers, 1):
        try:
            scores = fetch_scores_with_playwright(t, wait_ms=wait_ms)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: error ({e})")
            scores = {"environmental_risk": None, "social_risk": None, "governance_risk": None, "esg_total": None}

        mask = df["ticker"].astype(str).str.upper() == t
        df.loc[mask, ["environmental_risk","social_risk","governance_risk","esg_total"]] = [
            scores["environmental_risk"], scores["social_risk"],
            scores["governance_risk"], scores["esg_total"]
        ]
        print(f"[{i}/{len(tickers)}] {t}: {scores}")
        time.sleep(sleep)

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sp500_companies.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--wait-ms", type=int, default=1200, help="Render wait before reading page text")
    args = ap.parse_args()
    main(args.csv, limit=args.limit, sleep=args.sleep, wait_ms=args.wait_ms)