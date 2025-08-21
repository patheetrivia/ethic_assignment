import requests
from bs4 import BeautifulSoup
import yfinance as yf
from typing import Optional, Dict, Any

# Yahoo Snapshot
def fetch_company_data(ticker: str) -> Dict[str, Any]:
    """
    Returns a minimal, trustworthy snapshot for a US-listed equity.
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    fast = getattr(t, "fast_info", {}) or {}

    # Prefer real-timeish `fast_info.last_price`; fall back to last close.
    last_price: Optional[float] = None
    lp = fast.get("last_price") or fast.get("lastPrice")  # yfinance versions differ
    if isinstance(lp, (int, float)):
        last_price = float(lp)
    else:
        hist = t.history(period="1d")
        if not hist.empty:
            last_price = float(hist["Close"].iloc[-1])

    # Currency: take fast_info.currency, else info.currency; default to "USD" for US tickers
    currency = fast.get("currency") or info.get("currency") or "USD"

    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "beta": info.get("beta"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "eps": info.get("trailingEps") or info.get("forwardEps"),
        "last_price": last_price,      # ✅ the only price we expose
        "currency": currency,          # ✅ usually "USD" for AAPL
        "as_of": fast.get("last_price_time") or info.get("regularMarketTime"),
        "source": "Yahoo Finance via yfinance"
    }