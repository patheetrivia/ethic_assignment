import pandas as pd
import yfinance as yf

# Path to your csv
csv_path = "sp500_companies.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Update market_cap and beta for each ticker, can be run daily
for i, t in enumerate(df["ticker"], 1):
    try:
        info = yf.Ticker(t).info
        df.loc[df["ticker"] == t, "market_cap"] = info.get("marketCap")
        df.loc[df["ticker"] == t, "beta"] = info.get("beta")
        print(f"[{i}] {t}: updated")
    except Exception as e:
        print(f"[{i}] {t}: failed ({e})")

# Write back to the same CSV
df.to_csv(csv_path, index=False)
print(f"Updated CSV saved to {csv_path}")