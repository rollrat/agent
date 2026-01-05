import sys
from typing import Dict, List

import yfinance as yf
import pandas as pd

TICKERS: Dict[str, str] = {
    "KRW=X": "USD_KRW",
    "DX-Y.NYB": "Dollar_Index",
    "GC=F": "Gold",
    "^IXIC": "Nasdaq",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}


def fetch_financial_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV data for the provided tickers and return a close-price table."""
    # Download all tickers at once (daily data for the selected period)
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No data returned from yfinance. Check tickers or network.")

    # Collect close (or adj close) series for each ticker
    close_data = {}
    available_tickers = set(data.columns.get_level_values(0))

    for ticker in tickers:
        if ticker not in available_tickers:
            print(f"[ERROR] Missing data for ticker: {ticker}")
            continue

        ticker_frame = data[ticker]
        if "Adj Close" in ticker_frame.columns and ticker_frame["Adj Close"].notna().any():
            close_series = ticker_frame["Adj Close"]
        elif "Close" in ticker_frame.columns and ticker_frame["Close"].notna().any():
            close_series = ticker_frame["Close"]
        else:
            print(f"[ERROR] Close/Adj Close data not available for ticker: {ticker}")
            continue

        close_data[ticker] = close_series

    if not close_data:
        raise RuntimeError("No valid close data collected from yfinance response.")

    # Combine into a single dataframe and forward-fill missing values
    combined = pd.DataFrame(close_data).sort_index().ffill()

    # Rename columns to user-friendly names
    combined = combined.rename(columns=TICKERS)

    # Reset index to include Date column
    combined = combined.reset_index()

    return combined


def main() -> None:
    try:
        df = fetch_financial_data(list(TICKERS.keys()), period="1y")
    except Exception as exc:
        print(f"[ERROR] Failed to fetch data: {exc}")
        sys.exit(1)

    # Save to CSV with UTF-8 BOM for Excel compatibility
    output_path = "financial_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved data to {output_path}")


if __name__ == "__main__":
    main()
