import sys
from typing import Dict, List, Tuple

import yfinance as yf
import pandas as pd

MACRO_TICKERS: Dict[str, str] = {
    "KRW=X": "USD_KRW",
    "DX-Y.NYB": "Dollar_Index",
    "GC=F": "Gold",
    "^IXIC": "Nasdaq",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

RAW_TICKERS: List[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "BRK.B",
    "TSLA",
    "RKLB",
    "CRWV",
    "IREN",
    "NBIS",
    "AVGO",
    "UNH",
    "JPM",
    "V",
    "XOM",
    "WMT",
    "HOOD",
    "LLY",
]

TICKER_ALIASES: Dict[str, str] = {
    "BRK.B": "BRK-B",
}


def normalize_tickers(tickers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Normalize user tickers to yfinance-compatible symbols."""
    normalized = []
    reverse_map = {}
    for ticker in tickers:
        yf_symbol = TICKER_ALIASES.get(ticker, ticker)
        normalized.append(yf_symbol)
        reverse_map[yf_symbol] = ticker
    return normalized, reverse_map


def fetch_financial_data(tickers: List[str], rename_map: Dict[str, str], period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV data for the provided tickers and return a close-price table."""
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

    combined = pd.DataFrame(close_data).sort_index().ffill()
    combined = combined.rename(columns=rename_map)
    combined = combined.reset_index()
    return combined


def fetch_price_history(tickers: List[str], period: str = "3mo") -> pd.DataFrame:
    """Fetch daily close data for the provided tickers."""
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
    return combined


def fetch_company_names(tickers: List[str]) -> Dict[str, str]:
    """Fetch company names for the provided tickers."""
    names = {}
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
        names[ticker] = name
    return names


def compute_returns(price_history: pd.DataFrame) -> pd.DataFrame:
    """Compute latest price, 7-day %, and 30-day % returns from price history."""
    records = []
    for ticker in price_history.columns:
        series = price_history[ticker].dropna()
        if series.empty:
            records.append((ticker, None, None, None))
            continue

        last_close = float(series.iloc[-1])
        day_7_change = None
        day_30_change = None
        if len(series) >= 8:
            day_7_change = (last_close / float(series.iloc[-8]) - 1) * 100
        if len(series) >= 31:
            day_30_change = (last_close / float(series.iloc[-31]) - 1) * 100

        records.append((ticker, last_close, day_7_change, day_30_change))

    return pd.DataFrame(
        records,
        columns=["Ticker", "Price", "Change_7d_pct", "Change_30d_pct"],
    )


def main() -> None:
    try:
        macro_df = fetch_financial_data(list(MACRO_TICKERS.keys()), MACRO_TICKERS, period="1y")
        normalized_tickers, reverse_map = normalize_tickers(RAW_TICKERS)
        price_history = fetch_price_history(normalized_tickers, period="3mo")
        returns_df = compute_returns(price_history)
        company_names = fetch_company_names(normalized_tickers)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch data: {exc}")
        sys.exit(1)

    returns_df["Name"] = returns_df["Ticker"].map(company_names).fillna("")
    returns_df["Ticker"] = returns_df["Ticker"].map(reverse_map).fillna(returns_df["Ticker"])
    returns_df = returns_df[["Ticker", "Name", "Price", "Change_7d_pct", "Change_30d_pct"]]

    output_path = "stock_summary.csv"
    returns_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved data to {output_path}")

    macro_output_path = "financial_data.csv"
    macro_df.to_csv(macro_output_path, index=False, encoding="utf-8-sig")
    print(f"Saved data to {macro_output_path}")


if __name__ == "__main__":
    main()
