import pandas as pd
import yfinance as yf


# Hang Seng TECH-style HKEX basket; not the official index constituents.
TICKERS = ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK"]
BENCHMARK = "^HSI"
SAFE_HAVEN = "Cash"
START_DATE = "2000-01-01"
END_DATE = "2026-10-01"

def main() -> None:
    print("Downloading historical market data...")
    raw_data = yf.download(
        TICKERS + [BENCHMARK],
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        auto_adjust=False,
    )

    columns = TICKERS + [BENCHMARK]
    price_matrix = pd.DataFrame(
        {ticker: raw_data[ticker]["Adj Close"] for ticker in columns}
    ).dropna()
    volume_matrix = pd.DataFrame(
        {ticker: raw_data[ticker]["Volume"] for ticker in columns}
    ).reindex(price_matrix.index)
    price_matrix[SAFE_HAVEN] = 100.0
    volume_matrix[SAFE_HAVEN] = 0.0
    price_matrix.to_csv("price_matrix.csv")
    volume_matrix.to_csv("volume_matrix.csv")
    print(f"Data saved successfully! Total trading days: {len(price_matrix)}")


if __name__ == "__main__":
    main()
