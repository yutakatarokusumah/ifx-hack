import pandas as pd
import yfinance as yf


TICKERS = ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK"]
START_DATE = "1900-01-01"
END_DATE = "2026-10-01"


def extract_field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract one field from yfinance's ticker-grouped result."""
    return pd.DataFrame(
        {ticker: data[ticker][field] for ticker in TICKERS}
    ).dropna()


def main() -> None:
    print("Downloading historical market data...")
    raw_data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        auto_adjust=False,
    )

    price_matrix = extract_field(raw_data, "Adj Close")
    volume_matrix = extract_field(raw_data, "Volume")
    price_matrix.to_csv("price_matrix.csv")
    volume_matrix.to_csv("volume_matrix.csv")
    print(f"Data saved successfully! Total trading days: {len(price_matrix)}")


if __name__ == "__main__":
    main()
