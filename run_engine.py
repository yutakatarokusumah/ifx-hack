import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = ["vol_5d", "vol_20d", "vol_momentum"]


def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col=0)
    except FileNotFoundError as error:
        raise RuntimeError(f"Required input file is missing: {path}") from error
    except PermissionError as error:
        raise RuntimeError(f"Cannot read input file: {path}") from error
    except (OSError, pd.errors.ParserError) as error:
        raise RuntimeError(f"Could not read input file: {path}") from error


def load_returns() -> tuple[pd.DataFrame, pd.Series]:
    prices = read_csv("price_matrix.csv")
    volumes = read_csv("volume_matrix.csv")
    returns = prices.pct_change().dropna()
    return returns, volumes.mean(axis=1).reindex(returns.index)


def build_features(
    returns: pd.DataFrame, average_volume: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    market_returns = returns.mean(axis=1)
    features = pd.DataFrame(index=returns.index)
    features["vol_5d"] = market_returns.rolling(5).std()
    features["vol_20d"] = market_returns.rolling(20).std()
    features["vol_momentum"] = (
        average_volume.replace(0, np.nan).pct_change(5, fill_method=None)
    )
    features["target"] = market_returns.rolling(5).std().shift(-5)
    data = features.replace([np.inf, -np.inf], np.nan).dropna()
    return data[FEATURE_COLUMNS], data["target"]


def train_model(features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    model = RandomForestRegressor(
        n_estimators=50, max_depth=4, random_state=42
    )
    model.fit(features, target)
    return dict(zip(features.columns, model.feature_importances_))


def calculate_centrality(
    returns: pd.DataFrame, short_volatility_weight: float
) -> dict[str, float]:
    correlations = returns.corr()
    distances = np.sqrt(np.maximum(0, 2 * (1 - correlations)))
    adjusted_distances = distances * (1 - 0.5 * short_volatility_weight)

    tickers = correlations.columns.to_list()
    weights = np.zeros_like(adjusted_distances, dtype=float)
    upper = np.triu_indices_from(weights, k=1)
    weights[upper] = 1 / (adjusted_distances.to_numpy()[upper] + 1e-5)
    weights[(upper[1], upper[0])] = weights[upper]
    graph = nx.from_numpy_array(weights)
    graph = nx.relabel_nodes(graph, dict(enumerate(tickers)))
    strength = dict(graph.degree(weight="weight"))
    maximum = max(strength.values(), default=1)
    return {ticker: score / maximum for ticker, score in strength.items()}


def allocate(
    centrality: dict[str, float], aggressiveness: float = 3.0
) -> dict[str, float]:
    tickers = np.array(list(centrality))
    scores = np.array(list(centrality.values()), dtype=float)
    inverse = 1 / (scores + 1e-4)
    weighted = np.power(inverse, aggressiveness)
    weights = weighted / weighted.sum()
    return dict(zip(tickers, weights))


def main() -> None:
    returns, average_volume = load_returns()
    features, target = build_features(returns, average_volume)
    importances = train_model(features, target)
    centrality = calculate_centrality(returns, importances["vol_5d"])
    ai_weights = allocate(centrality)
    naive_weights = {ticker: 1 / len(ai_weights) for ticker in ai_weights}

    print("=== AI Feature Importances (Explainability) ===")
    for feature, score in importances.items():
        print(f"{feature}: {score:.3f}")

    print("\n=== Systemic Centrality Score (Super-Spreader Risk) ===")
    for ticker, score in centrality.items():
        print(f"{ticker}: {score:.4f}")

    result = pd.DataFrame(
        {
            "Naive Baseline Weight": naive_weights,
            "AI Risk-Adjusted Weight": ai_weights,
        }
    )
    result["Adjustment (%)"] = (
        result["AI Risk-Adjusted Weight"] - result["Naive Baseline Weight"]
    ) * 100
    print("\n=== Final Portfolio Rebalancing Comparison ===")
    print(result.round(4))


if __name__ == "__main__":
    main()
