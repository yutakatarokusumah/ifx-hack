import altair as alt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

from run_engine import (
    allocate,
    apply_regime_filter,
    is_stress_regime,
    market_volatility,
)


FEATURE_COLUMNS = ["vol_5d", "vol_20d", "momentum"]
WINDOW = 60
MIN_TRAIN = 120
HORIZON = 5
REBALANCE = 20
SAFE_HAVEN = "Cash"
BENCHMARK = "^HSI"
SCENARIO_MARKETS = {
    "Hang Seng Index": "^HSI",
    "Shanghai Composite": "000001.SS",
    "Singapore STI": "^STI",
}
BASKET_NAME = "Hang Seng Tech basket"
EQUITY_LABEL = "Hang Seng Index (^HSI)"
STRESS_SHOCKS = np.array(
    [-0.02, -0.04, -0.06, -0.05, -0.03, -0.04, -0.02, 0.01, 0.02, 0.03]
)

st.set_page_config(layout="wide")
st.title("AI-driven Hang Seng Tech index")
st.caption(
    "Walk-forward crisis protection for a Hang Seng Tech basket of "
    "HKEX-listed equities."
)


@st.cache_data
def load_returns() -> pd.DataFrame:
    prices = pd.read_csv("price_matrix.csv", index_col=0, parse_dates=True)
    if BENCHMARK not in prices.columns:
        if prices.empty:
            raise RuntimeError("price_matrix.csv contains no market data.")
        try:
            benchmark = yf.download(
                BENCHMARK,
                start=prices.index.min().date().isoformat(),
                end=(prices.index.max() + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=False,
                progress=False,
            )
            close = benchmark["Adj Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            prices[BENCHMARK] = close.reindex(prices.index)
        except (KeyError, ValueError, OSError) as error:
            raise RuntimeError(
                "The benchmark ^HSI is missing from price_matrix.csv and could "
                "not be downloaded. Run `fetch_data.py` to refresh the data."
            ) from error
        if prices[BENCHMARK].isna().all():
            raise RuntimeError(
                "The benchmark ^HSI could not be aligned with price_matrix.csv. "
                "Run `fetch_data.py` to refresh the data."
            )
    return prices.pct_change().dropna()


@st.cache_data(max_entries=8)
def load_scenario_returns(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    try:
        data = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False,
            progress=False,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not download regional scenario data for {ticker}."
        ) from error
    if data.empty:
        raise RuntimeError(f"No data was returned for scenario index {ticker}.")
    close = data["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.pct_change().dropna().rename(ticker)


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    market = returns.drop(
        columns=[SAFE_HAVEN, BENCHMARK], errors="ignore"
    ).mean(axis=1)
    return pd.DataFrame(
        {
            "vol_5d": market.rolling(5).std(),
            "vol_20d": market.rolling(20).std(),
            "momentum": market.rolling(5).mean(),
            "target": market.rolling(HORIZON).std().shift(-HORIZON),
        },
        index=returns.index,
    ).replace([np.inf, -np.inf], np.nan)


def build_graph(returns: pd.DataFrame, sensitivity: float) -> nx.Graph:
    correlations = returns.drop(
        columns=BENCHMARK, errors="ignore"
    ).corr().fillna(0)
    distances = np.sqrt(np.maximum(0, 2 - 2 * correlations.to_numpy()))
    weights = 1 / np.power(distances + 1e-5, 1 + sensitivity)
    np.fill_diagonal(weights, 0)
    graph = nx.from_numpy_array(weights)
    return nx.relabel_nodes(graph, dict(enumerate(correlations.columns)))


def graph_centrality(graph: nx.Graph) -> dict[str, float]:
    strength = dict(graph.degree(weight="weight"))
    maximum = max(strength.values(), default=1)
    return {ticker: score / maximum for ticker, score in strength.items()}


@st.cache_data(max_entries=16)
def backtest(
    returns: pd.DataFrame, sensitivity: float, aggressiveness: float
) -> pd.DataFrame:
    features = build_features(returns)
    risk_assets = returns.drop(columns=[SAFE_HAVEN, BENCHMARK], errors="ignore")
    equal = 1 / risk_assets.shape[1]
    portfolio_returns: list[tuple[pd.Timestamp, float, float, bool]] = []

    for position in range(MIN_TRAIN, len(returns) - HORIZON, REBALANCE):
        date = returns.index[position]
        train = features.iloc[:position].dropna()
        if len(train) < MIN_TRAIN:
            continue

        model = RandomForestRegressor(
            n_estimators=20, max_depth=3, random_state=42, n_jobs=-1
        )
        model.fit(train[FEATURE_COLUMNS], train["target"])
        predicted_vol = model.predict(features.loc[[date], FEATURE_COLUMNS])[0]
        crisis_level = np.clip(
            (predicted_vol / train["target"].median() - 1) * sensitivity, 0, 1
        )

        graph = build_graph(returns.iloc[position - WINDOW : position], crisis_level)
        risk = allocate(graph_centrality(graph), aggressiveness)
        risk = apply_regime_filter(
            risk,
            is_stress_regime(returns.iloc[:position]),
        )
        base_weights = {ticker: equal for ticker in risk_assets}
        base_weights[SAFE_HAVEN] = 0.0
        weights = {
            ticker: (1 - crisis_level) * base_weights.get(ticker, 0.0)
            + crisis_level * risk.get(ticker, 0.0)
            for ticker in base_weights
        }
        realized = risk_assets.iloc[position : position + HORIZON].mean(axis=1)
        crisis = realized.std() > risk_assets.mean(axis=1).rolling(20).std().quantile(
            0.75
        )
        for day, row in returns.iloc[position : position + HORIZON].iterrows():
            portfolio_returns.append(
                (
                    day,
                    float(row[list(weights)] @ pd.Series(weights)),
                    float(row[BENCHMARK]),
                    crisis,
                )
            )

    return pd.DataFrame(
        portfolio_returns,
        columns=["date", "ai", "index", "crisis"],
    ).set_index("date")


def performance(returns: pd.Series) -> tuple[float, float]:
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    return float(curve.iloc[-1] - 1), float(drawdown.min())


def stress_test(
    returns: pd.DataFrame,
    weights: dict[str, float],
    scenario_returns: pd.Series,
) -> pd.DataFrame:
    risk_assets = returns.drop(columns=[SAFE_HAVEN, BENCHMARK], errors="ignore")
    aligned = risk_assets.join(scenario_returns.rename("scenario"), how="inner")
    if len(aligned) < len(STRESS_SHOCKS):
        raise RuntimeError(
            "The selected regional scenario does not overlap enough with the "
            "portfolio history to build a stress replay."
        )
    market = aligned["scenario"]
    beta = aligned[risk_assets.columns].apply(
        lambda column: column.cov(market) / market.var()
    )
    beta[SAFE_HAVEN] = 0.0
    shock_window = market.rolling(len(STRESS_SHOCKS)).sum().dropna()
    shock_end = shock_window.index[shock_window.to_numpy().argmin()]
    shock_end_position = market.index.get_loc(shock_end)
    shocks = market.iloc[
        shock_end_position - len(STRESS_SHOCKS) + 1 : shock_end_position + 1
    ].to_numpy()
    dates = pd.date_range("2008-09-15", periods=len(shocks))
    shocked_assets = pd.DataFrame(
        np.outer(shocks, beta),
        columns=beta.index,
        index=dates,
    )
    return pd.DataFrame(
        {
            "ai": shocked_assets @ pd.Series(weights),
            "index": pd.Series(shocks, index=shocked_assets.index),
        }
    )


def performance_chart(curves: pd.DataFrame) -> alt.Chart:
    data = curves.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="portfolio", value_name="value"
    )
    color = alt.Color(
        "portfolio:N",
        scale=alt.Scale(
            domain=["AI portfolio", EQUITY_LABEL],
            range=["#5b3fd1", "#8fb4ff"],
        ),
        legend=alt.Legend(title=None),
    )
    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title="Growth of $1"),
        color=color,
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("portfolio:N", title="Portfolio"),
            alt.Tooltip("value:Q", title="Value", format=".3f"),
        ],
    )
    return base.mark_line(size=3).encode(
        strokeDash=alt.condition(
            alt.datum.portfolio == EQUITY_LABEL,
            alt.value([6, 4]),
            alt.value([1, 0]),
        )
    ).properties(height=360)


returns = load_returns()
available_start = returns.index.min().date()
available_end = returns.index.max().date()
start_date, end_date = st.sidebar.slider(
    "Backtest date range",
    min_value=available_start,
    max_value=available_end,
    value=(available_start, available_end),
    format="YYYY-MM-DD",
)
if start_date == end_date:
    st.error("Choose a date range containing at least two trading days.")
    st.stop()
returns = returns.loc[
    pd.Timestamp(start_date) : pd.Timestamp(end_date)
]
st.sidebar.caption(
    f"Available history: {available_start:%Y-%m-%d} to {available_end:%Y-%m-%d}"
)
st.sidebar.markdown(
    "**Basket constituents:** `0700.HK`, `9988.HK`, `3690.HK`, "
    "`1810.HK`, `9618.HK`"
)
if len(returns) <= MIN_TRAIN + HORIZON:
    st.error(
        f"Select at least {MIN_TRAIN + HORIZON + 1} trading days "
        "for the walk-forward backtest."
    )
    st.stop()
vol_weight = st.sidebar.slider(
    "AI crisis sensitivity",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.25,
)
allocation_aggressiveness = st.sidebar.slider(
    "Allocation Aggressiveness",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
)

results = backtest(returns, vol_weight, allocation_aggressiveness)
ai_return, ai_drawdown = performance(results["ai"])
index_return, index_drawdown = performance(results["index"])
crisis_results = results[results["crisis"]]
ai_crisis, index_crisis = performance(crisis_results["ai"]), performance(
    crisis_results["index"]
)

st.subheader("Walk-forward backtest")
st.caption(
    "Each prediction uses only data available before that rebalance. "
    "The blue benchmark is the actual Hang Seng Index (^HSI); "
    f"the AI portfolio allocates across the {BASKET_NAME} constituents."
)
chart = results[["ai", "index"]].add(1).cumprod()
chart.columns = ["AI portfolio", EQUITY_LABEL]
st.altair_chart(
    performance_chart(chart),
    width="stretch",
    theme=None,
)

spread = chart["AI portfolio"] - chart[EQUITY_LABEL]
st.area_chart(
    (spread * 100).rename("AI advantage versus index (%)"),
    y_label="Percentage-point difference",
    width="stretch",
)

metrics = pd.DataFrame(
    {
        "AI portfolio": [ai_return, ai_drawdown, ai_crisis[0]],
        EQUITY_LABEL: [index_return, index_drawdown, index_crisis[0]],
    },
    index=["Total return", "Max drawdown", "Crisis-period return"],
)
st.dataframe(
    metrics.style.format("{:.2%}"),
    width="stretch",
    alt="Backtest performance comparison",
)

st.info(
    f"Detected {len(crisis_results)} crisis-period trading days. "
    "The model is a risk signal, not a guarantee of outperformance; "
    "use the backtest to judge whether protection helped on this dataset."
)

graph = build_graph(returns.tail(WINDOW), vol_weight)
centrality = dict(graph.degree(weight="weight"))
current_volatility = market_volatility(returns).iloc[-1]
volatility_threshold = market_volatility(returns).dropna().quantile(0.9)
stress_active = is_stress_regime(returns)

scenario_name = st.sidebar.selectbox(
    "Regional stress scenario",
    options=list(SCENARIO_MARKETS),
    index=0,
)
scenario_returns = load_scenario_returns(
    SCENARIO_MARKETS[scenario_name], returns.index.min(), returns.index.max()
)

st.subheader(f"Illustrative {scenario_name} crisis replay")
st.caption(
    f"This is a scenario replay, not a claim of historical AI performance in "
    f"{scenario_name}. It uses that index's worst observed {len(STRESS_SHOCKS)}-"
    "day window and applies the assets' observed regional betas."
)
stress_weights = apply_regime_filter(
    allocate(graph_centrality(graph), allocation_aggressiveness),
    True,
)
stress = stress_test(returns, stress_weights, scenario_returns)
stress_chart = (1 + stress).cumprod()
stress_chart.columns = ["AI portfolio", EQUITY_LABEL]
st.altair_chart(
    performance_chart(stress_chart),
    width="stretch",
    theme=None,
)
stress_ai = performance(stress["ai"])
stress_index = performance(stress["index"])
stress_metrics = pd.DataFrame(
    {
        "AI portfolio": [stress_ai[0], stress_ai[1]],
        EQUITY_LABEL: [stress_index[0], stress_index[1]],
    },
    index=["Scenario return", "Scenario max drawdown"],
)
st.dataframe(
    stress_metrics.style.format("{:.2%}"),
    width="stretch",
    alt="Illustrative crisis scenario comparison",
)

st.subheader("Current allocation")
allocation = pd.DataFrame(
    {
        "Equal-weight": {
            ticker: 1 / len(returns.columns.drop(SAFE_HAVEN))
            if ticker != SAFE_HAVEN
            else 0.0
            for ticker in centrality
        },
        "AI risk-adjusted": apply_regime_filter(
            allocate(graph_centrality(graph), allocation_aggressiveness),
            stress_active,
        ),
    }
)
st.metric(
    "Regime filter",
    "CASH UNLOCKED" if stress_active else "EQUITIES ONLY",
    f"{current_volatility:.2%} current vs {volatility_threshold:.2%} 90th percentile",
)
st.dataframe(
    allocation.style.format("{:.2%}"),
    width="stretch",
    alt="Current portfolio allocation",
)

fig, ax = plt.subplots(figsize=(7, 4))
nx.draw(
    graph,
    with_labels=True,
    node_color="#ff4b4b",
    node_size=[centrality[ticker] * 3000 for ticker in graph],
    font_color="white",
    edge_color="gray",
    ax=ax,
)
st.pyplot(fig)
plt.close(fig)
