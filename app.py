import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from run_engine import allocate


FEATURE_COLUMNS = ["vol_5d", "vol_20d", "momentum"]
WINDOW = 60
MIN_TRAIN = 120
HORIZON = 5
REBALANCE = 20
SENSITIVITY = 1.0
AGGRESSIVENESS = 3.0
SAFE_HAVEN = "Cash"
DISPLAY_NAMES = {
    "0700.HK": "Tencent Holdings",
    "9988.HK": "Alibaba Group",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi Corporation",
    "9618.HK": "JD.com",
    SAFE_HAVEN: SAFE_HAVEN,
}

st.set_page_config(layout="wide")
st.title("AI-driven dynamic index")
st.caption("Walk-forward crisis protection versus an equal-weight index.")


def load_returns() -> pd.DataFrame:
    prices = pd.read_csv("price_matrix.csv", index_col=0, parse_dates=True)
    return prices.pct_change().dropna()


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    market = returns.drop(columns=SAFE_HAVEN, errors="ignore").mean(axis=1)
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
    correlations = returns.corr().fillna(0)
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
    returns: pd.DataFrame,
) -> pd.DataFrame:
    features = build_features(returns)
    equal = 1 / returns.shape[1]
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
            (predicted_vol / train["target"].median() - 1) * SENSITIVITY, 0, 1
        )

        graph = build_graph(returns.iloc[position - WINDOW : position], crisis_level)
        risk = allocate(graph_centrality(graph), AGGRESSIVENESS)
        weights = {
            ticker: (1 - crisis_level) * equal + crisis_level * risk[ticker]
            for ticker in returns.columns
        }
        risk_returns = returns.drop(columns=SAFE_HAVEN, errors="ignore")
        realized = risk_returns.iloc[position : position + HORIZON].mean(axis=1)
        crisis = realized.std() > risk_returns.mean(axis=1).rolling(20).std().quantile(
            0.75
        )
        for day, row in returns.iloc[position : position + HORIZON].iterrows():
            portfolio_returns.append(
                (day, float(row @ pd.Series(weights)), float(row.mean()), crisis)
            )

    return pd.DataFrame(
        portfolio_returns,
        columns=["date", "ai", "index", "crisis"],
    ).set_index("date")


def performance(returns: pd.Series) -> tuple[float, float]:
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    return float(curve.iloc[-1] - 1), float(drawdown.min())


def performance_chart(curves: pd.DataFrame) -> alt.Chart:
    data = curves.rename_axis("date").reset_index().melt(
        id_vars="date", var_name="portfolio", value_name="value"
    )
    color = alt.Color(
        "portfolio:N",
        scale=alt.Scale(
            domain=["AI portfolio", "Equal-weight index"],
            range=["#5b3fd1", "#8fb4ff"],
        ),
        legend=alt.Legend(title=None),
    )
    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y(
            "value:Q",
            title="Growth of $1",
            scale=alt.Scale(domainMin=0.8),
        ),
        color=color,
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("portfolio:N", title="Portfolio"),
            alt.Tooltip("value:Q", title="Value", format=".3f"),
        ],
    )
    return base.mark_line(size=3).encode(
        strokeDash=alt.condition(
            alt.datum.portfolio == "Equal-weight index",
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
st.sidebar.caption(
    f"Available history: {available_start:%Y-%m-%d} to "
    f"{available_end:%Y-%m-%d}"
)
st.sidebar.markdown(
    "**Basket companies:** Tencent Holdings, Alibaba Group, Meituan, "
    "Xiaomi Corporation, JD.com, and Cash"
)
if start_date >= end_date:
    st.error("Choose a backtest range with at least two trading days.")
    st.stop()
returns = returns.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
if len(returns) <= MIN_TRAIN + HORIZON:
    st.error(
        f"Choose at least {MIN_TRAIN + HORIZON + 1} trading days for the "
        "walk-forward backtest."
    )
    st.stop()
results = backtest(returns)
ai_return, ai_drawdown = performance(results["ai"])
index_return, index_drawdown = performance(results["index"])
crisis_results = results[results["crisis"]]
ai_crisis, index_crisis = performance(crisis_results["ai"]), performance(
    crisis_results["index"]
)

st.subheader("Walk-forward backtest")
st.caption(
    "Each prediction uses only data available before that rebalance. "
    "The index is an equal-weight portfolio."
)
chart = results[["ai", "index"]].add(1).cumprod()
chart.columns = ["AI portfolio", "Equal-weight index"]
st.altair_chart(
    performance_chart(chart),
    width="stretch",
    theme=None,
)

spread = chart["AI portfolio"] - chart["Equal-weight index"]
st.area_chart(
    (spread * 100).rename("AI advantage versus index (%)"),
    y_label="Percentage-point difference",
    width="stretch",
)

metrics = pd.DataFrame(
    {
        "AI portfolio": [ai_return, ai_drawdown, ai_crisis[0]],
        "Equal-weight index": [index_return, index_drawdown, index_crisis[0]],
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

graph = build_graph(returns.tail(WINDOW), SENSITIVITY)
centrality = dict(graph.degree(weight="weight"))

st.subheader("Current allocation")
allocation = pd.DataFrame(
    {
        "Equal-weight": 1 / len(centrality),
        "AI risk-adjusted": allocate(
            graph_centrality(graph), AGGRESSIVENESS
        ),
    }
)
allocation.index = allocation.index.map(DISPLAY_NAMES)
st.dataframe(
    allocation.style.format("{:.2%}"),
    width="stretch",
    alt="Current portfolio allocation",
)

fig, ax = plt.subplots(figsize=(7, 4))
display_graph = nx.relabel_nodes(graph, DISPLAY_NAMES)
nx.draw(
    display_graph,
    with_labels=True,
    node_color="#ff4b4b",
    node_size=[centrality[ticker] * 3000 for ticker in graph],
    font_color="white",
    edge_color="gray",
    ax=ax,
)
st.pyplot(fig)
plt.close(fig)
