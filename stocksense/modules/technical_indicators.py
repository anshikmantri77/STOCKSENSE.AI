import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["MA20"] = result["Close"].rolling(20).mean()
    result["MA50"] = result["Close"].rolling(50).mean()
    result["MA200"] = result["Close"].rolling(200).mean()

    delta = result["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    result["RSI"] = 100 - (100 / (1 + rs))

    result["BB_mid"] = result["Close"].rolling(20).mean()
    std = result["Close"].rolling(20).std()
    result["BB_upper"] = result["BB_mid"] + 2 * std
    result["BB_lower"] = result["BB_mid"] - 2 * std

    ema12 = result["Close"].ewm(span=12, adjust=False).mean()
    ema26 = result["Close"].ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_hist"] = result["MACD"] - result["MACD_signal"]

    return result


def create_technical_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} - Price & MAs", "RSI", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00D4AA", decreasing_line_color="#FF4757",
            name="Price",
        ),
        row=1, col=1,
    )

    for ma, color, width in [
        ("MA20", "#FFA500", 1), ("MA50", "#3498DB", 1.5), ("MA200", "#E74C3C", 1.5)
    ]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[ma], mode="lines",
                    name=ma, line=dict(color=color, width=width),
                ),
                row=1, col=1,
            )

    if "BB_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_upper"], mode="lines",
                name="BB Upper", line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_lower"], mode="lines",
                name="BB Lower", line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI"], mode="lines",
            name="RSI", line=dict(color="#9B59B6", width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    colors = ["#00D4AA" if v >= 0 else "#FF4757" for v in df["MACD_hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist", marker_color=colors),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD"], mode="lines",
            name="MACD", line=dict(color="#3498DB", width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD_signal"], mode="lines",
            name="Signal", line=dict(color="#E74C3C", width=1.5),
        ),
        row=3, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F9FAFB", family="Inter"),
        hovermode="x unified",
        showlegend=True,
        height=800,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor="#1F2937", linecolor="#374151")
    fig.update_yaxes(gridcolor="#1F2937", linecolor="#374151")

    return fig
