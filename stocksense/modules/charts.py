import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import CHART_TEMPLATE


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="#00D4AA",
                decreasing_line_color="#FF4757",
            )
        ]
    )
    fig.update_layout(
        **CHART_TEMPLATE,
        title=f"{ticker} Price",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def portfolio_pie_chart(allocation_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        allocation_df,
        values="Allocation (%)",
        names="Asset Class",
        title="Suggested Portfolio Allocation",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**CHART_TEMPLATE)
    return fig


def line_chart(
    df: pd.DataFrame, ticker: str, column: str = "Close"
) -> go.Figure:
    fig = go.Figure(
        data=[go.Scatter(x=df.index, y=df[column], mode="lines", name=ticker)]
    )
    fig.update_layout(**CHART_TEMPLATE, title=f"{ticker} {column}")
    return fig


def comparison_chart(comparison_df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=comparison_df["Symbol"],
                y=comparison_df[metric],
                name=metric,
            )
        ]
    )
    fig.update_layout(**CHART_TEMPLATE, title=f"{metric} Comparison")
    return fig
