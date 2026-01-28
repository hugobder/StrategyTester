import streamlit as st
import plotly.express as px
import numpy as np

from data import load_ohlcv, resample_ohlcv
from strategies.sma_cross import sma_cross_position
from strategies.donchian_atr import donchian_atr_position
from strategies.volatility_activity_surface import volatility_activity_surface

from backtest.execution import simulate_next_open_risk_based
from backtest.metrics import compute_report


st.set_page_config(page_title="Strategy Tester", layout="wide")
st.title("Strategy Tester")

with st.sidebar:
    st.header("Data")
    path = st.text_input("CSV/Parquet path", "data/EURUSD_5m.csv")
    resample_rule = st.selectbox("Resample (optional)", ["(none)", "5min", "15min", "1H", "1D"], index=0)

    st.header("Strategy")
    strategy = st.selectbox("Strategy", ["SMA Cross", "Donchian + ATR", "Volatility Activity (3D)"], index=0)

    if strategy == "SMA Cross":
        fast = st.slider("Fast SMA", 2, 200, 20)
        slow = st.slider("Slow SMA", 5, 400, 50)

    elif strategy == "Donchian + ATR":
        breakout_n = st.slider("Donchian breakout (bars)", 5, 200, 20)
        atr_n = st.slider("ATR period", 5, 100, 14)
        atr_mult = st.number_input("ATR multiplier", value=2.0, step=0.1)
        trend_sma = st.slider("Trend filter SMA (0 = off)", 0, 400, 200)

    elif strategy == "Volatility Activity (3D)":
        max_lag = st.slider("Max lag", 5, 200, 60)
        window = st.slider("Rolling window", 50, 2000, 200)
        stride = st.slider("Stride (speed vs detail)", 1, 50, 5)
        mode = st.selectbox("Activity mode", ["abs", "sq"], index=0)

    st.header("Risk")
    initial_capital = st.number_input("Initial Capital", value=10_000.0, step=1000.0)
    risk_per_trade = st.number_input("Risk per trade (%)", value=1.0, step=0.1) / 100.0
    stop_distance = st.number_input("Stop distance (price units)", value=100.0, step=10.0)

    st.header("Execution")
    spread = st.number_input("Spread (price units)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    slippage = st.number_input("Slippage (price units)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")

run = st.button("Run backtest")

if run:
    df = load_ohlcv(path)

    if strategy == "Volatility Activity (3D)":
        lags, ticks, Z = volatility_activity_surface(
            df,
            max_lag=max_lag,
            window=window,
            stride=stride,
            mode=mode,
        )

        st.subheader("Volatility Activity Surface (3D)")

        # Plotly surface expects (y, x) mapping depending on how you pass arrays.
        # We'll use: X=Lag, Y=Tick, Z=Activity
        fig = px.surface(
            x=lags,
            y=ticks,
            z=Z.T,  # transpose to match y rows, x cols
            labels={"x": "Lag", "y": "Tick", "z": "Activity"},
        )

        # Make it look more like the “quant vibe”
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_title="Lag",
                yaxis_title="Tick",
                zaxis_title="Activity",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.stop()

    if resample_rule != "(none)":
        df = resample_ohlcv(df, resample_rule)

    if strategy == "SMA Cross":
        pos = sma_cross_position(df, fast=fast, slow=slow)
        stop_dist = stop_distance  # fixed float from sidebar

    else:  # Donchian + ATR
        pos, stop_dist = donchian_atr_position(
            df,
            breakout_n=breakout_n,
            atr_n=atr_n,
            atr_mult=atr_mult,
            trend_filter_sma=(None if trend_sma == 0 else trend_sma),
        )

    trades = simulate_next_open_risk_based(
        df,
        pos,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        stop_distance=stop_dist,   # float for SMA, Series for Donchian
        spread=spread,
        slippage=slippage,
    )

    report = compute_report(trades, initial_capital)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net PnL", f"{report['net_pnl']:.2f}")
    c2.metric("Return", f"{report['return_pct']:.2%}")
    c3.metric("Max Drawdown", f"{report['max_dd']:.2%}")
    c4.metric("Profit Factor", f"{report['profit_factor']:.2f}")
    c5.metric("Expectancy (R)", f"{report['expectancy_r']:.2f}")


    left, right = st.columns(2)

    with left:
        st.subheader("Equity (cumulative PnL)")
        if len(report["equity"]):
            fig = px.line(report["equity"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades generated.")

    with right:
        st.subheader("Drawdown")
        if len(report["drawdown"]):
            fig = px.line(report["drawdown"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drawdown data.")

    st.subheader("Trades")
    st.dataframe(report["trades"], use_container_width=True)
