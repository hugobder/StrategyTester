import streamlit as st
import plotly.express as px

from data import load_ohlcv, resample_ohlcv
from strategies.sma_cross import sma_cross_position
from backtest.execution import simulate_next_open
from backtest.metrics import compute_report


st.set_page_config(page_title="Strategy Tester", layout="wide")
st.title("Strategy Tester")

with st.sidebar:
    st.header("Data")
    path = st.text_input("CSV/Parquet path", "data/EURUSD_5m.csv")
    resample_rule = st.selectbox("Resample (optional)", ["(none)", "5min", "15min", "1H", "1D"], index=0)

    st.header("Strategy: SMA Cross")
    fast = st.slider("Fast SMA", 2, 200, 20)
    slow = st.slider("Slow SMA", 5, 400, 50)

    st.header("Execution")
    qty = st.number_input("Qty", min_value=0.0001, value=1.0, step=1.0)
    spread = st.number_input("Spread (price units)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
    slippage = st.number_input("Slippage (price units)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")

run = st.button("Run backtest")

if run:
    df = load_ohlcv(path)

    if resample_rule != "(none)":
        df = resample_ohlcv(df, resample_rule)

    pos = sma_cross_position(df, fast=fast, slow=slow)
    trades = simulate_next_open(df, pos, qty=qty, spread=spread, slippage=slippage)
    report = compute_report(trades)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net PnL", f"{report['net_pnl']:.4f}")
    c2.metric("Max Drawdown", f"{report['max_dd']:.2%}")
    c3.metric("Win Rate", f"{report['win_rate']:.2%}")
    c4.metric("Profit Factor", f"{report['profit_factor']:.2f}")
    c5.metric("Expectancy", f"{report['expectancy']:.4f}")

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
