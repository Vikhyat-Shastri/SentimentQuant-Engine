"""
Streamlit Dashboard for Fear & Greed Sentiment Analysis Engine
Displays real-time system statistics, sentiment scores, and signals.

Usage:
    streamlit run tools/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import glob

st.set_page_config(page_title="Sentiment Engine Dashboard", layout="wide")
st.title("🚦 Fear & Greed Sentiment Analysis Dashboard")

# Paths to logs and signals (relative to project root)
project_root = Path(__file__).parent.parent
LOG_DIR = project_root / "logs"
SIGNALS_DIR = project_root / "data" / "signals"

# Sidebar controls
st.sidebar.header("Controls")
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 2, 30, 5)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)

# Display system logs
st.subheader("📋 System Logs")
log_files = sorted(LOG_DIR.glob("*.log"), reverse=True) if LOG_DIR.exists() else []
if log_files:
    latest_log = log_files[0]
    try:
        with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
            logs = f.readlines()[-100:]  # Last 100 lines
        st.text("".join(logs))
        st.caption(f"Showing last 100 lines from: {latest_log.name}")
    except Exception as e:
        st.error(f"Error reading log file: {e}")
else:
    st.info("No log files found in logs/ directory.")

# Display recent signals
st.subheader("🎯 Recent Trading Signals")
signal_files = sorted(SIGNALS_DIR.glob("signals_*.csv"), reverse=True) if SIGNALS_DIR.exists() else []
if signal_files:
    latest_signals = signal_files[0]
    try:
        signals_df = pd.read_csv(latest_signals)
        st.dataframe(signals_df.tail(20), use_container_width=True)
        st.caption(f"Showing last 20 signals from: {latest_signals.name}")
        
        # Signal statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Signals", len(signals_df))
        with col2:
            if 'action' in signals_df.columns:
                buy_signals = len(signals_df[signals_df['action'].str.contains('BUY', na=False)])
                st.metric("Buy Signals", buy_signals)
        with col3:
            if 'confidence' in signals_df.columns:
                avg_confidence = signals_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    except Exception as e:
        st.error(f"Error reading signals file: {e}")
else:
    st.info("No signal files found in data/signals/ directory.")
    st.caption("Run scripts/main.py to generate signals.")

# Display sentiment summary
st.subheader("📊 Sentiment Summary")
if signal_files and len(signals_df) > 0:
    # Calculate from actual signals
    if 'fear_greed_index' in signals_df.columns:
        latest_fgi = signals_df['fear_greed_index'].iloc[-1]
        avg_fgi = signals_df['fear_greed_index'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest Fear & Greed Index", f"{latest_fgi:.1f}")
            if latest_fgi < 30:
                st.warning("🔴 Extreme Fear")
            elif latest_fgi < 45:
                st.info("🟡 Fear")
            elif latest_fgi < 55:
                st.success("🟢 Neutral")
            elif latest_fgi < 70:
                st.info("🟡 Greed")
            else:
                st.warning("🔴 Extreme Greed")
        with col2:
            st.metric("Average FGI", f"{avg_fgi:.1f}")
    else:
        st.info("Fear & Greed Index not available in signals.")
else:
    # Mock data if no signals available
    st.info("No sentiment data available. This is mock data.")
    sentiment_data = {
        "Metric": ["Fear & Greed Index", "Sentiment Score", "Signal Strength"],
        "Value": [50.0, 0.0, "Neutral"],
        "Status": ["Neutral", "Neutral", "Hold"]
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    st.table(sentiment_df)

st.markdown("---")
st.caption(f"Dashboard refresh interval: {refresh_interval} seconds | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")

# Auto-refresh (only if enabled)
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()