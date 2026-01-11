
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import sys
import os
import numpy as np

# ENTERPRISE PATH HACK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import config
from core.etl.ingest import IngestionEngine
from core.models.lstm import ForecastEngine
from core.analytics.forensics import ForensicEngine

# 1. SETUP
st.set_page_config(page_title="SENTINEL", layout="wide", page_icon="🛡️")

# 2. SOVEREIGN THEME
st.markdown(f"""
<style>
    .stApp {{ background-color: {config.THEME_BG}; }}
    h1, h2, h3 {{ color: {config.THEME_PRIMARY} !important; font-family: 'Courier New', monospace; }}
    div[data-testid="stMetricValue"] {{ color: {config.THEME_PRIMARY}; }}
    .stDataFrame {{ border: 1px solid #333; }}
</style>
""", unsafe_allow_html=True)

# 3. DATA LOAD
@st.cache_resource
def load_data():
    engine = IngestionEngine()
    return engine.load_master_index(), engine.load_telecom_index()

with st.spinner("ESTABLISHING SECURE DATALINK..."):
    df, telecom_df = load_data()

# 4. HEADER
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ AADHAAR SENTINEL")
    st.caption("NATIONAL DEMOGRAPHIC INTELLIGENCE GRID | PRODUCTION BUILD v1.1")
with col2:
    if not df.empty:
        st.metric("LIVE RECORDS", f"{len(df):,}")
    else:
        st.error("DATA OFFLINE")

if df.empty:
    st.error("⚠️ NO DATA FOUND. Please check 'data/raw' folder.")
    st.stop()

# 5. TABS
tab1, tab2, tab3 = st.tabs(["GEOSPATIAL COMMAND", "DEEP FORENSICS", "NEURAL PREDICTION"])

# === TAB 1: 3D MAP ===
with tab1:
    col_vis, col_info = st.columns([3, 1])
    with col_vis:
        plot_df = df.groupby(['state', 'district']).agg({
            'total_activity': 'sum', 'lat': 'mean', 'lon': 'mean'
        }).reset_index()
        
        layer = pdk.Layer(
            "HexagonLayer",
            plot_df,
            get_position=["lon", "lat"],
            elevation_scale=100,
            radius=20000,
            extruded=True,
            pickable=True,
            get_fill_color=[0, 255, 194, 200],
        )
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(latitude=23, longitude=79, zoom=4, pitch=50),
            layers=[layer]
        ))
    
    with col_info:
        st.subheader("TELECOM CROSS-REF")
        if not telecom_df.empty:
            st.success("TRAI DATABASE: ONLINE")
            # Fixed Deprecation Warning
            st.dataframe(telecom_df.head(10), width=500)
        else:
            st.warning("TRAI DATABASE: OFFLINE")

# === TAB 2: FORENSICS ===
with tab2:
    st.subheader("🕵️ WHIPPLE INDEX ANALYSIS")
    st.markdown("Statistical detection of Age Heaping (Data Fraud).")
    
    whipple_stats = ForensicEngine.calculate_whipple(df)
    
    if not whipple_stats.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            # Fixed Deprecation Warning
            st.dataframe(whipple_stats, width=800)
        with col_b:
            heaped = whipple_stats[whipple_stats['is_suspicious'] == 1]
            st.metric("ANOMALOUS DISTRICTS", f"{len(heaped)}")
            if len(heaped) > 0:
                st.error(f"Potential manipulation in {len(heaped)} districts.")
    else:
        st.info("Insufficient granularity for forensic scan.")

# === TAB 3: NEURAL NET ===
with tab3:
    st.subheader("🧠 PYTORCH LSTM FORECAST")
    if st.button("RUN TENSOR COMPUTATION"):
        forecaster = ForecastEngine(df)
        with st.spinner("TRAINING NEURAL NETWORK..."):
            preds = forecaster.generate_forecast(days=30)
        
        if not preds.empty:
            fig = px.line(preds, x='Date', y='Predicted_Load', title="30-Day Infrastructure Load")
            fig.update_traces(line_color=config.THEME_PRIMARY)
            fig.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
            # Fixed Deprecation Warning
            st.plotly_chart(fig, width=1200)
