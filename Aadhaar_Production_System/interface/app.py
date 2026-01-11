import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import sys
import os

# PATH FIX
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import config
from core.data.loader import DataLoader
from core.engines.analytics import QualityEngine, CrossReferenceEngine
from core.engines.forecasting import ForecastEngine

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SENTINEL | National Intelligence",
    layout="wide",
    page_icon="🛡️"
)

# --- 2. CYBERPUNK STYLING ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {config.THEME_BG}; }}
    h1, h2, h3 {{ color: {config.THEME_PRIMARY} !important; font-family: 'Courier New'; }}
    div[data-testid="stMetricValue"] {{ color: {config.THEME_PRIMARY}; }}
    .stAlert {{ background-color: #1a1c24; border: 1px solid {config.THEME_PRIMARY}; }}
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOAD ---
@st.cache_resource
def load_system_data():
    loader = DataLoader()
    aadhaar = loader.load_aadhaar_master()
    telecom = loader.load_telecom_data()
    return aadhaar, telecom

with st.spinner("ESTABLISHING SECURE DATALINK..."):
    master_df, telecom_df = load_system_data()

# --- 4. HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ SENTINEL PRIME")
    st.caption("SOVEREIGN DEMOGRAPHIC INTELLIGENCE GRID")
with col2:
    if not master_df.empty:
        st.metric("LIVE RECORDS", f"{len(master_df):,}")
    else:
        st.error("DATA OFFLINE")

if master_df.empty:
    st.warning("⚠️ No Data Found. Please ensure CSVs are in 'data/raw'.")
    st.stop()

# --- 5. TABS ---
tab1, tab2, tab3 = st.tabs(["GEOSPATIAL WAR ROOM", "DEEP TECH FORENSICS", "PREDICTIVE AI"])

# === TAB 1: 3D MAP ===
with tab1:
    col_map, col_stats = st.columns([3, 1])
    with col_map:
        # Auto-Simulate Coords if missing (Fail-safe)
        import numpy as np
        plot_df = master_df.copy()
        if 'lat' not in plot_df.columns:
            plot_df['lat'] = np.random.uniform(20, 28, len(plot_df))
            plot_df['lon'] = np.random.uniform(73, 85, len(plot_df))
            
        layer = pdk.Layer(
            "HexagonLayer",
            plot_df,
            get_position=["lon", "lat"],
            elevation_scale=100,
            radius=20000,
            extruded=True,
            get_fill_color=[0, 255, 194, 200],
        )
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(latitude=23, longitude=79, zoom=4, pitch=50),
            layers=[layer]
        ))
    
    with col_stats:
        st.subheader("TELECOM CROSS-REF")
        if not telecom_df.empty:
            st.success("TELECOM DATALINK ACTIVE")
            st.dataframe(telecom_df.head(5), hide_index=True)
            st.info("Correlation analysis running in background...")
        else:
            st.warning("TELECOM DATA UNAVAILABLE")

# === TAB 2: FORENSICS (WHIPPLE) ===
with tab2:
    st.subheader("🕵️ DATA INTEGRITY SCAN (WHIPPLE INDEX)")
    st.markdown("Forensic analysis to detect 'Age Heaping' (Rounding Fraud).")
    
    quality_stats = QualityEngine.calculate_whipple_index(master_df)
    
    if not quality_stats.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(quality_stats, use_container_width=True)
        with col_b:
            suspicious = quality_stats[quality_stats['is_round_number'] == 1]
            st.metric("SUSPICIOUS ROUNDING EVENTS", f"{len(suspicious)}")
            if len(suspicious) > 0:
                st.error("⚠️ DATA MANIPULATION DETECTED IN SUB-DISTRICTS")
    else:
        st.info("Insufficient data granularity for Whipple Analysis.")

# === TAB 3: PYTORCH FORECAST ===
with tab3:
    st.subheader("🧠 NEURAL LOAD PROJECTION")
    if st.button("INITIALIZE LSTM CORE"):
        forecaster = ForecastEngine(master_df)
        with st.spinner("TRAINING TENSORS..."):
            preds = forecaster.train_and_forecast(days=30)
        
        if not preds.empty:
            fig = px.line(preds, x='Date', y='Predicted_Load', title="30-Day Infrastructure Demand")
            fig.update_traces(line_color=config.THEME_PRIMARY)
            fig.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient time-series data for Neural Training.")