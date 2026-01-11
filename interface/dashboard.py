import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import sys
import os
import numpy as np
import time

# ENTERPRISE PATH HACK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import config
from core.etl.ingest import IngestionEngine
from core.models.lstm import ForecastEngine
from core.analytics.forensics import ForensicEngine
from core.analytics.segmentation import SegmentationEngine 

# 1. SETUP
st.set_page_config(
    page_title="SENTINEL COMMAND", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# 2. SOVEREIGN THEME
st.markdown(f"""
<style>
    .stApp {{ background-color: {config.THEME_BG}; }}
    h1, h2, h3 {{ color: {config.THEME_PRIMARY} !important; font-family: 'Courier New', monospace; text-shadow: 0 0 10px {config.THEME_PRIMARY}; }}
    div[data-testid="stMetricValue"] {{ color: {config.THEME_PRIMARY}; font-family: 'Courier New'; font-weight: bold; }}
    .stDataFrame {{ border: 1px solid #333; }}
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; background-color: #111; border-radius: 5px; color: white; }}
    .stTabs [aria-selected="true"] {{ background-color: {config.THEME_PRIMARY}; color: black !important; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)

# 3. DATA LOAD
@st.cache_resource
def load_data():
    engine = IngestionEngine()
    return engine.load_master_index(), engine.load_telecom_index()

with st.spinner("ESTABLISHING SECURE DATALINK..."):
    master_df, telecom_df = load_data()

if master_df.empty:
    st.error("⚠️ CRITICAL ERROR: Data Vault Empty. Check 'data/raw'.")
    st.stop()

# 4. SIDEBAR NAVIGATION (DRILL-DOWN)
with st.sidebar:
    st.title("🛰️ COMMAND NODE")
    
    # View Mode Selection
    view_mode = st.radio("OPERATIONAL SCOPE:", ["🇮🇳 NATIONAL OVERVIEW", "📍 DISTRICT DRILL-DOWN"])
    
    selected_state = None
    selected_district = None
    
    if view_mode == "📍 DISTRICT DRILL-DOWN":
        st.markdown("---")
        # Dynamic State Filter
        states = sorted(master_df['state'].unique())
        selected_state = st.selectbox("SELECT STATE", states)
        
        # Dynamic District Filter
        districts = sorted(master_df[master_df['state'] == selected_state]['district'].unique())
        selected_district = st.selectbox("SELECT DISTRICT", districts)
        
    st.markdown("---")
    st.markdown("### ⚙️ SIMULATION PARAMS")
    forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
    sensitivity = st.slider("Anomaly Sensitivity", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    if st.button("🔄 REBOOT SYSTEM"):
        st.cache_resource.clear()
        st.rerun()

# 5. DATA FILTERING LOGIC
if view_mode == "📍 DISTRICT DRILL-DOWN" and selected_district:
    # Filter Data for Specific District
    active_df = master_df[
        (master_df['state'] == selected_state) & 
        (master_df['district'] == selected_district)
    ].copy()
    display_title = f"DISTRICT COMMAND: {selected_district.upper()}"
else:
    # Use Full Data
    active_df = master_df.copy()
    display_title = "NATIONAL INTELLIGENCE GRID"

# 6. HEADER METRICS
st.title(display_title)
m1, m2, m3, m4 = st.columns(4)
m1.metric("ACTIVE RECORDS", f"{len(active_df):,}")
m2.metric("TOTAL ACTIVITY", f"{int(active_df['total_activity'].sum()):,}")
m3.metric("DATA SOURCES", f"{active_df['state'].nunique()} States")
m4.metric("SYSTEM STATUS", "ONLINE", delta="SECURE")

# 7. MAIN INTERFACE TABS (EXPANDED TO 6 TABS)
tabs = st.tabs([
    "🗺️ GEOSPATIAL MAP", 
    "📈 AI FORECASTING", 
    "🚨 ANOMALY GRID", 
    "🧩 SEGMENTATION",
    "🔮 SIMULATOR",
    "🤖 POLICY ADVISOR"
])

# === TAB 1: REAL-TIME DEMOGRAPHIC DENSITY MAP ===
with tabs[0]:
    col_map, col_stats = st.columns([3, 1])
    
    with col_map:
        # Aggregation for Map
        map_data = active_df.groupby(['state', 'district']).agg({
            'total_activity': 'sum', 
            'lat': 'mean', 
            'lon': 'mean'
        }).reset_index()
        
        # Determine Zoom Level based on Scope
        zoom_level = 4 if view_mode == "🇮🇳 NATIONAL OVERVIEW" else 9
        
        # Heatmap Layer (Density) + Column Layer (Volume)
        layers = [
            pdk.Layer(
                "HeatmapLayer",
                map_data,
                get_position=["lon", "lat"],
                get_weight="total_activity",
                radius_pixels=60,
                intensity=1,
                threshold=0.3
            ),
            pdk.Layer(
                "ColumnLayer",
                map_data,
                get_position=["lon", "lat"],
                get_elevation="total_activity",
                elevation_scale=50,
                radius=10000 if view_mode == "🇮🇳 NATIONAL OVERVIEW" else 2000,
                get_fill_color=[0, 255, 194, 200],
                pickable=True,
                extruded=True
            )
        ]
        
        # Interactive Tooltip
        tooltip = {"html": "<b>{district}</b><br/>Activity: {total_activity}", "style": {"background": "black", "color": "white"}}
        
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(
                latitude=map_data['lat'].mean(), 
                longitude=map_data['lon'].mean(), 
                zoom=zoom_level, 
                pitch=50
            ),
            layers=layers,
            tooltip=tooltip
        ))
        
    with col_stats:
        st.markdown("### 📡 DENSITY SIGNALS")
        top_districts = map_data.sort_values('total_activity', ascending=False).head(10)
        # FIXED: Removed use_container_width warning
        st.dataframe(top_districts[['district', 'total_activity']], hide_index=True)
        
        st.info("Visualizes real-time enrolment velocity across the grid. High pillars represent critical load centers.")

# === TAB 2: AI RESOURCE DEMANDING FORECASTING ===
with tabs[1]:
    st.subheader("🧠 PREDICTIVE INFRASTRUCTURE SCALING (LSTM)")
    
    if len(active_df) > 50:
        forecaster = ForecastEngine(active_df)
        
        with st.spinner(f"CALCULATING RESOURCE DEMAND FOR NEXT {forecast_days} DAYS..."):
            resource_pred = forecaster.calculate_resource_demand(days=forecast_days)
            
        if not resource_pred.empty:
            c1, c2 = st.columns([3, 1])
            
            with c1:
                # Advanced Plotly Chart
                fig = go.Figure()
                
                # Confidence Tunnel
                fig.add_trace(go.Scatter(
                    x=resource_pred['Date'].tolist() + resource_pred['Date'].tolist()[::-1],
                    y=resource_pred['Upper_Bound'].tolist() + resource_pred['Lower_Bound'].tolist()[::-1],
                    fill='toself', fillcolor='rgba(0, 255, 157, 0.1)', line=dict(color='rgba(0,0,0,0)'),
                    name='Demand Uncertainty'
                ))
                
                # Actual Demand
                fig.add_trace(go.Scatter(
                    x=resource_pred['Date'], y=resource_pred['Predicted_Load'],
                    mode='lines', name='Projected Load', line=dict(color=config.THEME_PRIMARY, width=3)
                ))
                
                fig.update_layout(
                    title="Projected Transaction Volume",
                    template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG,
                    hovermode="x", height=500
                )
                # FIXED: Updated parameter
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                # Resource Cards
                peak_load = resource_pred['Upper_Bound'].max()
                required_servers = resource_pred['Required_Server_Units'].max()
                required_staff = resource_pred['Required_Manpower'].max()
                
                st.markdown("### 🏗️ RESOURCE REQ.")
                st.metric("PEAK LOAD (Est.)", f"{int(peak_load):,}")
                st.divider()
                st.metric("🖥️ SERVERS NEEDED", f"{int(required_servers)}")
                st.metric("👥 STAFF NEEDED", f"{int(required_staff)}")
                
                if required_servers > 50:
                    st.error("CRITICAL LOAD EXPECTED.")
                else:
                    st.success("INFRASTRUCTURE STABLE.")
    else:
        st.warning("Insufficient data points for Neural Network training in this view.")

# === TAB 3: ANOMALY DETECTION GRID ===
with tabs[2]:
    st.subheader("🚨 OPERATIONAL INTEGRITY GRID")
    
    # 1. Forensic Scan
    forensics = ForensicEngine.calculate_whipple(active_df)
    
    if not forensics.empty:
        # Filter anomalies
        anomalies = forensics[forensics['is_suspicious'] == 1]
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("TOTAL SCANNED", len(forensics))
        kpi2.metric("FLAGGED ENTITIES", len(anomalies), delta="-RISK")
        risk_ratio = (len(anomalies)/len(forensics))*100 if len(forensics) > 0 else 0
        kpi3.metric("RISK RATIO", f"{risk_ratio:.1f}%")
        
        st.markdown("### ⚠️ ANOMALY VECTOR LIST")
        if not anomalies.empty:
            # FIXED: Updated applymap to map to fix FutureWarning
            st.dataframe(
                anomalies.style.map(lambda x: "background-color: #330000; color: #ff4b4b", subset=['total_activity']),
                use_container_width=True
            )
        else:
            st.success("NO ANOMALIES DETECTED IN CURRENT SECTOR.")
            
        # Benford Analysis (Aggregated)
        benford_df, is_bad = ForensicEngine.calculate_benfords_law(active_df)
        if not benford_df.empty:
            st.markdown("### 🔢 DIGITAL FORENSICS (BENFORD'S LAW)")
            fig_b = px.bar(benford_df, x='Digit', y=['Expected_Benford', 'Observed_Real'], barmode='group',
                          color_discrete_map={'Expected_Benford': 'gray', 'Observed_Real': config.THEME_PRIMARY})
            fig_b.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_b, use_container_width=True)

# === TAB 4: DISTRICT SEGMENTATION (CLUSTERING) ===
with tabs[3]:
    st.subheader("🧩 BEHAVIORAL SEGMENTATION MAP")
    
    if len(active_df) > 10:
        with st.spinner("RUNNING K-MEANS CLUSTERING ALGORITHMS..."):
            segmented_df = SegmentationEngine.segment_districts(active_df)
            
        if not segmented_df.empty:
            row1, row2 = st.columns([3, 1])
            
            with row1:
                # SAFEGUARD: Check if clustering succeeded
                if 'cluster_label' in segmented_df.columns:
                    fig_c = px.scatter(
                        segmented_df, 
                        x="total_volume", 
                        y="volatility", 
                        color="cluster_label",
                        size="daily_avg",
                        hover_data=['state', 'district'],
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        title="District Clustering: Volume vs. Volatility"
                    )
                    fig_c.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.warning("Data insufficient for cluster convergence.")
                
            with row2:
                st.markdown("### 🏷️ CLUSTER LEGEND")
                if 'cluster_label' in segmented_df.columns:
                    counts = segmented_df['cluster_label'].value_counts()
                    st.write(counts)
                
                st.info("Segments districts into operational categories (Hubs, Dormant, Steady) using Unsupervised Learning.")
    else:
        st.warning("Need more data points for segmentation analysis. Switch to National View.")

# === TAB 5: SCENARIO SIMULATOR ===
with tabs[4]:
    st.subheader("🔮 STRATEGIC WARGAMES SIMULATOR")
    st.markdown("Simulate the impact of policy shifts or migration waves on infrastructure.")
    
    col_sim_input, col_sim_output = st.columns([1, 2])
    
    with col_sim_input:
        st.markdown("#### 🎛️ SIMULATION CONTROLS")
        sim_migration = st.slider("Migration Surge (%)", 0, 100, 10)
        sim_policy = st.selectbox("Policy Trigger", ["None", "Mandatory Update Drive", "New Benefit Scheme"])
        
        policy_multiplier = 1.0
        if sim_policy == "Mandatory Update Drive":
            policy_multiplier = 1.5
        elif sim_policy == "New Benefit Scheme":
            policy_multiplier = 1.2
            
        total_impact_factor = (1 + (sim_migration / 100)) * policy_multiplier
        
        st.metric("IMPACT MULTIPLIER", f"{total_impact_factor:.2f}x")
        
        if st.button("RUN SIMULATION"):
            st.session_state['run_sim'] = True
            
    with col_sim_output:
        if st.session_state.get('run_sim', False):
            baseline_load = active_df['total_activity'].sum()
            simulated_load = baseline_load * total_impact_factor
            gap = np.ceil(simulated_load / 5000) - np.ceil(baseline_load / 5000)
            
            st.markdown("#### 📊 SIMULATION RESULTS")
            s1, s2, s3 = st.columns(3)
            s1.metric("CURRENT LOAD", f"{int(baseline_load):,}")
            s2.metric("SIMULATED LOAD", f"{int(simulated_load):,}", delta=f"+{int(simulated_load-baseline_load):,}")
            s3.metric("INFRA GAP", f"{int(gap)} Units", delta_color="inverse")
            
            sim_data = pd.DataFrame({'Scenario': ['Baseline', 'Simulated'], 'Load': [baseline_load, simulated_load]})
            fig_sim = px.bar(sim_data, x='Scenario', y='Load', color='Scenario', 
                            color_discrete_map={'Baseline': 'gray', 'Simulated': config.THEME_PRIMARY})
            fig_sim.update_layout(template="plotly_dark", plot_bgcolor=config.THEME_BG, paper_bgcolor=config.THEME_BG)
            st.plotly_chart(fig_sim, use_container_width=True)

# === TAB 6: AI POLICY ADVISOR (NEW) ===
with tabs[5]:
    st.subheader("🤖 AUTOMATED POLICY RECOMMENDATION ENGINE")
    st.markdown("Generates strategic directives based on real-time forensic and predictive analysis.")
    
    if st.button("GENERATE STRATEGIC BRIEF"):
        with st.spinner("ANALYZING VECTORS..."):
            time.sleep(1) # UX Pause
            
            # Logic for text generation
            total_vol = active_df['total_activity'].sum()
            risk_count = len(forensics[forensics['is_suspicious']==1]) if not forensics.empty else 0
            
            # Dynamic Policy Text
            policy_text = f"""
            **EXECUTIVE SUMMARY:**
            The grid is currently processing {int(total_vol):,} transactions. 
            Operational risk is detected in {risk_count} sectors.
            
            **STRATEGIC DIRECTIVES:**
            """
            
            if risk_count > 5:
                policy_text += "\n1. 🔴 **INITIATE AUDIT:** High fraud risk detected. Deploy mobile vigilance teams to flagged districts immediately."
            else:
                policy_text += "\n1. 🟢 **MAINTAIN VIGILANCE:** Fraud vectors are within acceptable tolerance."
                
            if total_vol > 500000:
                policy_text += "\n2. 🟡 **INFRASTRUCTURE SCALE:** High load volume requires server cluster expansion by 20%."
            
            st.markdown(f"""
            <div style="background-color: #1a1a1a; padding: 20px; border-left: 5px solid {config.THEME_PRIMARY};">
                {policy_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📥 DOWNLOAD INTEL")
            csv = active_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "DOWNLOAD CLASSIFIED REPORT (CSV)",
                csv,
                "sentinel_intel_report.csv",
                "text/csv",
                key='download-csv'
            )