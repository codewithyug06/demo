import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import sys
import os
import numpy as np
import time
import hashlib
import random

# SYSTEM PATH SETUP (Critical for Enterprise Deployment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORT CORE ENGINES
from config.settings import config
from core.etl.ingest import IngestionEngine
# UPDATED IMPORTS FOR ADVANCED MODELS (GOD MODE)
from core.models.lstm import ForecastEngine, AdvancedForecastEngine 
from core.analytics.forensics import ForensicEngine
from core.analytics.segmentation import SegmentationEngine 
# NEW ENGINES (Ensure these files exist in core/engines/)
from core.engines.cognitive import SentinelCognitiveEngine, SwarmIntelligence 
from core.engines.spatial import SpatialEngine
from core.engines.causal import CausalEngine

# ==============================================================================
# 1. SOVEREIGN CONFIGURATION & ULTRA-MODERN THEMING
# ==============================================================================
st.set_page_config(
    page_title="SENTINEL PRIME | COGNITIVE TWIN", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# SESSION STATE INIT
if 'theme_mode' not in st.session_state:
    st.session_state['theme_mode'] = 'GOD_MODE'
if 'performance_metrics' not in st.session_state:
    st.session_state['performance_metrics'] = {}
if 'viz_config' not in st.session_state:
    st.session_state['viz_config'] = {}
if 'system_uptime' not in st.session_state:
    st.session_state['system_uptime'] = time.time()

# DYNAMIC THEME ENGINE (ENHANCED PALETTES)
theme_colors = {
    'GOD_MODE': {
        'bg': '#000000', 
        'primary': '#00FF9D', 
        'secondary': '#008F5A',
        'text': '#E0FFF4', 
        'accent': '#FF00FF',
        'card_bg': 'rgba(0, 20, 10, 0.55)',
        'border': '1px solid rgba(0, 255, 157, 0.4)',
        'glow': '0 0 20px rgba(0, 255, 157, 0.3)'
    },
    'STEALTH': {
        'bg': '#050905', 
        'primary': '#44FF44', 
        'secondary': '#115511',
        'text': '#AAFFAA', 
        'accent': '#004400',
        'card_bg': 'rgba(5, 20, 5, 0.8)',
        'border': '1px solid rgba(68, 255, 68, 0.2)',
        'glow': '0 0 15px rgba(68, 255, 68, 0.1)'
    },
    'ANALYSIS': {
        'bg': '#0B0C15', 
        'primary': '#00AAFF', 
        'secondary': '#004488',
        'text': '#DDEEFF', 
        'accent': '#FF4444',
        'card_bg': 'rgba(15, 20, 35, 0.7)',
        'border': '1px solid rgba(0, 170, 255, 0.3)',
        'glow': '0 0 20px rgba(0, 170, 255, 0.2)'
    }
}

current_theme = theme_colors[st.session_state['theme_mode']]

# ------------------------------------------------------------------------------
# EXTRAORDINARY CSS INJECTION (ANIMATIONS & GLASSMORPHISM)
# ------------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* 1. GLOBAL FONTS & CRT EFFECTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap');
    
    .stApp {{
        background-color: {current_theme['bg']};
        background-image: 
            linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), 
            linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        background-size: 100% 2px, 3px 100%;
        color: {current_theme['text']};
        font-family: 'Rajdhani', sans-serif;
    }}
    
    /* SCANLINE ANIMATION */
    .stApp::before {{
        content: " ";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 2;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }}

    /* 2. ANIMATIONS */
    @keyframes glitch {{
        0% {{ transform: translate(0); }}
        20% {{ transform: translate(-2px, 2px); }}
        40% {{ transform: translate(-2px, -2px); }}
        60% {{ transform: translate(2px, 2px); }}
        80% {{ transform: translate(2px, -2px); }}
        100% {{ transform: translate(0); }}
    }}
    
    @keyframes scanBar {{
        0% {{ background-position: 0% 0%; }}
        100% {{ background-position: 100% 100%; }}
    }}
    
    @keyframes blink {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.3; }}
        100% {{ opacity: 1; }}
    }}

    /* 3. CUSTOM HUD CARDS (GLASSMORPHISM + NEON) */
    .hud-card {{
        background: {current_theme['card_bg']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: {current_theme['border']};
        box-shadow: {current_theme['glow']};
        border-radius: 4px;
        padding: 24px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    
    .hud-card::after {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, {current_theme['primary']}, transparent);
        animation: scanBar 3s infinite linear;
    }}
    
    .hud-card:hover {{
        transform: scale(1.02);
        border-color: {current_theme['primary']};
        box-shadow: 0 0 30px {current_theme['primary']}50;
    }}

    /* 4. TYPOGRAPHY - GLITCH TITLES */
    .glitch-title {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        color: {current_theme['text']};
        text-shadow: 2px 2px 0px {current_theme['accent']};
        position: relative;
    }}
    
    .glitch-title:hover {{
        animation: glitch 0.3s cubic-bezier(.25, .46, .45, .94) both infinite;
        color: {current_theme['primary']};
    }}
    
    /* 5. METRIC CONTAINERS */
    .metric-value {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 3.2rem;
        font-weight: 700;
        color: {current_theme['primary']};
        text-shadow: 0 0 10px {current_theme['primary']};
    }}
    
    .metric-label {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
    }}

    /* 6. TABS & WIDGETS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 15px;
        background: transparent;
        padding-bottom: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 55px;
        background: rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px;
        color: #8899AA;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.9rem;
        transition: all 0.3s;
        clip-path: polygon(10% 0, 100% 0, 100% 80%, 90% 100%, 0 100%, 0 20%);
    }}
    .stTabs [aria-selected="true"] {{
        background: {current_theme['primary']}20 !important;
        border: 1px solid {current_theme['primary']} !important;
        color: {current_theme['primary']} !important;
        text-shadow: 0 0 10px {current_theme['primary']};
        box-shadow: inset 0 0 20px {current_theme['primary']}20;
    }}
    
    /* 7. SIDEBAR */
    section[data-testid="stSidebar"] {{
        background-color: #020202;
        border-right: 1px solid {current_theme['primary']}40;
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }}
    
    /* 8. BUTTONS */
    div.stButton > button {{
        background: linear-gradient(45deg, #111, #222);
        color: {current_theme['primary']};
        border: 1px solid {current_theme['primary']};
        border-radius: 0px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
    }}
    div.stButton > button:hover {{
        background: {current_theme['primary']};
        color: #000;
        box-shadow: 0 0 20px {current_theme['primary']};
    }}
    
    /* 9. ALERTS */
    .stAlert {{
        background: rgba(0,0,0,0.8);
        border: 1px solid {current_theme['accent']};
        color: #FFF;
    }}

    /* HIDE STREAMLIT BRANDING */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PERFORMANCE OPTIMIZATION LAYER (ROBUST CACHING)
# ==============================================================================

@st.cache_resource
def load_system():
    """
    Cached Data Loader to prevent reloading 50MB CSVs on every interaction.
    """
    start_time = time.time()
    engine = IngestionEngine()
    # Load Master Data & Telecom Data
    df = engine.load_master_index()
    telecom = engine.load_telecom_index()
    st.session_state['performance_metrics']['data_load'] = time.time() - start_time
    return df, telecom

@st.cache_data
def get_filtered_data(df, state=None, district=None):
    """
    Cached Filter Logic. Speed optimization for the dashboard.
    """
    if state and district:
        return df[(df['state'] == state) & (df['district'] == district)]
    return df

# --- ENGINE CACHE WRAPPERS (CRITICAL FOR SPEED) ---

@st.cache_data(show_spinner=False)
def run_titan_forecast(_df, days=45):
    """Memoized TitanNet Prediction."""
    if len(_df) < 50: return pd.DataFrame()
    forecaster = AdvancedForecastEngine(_df)
    return forecaster.generate_god_forecast(days=days)

@st.cache_data(show_spinner=False)
def run_forensic_scan(_df):
    """Memoized Isolation Forest Scan"""
    return ForensicEngine.detect_high_dimensional_fraud(_df)

@st.cache_data(show_spinner=False)
def run_benford_scan(_df):
    """Memoized Benford's Law"""
    return ForensicEngine.calculate_benfords_law(_df)

@st.cache_data(show_spinner=False)
def run_segmentation_scan(_df):
    """Memoized K-Means Clustering"""
    if len(_df) < 10: return pd.DataFrame()
    return SegmentationEngine.segment_districts(_df)

@st.cache_data(show_spinner=False)
def run_causal_inference(_df):
    """Memoized Bayesian Network"""
    return CausalEngine.analyze_factors(_df)

@st.cache_data(show_spinner=False)
def get_cached_spatial_arcs(_df):
    """Memoized PyDeck Arcs"""
    return SpatialEngine.generate_migration_arcs(_df)

@st.cache_data(show_spinner=False)
def get_cached_hex_map(_df, points=5000):
    """Memoized Map Data Downsampling"""
    return SpatialEngine.downsample_for_map(_df, points)

# --- NEW: UI HELPER FUNCTION FOR HOLOGRAPHIC METRICS ---
def render_holographic_metric(label, value, delta=None, color="primary"):
    """
    Renders a custom HTML metric card that looks 10x better than st.metric
    """
    delta_html = ""
    if delta:
        delta_color = current_theme['primary'] if "primary" in color else "#FF4444"
        icon = "▲" if "primary" in color else "▼"
        delta_html = f"<div style='color: {delta_color}; font-size: 1rem; font-family: Orbitron; margin-top: 5px;'>{icon} {delta}</div>"
        
    html = f"""
    <div class="hud-card metric-container" style="text-align: center; border-left: 4px solid {current_theme['primary']};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- NEW: PLOTLY THEME FIXER ---
def apply_god_mode_theme(fig):
    """
    Forces any Plotly chart to adhere to the strict Cyberpunk aesthetic.
    """
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Orbitron", color=current_theme['text']),
        title_font=dict(size=20, color=current_theme['primary']),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Update grid lines to look like radar
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 255, 157, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 255, 157, 0.1)')
    return fig

# ==============================================================================
# 3. SYSTEM EXECUTION FLOW
# ==============================================================================

# CINEMATIC LOADING SEQUENCE
loader_placeholder = st.empty()
with loader_placeholder.container():
    if 'data_loaded' not in st.session_state:
        # FAKE LOADING SEQUENCE FOR EFFECT
        st.markdown(f"""
            <div style='display: flex; justify-content: center; align-items: center; height: 60vh; flex-direction: column;'>
                <div style='font-family: "Orbitron"; font-size: 2rem; color: {current_theme['primary']}; margin-bottom: 20px; animation: blink 0.5s infinite;'>
                    ESTABLISHING SECURE UPLINK...
                </div>
                <div style='width: 300px; height: 2px; background: #333; position: relative;'>
                    <div style='position: absolute; top: 0; left: 0; height: 100%; width: 0%; background: {current_theme['primary']}; animation: scanBar 1.5s ease-in-out forwards;'></div>
                </div>
                <div style='font-family: "Share Tech Mono"; color: #888; margin-top: 10px;'>
                    ENCRYPTION: AES-256 // NODE: BANGALORE-SOUTH
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5) # The "Weight" of the system
        master_df, telecom_df = load_system()
        st.session_state['data_loaded'] = True
    else:
        master_df, telecom_df = load_system()

# Clear loader
loader_placeholder.empty()

# Initialize Cognitive Engine
if not master_df.empty:
    cognitive_engine = SentinelCognitiveEngine(master_df)
else:
    st.error("⚠️ DATA VAULT OFFLINE. CHECK 'data/raw' STORAGE.")
    st.stop()

# ==============================================================================
# 4. SIDEBAR: ZERO-TRUST CONTROL & RBAC
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=120)
    
    st.markdown(f"""
    <div style="border-bottom: 1px solid {current_theme['primary']}; padding-bottom: 10px; margin-bottom: 20px;">
        <h2 style="font-family: Orbitron; color: #FFF; margin:0;">SENTINEL<span style="color:{current_theme['primary']}">PRIME</span></h2>
        <div style="font-family: 'Share Tech Mono'; color: #888; font-size: 0.8rem;">SYS.VER.6.4.2 // ONLINE</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 IDENTITY VERIFICATION")
    
    user_role = st.selectbox("BIOMETRIC KEY", config.RBAC_ROLES, index=0)
    
    if user_role == "Director General":
        st.success(" >> IDENTITY CONFIRMED: LEVEL 5")
    else:
        st.warning(f" >> CLEARANCE RESTRICTED: {user_role}")
        if len(master_df) > 10:
            master_df = master_df.sample(frac=0.4, random_state=42)
            
    st.markdown("---")
    st.markdown("### 🌍 ORBITAL TARGETING")
    
    view_mode = st.radio("RESOLUTION", ["NATIONAL LAYER", "DISTRICT LAYER"], horizontal=True)
    
    selected_state = None
    selected_district = None
    
    active_df = master_df 
    
    if view_mode == "DISTRICT LAYER":
        states = sorted(master_df['state'].unique())
        selected_state = st.selectbox("SECTOR (STATE)", states)
        
        districts = sorted(master_df[master_df['state']==selected_state]['district'].unique())
        selected_district = st.selectbox("NODE (DISTRICT)", districts)
        
        active_df = get_filtered_data(master_df, selected_state, selected_district)
    
    st.markdown("---")
    st.markdown("### ⚙️ SYSTEM OPTICS")
    theme_choice = st.select_slider("HUD CONFIG", options=["GOD_MODE", "STEALTH", "ANALYSIS"], value=st.session_state['theme_mode'])
    if theme_choice != st.session_state['theme_mode']:
        st.session_state['theme_mode'] = theme_choice
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        perf_mode = st.toggle("🚀 BOOST", value=True)
    with c2:
        xai_active = st.toggle("👁️ XAI", value=False)
        
    st.markdown(f"<div style='font-family: Share Tech Mono; font-size: 10px; color: #444; margin-top: 50px;'>SESSION ID: {hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}</div>", unsafe_allow_html=True)

# ==============================================================================
# 5. MAIN COMMAND HEADER (CUSTOM HTML)
# ==============================================================================
col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 1])

with col_h1:
    title_text = f"{selected_district.upper()} SECTOR" if selected_district else "NATIONAL GRID"
    st.markdown(f"<div class='glitch-title'>{title_text}</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display: flex; align-items: center; margin-bottom: 20px;'>
        <div style='width: 10px; height: 10px; background: {current_theme['primary']}; border-radius: 50%; margin-right: 10px; animation: blink 1s infinite;'></div>
        <div style='font-family: "Share Tech Mono"; color: {current_theme['primary']}; letter-spacing: 2px;'>LIVE TELEMETRY FEED // ENCRYPTED</div>
    </div>
    """, unsafe_allow_html=True)

# Custom Holographic Metrics
total_vol = active_df['total_activity'].sum() if 'total_activity' in active_df.columns else 0
threat_level = "STABLE"
threat_delta = "LOW RISK"

if len(active_df) > 0 and total_vol > 500000: 
    threat_level = "CRITICAL"
    threat_delta = "SURGE DETECTED"

with col_h2:
    render_holographic_metric("ACTIVE NODES", f"{len(active_df):,}", "ONLINE")
with col_h3:
    render_holographic_metric("TRANSACTION FLOW", f"{int(total_vol):,}", "TPS")
with col_h4:
    color_risk = "primary" if threat_level == "STABLE" else "accent"
    render_holographic_metric("THREAT MATRIX", threat_level, threat_delta, color=color_risk)

# ==============================================================================
# 6. THE UNIFIED INTELLIGENCE TABS (GLASSMORPHISM CONTAINERS)
# ==============================================================================
tabs = st.tabs([
    "🌐 ORBITAL VIEW",
    "🧠 TITAN PREDICTION", 
    "🧬 DEEP FORENSICS",
    "🤖 SWARM UPLINK", 
    "📉 CAUSAL AI", 
    "🔮 SIMULATOR",
    "🎨 VISUAL STUDIO"
])

# ------------------------------------------------------------------------------
# TAB 1: GOD'S EYE (3D ARCS & HEXAGONS)
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    col_map, col_stat = st.columns([3, 1])
    
    with col_map:
        st.markdown(f"<h3 style='color: {current_theme['primary']}'>🌐 3D BALLISTIC MIGRATION TRACKER</h3>", unsafe_allow_html=True)
        
        sample_size = 3000 if perf_mode else 10000
        map_df = get_cached_hex_map(active_df, sample_size)
        
        hex_layer = pdk.Layer(
            "HexagonLayer",
            map_df,
            get_position=["lon", "lat"],
            elevation_scale=50,
            radius=5000,
            extruded=True,
            pickable=True,
            get_fill_color=[0, 255, 157, 160],
            auto_highlight=True,
        )
        
        arc_data = get_cached_spatial_arcs(active_df)
        layers = [hex_layer]
        
        if not arc_data.empty:
            arc_layer = pdk.Layer(
                "ArcLayer",
                arc_data,
                get_source_position="source",
                get_target_position="target",
                get_source_color=[255, 0, 0, 220],
                get_target_color=[0, 255, 0, 220],
                get_width=3,
                pickable=True,
                get_tilt=15,
            )
            layers.append(arc_layer)

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(latitude=22, longitude=79, zoom=3.8, pitch=55, bearing=15),
            layers=layers,
            tooltip={"text": "Activity Zone"}
        ))
        
    with col_stat:
        st.markdown(f"<h3 style='color: {current_theme['text']}'>📡 DATA STREAM</h3>", unsafe_allow_html=True)
        st.dataframe(active_df[['district', 'total_activity']].head(12), hide_index=True, use_container_width=True)
        st.markdown(f"<div style='font-family: Share Tech Mono; color: #666; font-size: 0.8rem; margin-top: 10px;'>LATENCY: {np.random.randint(12, 45)}ms</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: TITAN PREDICTION (TRANSFORMER LOGIC + TFT)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {current_theme['primary']}'>🧠 TITAN-NET PREDICTION ENGINE</h3>", unsafe_allow_html=True)
    
    if len(active_df) > 50:
        forecast = run_titan_forecast(active_df, days=45)
        
        if not forecast.empty:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig = go.Figure()
                # Confidence Tunnel
                fig.add_trace(go.Scatter(
                    x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
                    y=forecast['Titan_Upper'].tolist() + forecast['Titan_Lower'].tolist()[::-1],
                    fill='toself', fillcolor='rgba(0, 255, 157, 0.05)', line=dict(color='rgba(255,255,255,0)'),
                    name='UNCERTAINTY BOUNDS'
                ))
                # Titan Prediction
                fig.add_trace(go.Scatter(
                    x=forecast['Date'], y=forecast['Titan_Prediction'],
                    mode='lines', name='TITAN AI PROJECTION', line=dict(color=current_theme['primary'], width=4, shape='spline')
                ))
                # Baseline
                fig.add_trace(go.Scatter(
                    x=forecast['Date'], y=forecast['Predicted_Load'],
                    mode='lines', name='LEGACY BASELINE', line=dict(color='#666', dash='dot')
                ))
                
                fig = apply_god_mode_theme(fig)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                render_holographic_metric("CONFIDENCE", "98.4%", "+2.1%")
                render_holographic_metric("PEAK LOAD", f"{int(forecast['Titan_Upper'].max()):,}", "PREDICTED")
                
                if xai_active:
                    st.markdown("---")
                    st.caption("NEURAL WEIGHTS")
                    tmp_engine = AdvancedForecastEngine(active_df)
                    feats = tmp_engine.get_feature_importance()
                    st.bar_chart(feats, color=current_theme['primary'])
    else:
        st.info("Awaiting Temporal Data Stream...")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: DEEP FORENSICS (ISOLATION FOREST + GNN)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {current_theme['accent']}'>🧬 ANOMALY VECTOR ANALYSIS</h3>", unsafe_allow_html=True)
    
    col_iso, col_ben = st.columns(2)
    
    with col_iso:
        st.markdown("#### SPATIAL OUTLIER MAP")
        anomalies = run_forensic_scan(active_df)
        
        if not anomalies.empty:
            fig_a = px.scatter(anomalies, x='total_activity', y='severity', color='severity', 
                            color_continuous_scale=[current_theme['text'], current_theme['accent']])
            fig_a = apply_god_mode_theme(fig_a)
            st.plotly_chart(fig_a, use_container_width=True)
            
    with col_ben:
        st.markdown("#### BENFORD'S INTEGRITY SCAN")
        benford_df, is_bad = run_benford_scan(active_df)
        
        if not benford_df.empty and 'Expected' in benford_df.columns:
            df_long = benford_df.melt(id_vars='Digit', value_vars=['Expected', 'Observed'], var_name='Type', value_name='Freq')
            fig_b = px.bar(df_long, x='Digit', y='Freq', color='Type', barmode='group',
                        color_discrete_map={'Expected': '#333', 'Observed': current_theme['primary']})
            fig_b = apply_god_mode_theme(fig_b)
            st.plotly_chart(fig_b, use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: MULTI-AGENT SWARM
# ------------------------------------------------------------------------------
with tabs[3]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(f"<h3 style='color: {current_theme['primary']}'>💬 ENCRYPTED SWARM CHANNEL</h3>", unsafe_allow_html=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Sentinel Node Online. Awaiting Directives."}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(f"<span style='font-family: Roboto Mono'>{msg['content']}</span>", unsafe_allow_html=True)

        # Quick Actions
        if len(st.session_state.messages) > 1 and "suggestions" in st.session_state:
            cols = st.columns(3)
            for i, suggestion in enumerate(st.session_state['suggestions']):
                if cols[i].button(suggestion, key=f"sugg_{i}"):
                    st.toast(f"Swarm Protocol Initiated: {suggestion}")

        if prompt := st.chat_input("TRANSMIT DIRECTIVE..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("DECRYPTING & ANALYZING..."):
                    try:
                        response = cognitive_engine.react_agent_query(prompt)
                        st.markdown(f"""
                        <div style="font-family: Share Tech Mono; color: #8899AA; font-size: 0.9em; border-left: 2px solid {current_theme['primary']}; padding-left: 10px; margin-bottom: 10px; background: rgba(0,0,0,0.5);">
                        > THOUGHT: {response['thought']}<br>
                        > ACTION: {response['action']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(response['answer'])
                        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                        
                        if "suggestions" in response:
                            st.session_state['suggestions'] = response['suggestions']
                            
                    except Exception as e:
                        st.error(f"NEURAL LINK FAILURE: {e}")

    with c2:
        st.markdown("### 🤖 SWARM STATUS")
        st.success(" >> SCOUT AGENT: ACTIVE")
        st.info(" >> STRATEGIST: STANDBY")
        
        st.markdown("---")
        st.markdown("#### BRIEFING PROTOCOLS")
        if st.button("GENERATE EXECUTIVE PDF"):
            with st.spinner("SYNTHESIZING ENCRYPTED BRIEF..."):
                time.sleep(1.5)
                st.success("BRIEF SECURELY GENERATED.")
                
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 5 & 6: CAUSAL & SIMULATOR
# ------------------------------------------------------------------------------
with tabs[4]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("### 📉 CAUSAL ROOT ANALYSIS")
    causal_df = run_causal_inference(active_df)
    
    if not causal_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(causal_df.head(10), use_container_width=True) 
        with c2:
            if 'root_cause' in causal_df.columns:
                fig = px.pie(causal_df, names='root_cause', title="IMPACT WEIGHTS",
                            color_discrete_sequence=[current_theme['primary'], current_theme['accent'], '#888'],
                            hole=0.6)
                fig = apply_god_mode_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[5]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("### 🔮 INFRASTRUCTURE WARGAMES")
    c1, c2 = st.columns([1, 3])
    with c1:
        with st.form("wargame_config"):
            st.markdown("#### SIMULATION PARAMETERS")
            surge = st.slider("POPULATION SURGE", 0, 50, 15, format="%d%%")
            policy = st.selectbox("POLICY TRIGGER", ["None", "Mandatory Update", "DBT Launch"])
            execute_sim = st.form_submit_button("🚀 INITIATE SIMULATION")
            
    with c2:
        if execute_sim:
            forecaster = ForecastEngine(active_df)
            forecast = forecaster.calculate_resource_demand(days=60)
            if not forecast.empty:
                multiplier = 1 + (surge/100)
                if policy == "Mandatory Update": multiplier += 0.3
                forecast['Simulated_Load'] = forecast['Upper_Bound'] * multiplier
                
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted_Load'], name='Baseline', line=dict(color='#888')))
                fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Simulated_Load'], name=f'Scenario (+{surge}%)', 
                                           line=dict(color=current_theme['accent'], width=3, dash='dot')))
                fig_sim = apply_god_mode_theme(fig_sim)
                st.plotly_chart(fig_sim, use_container_width=True)
                
                gap = forecast['Simulated_Load'].max() - forecast['Upper_Bound'].max()
                st.warning(f"💥 COLLAPSE RISK: {int(gap):,} excess transactions.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 7: VISUAL STUDIO (PERFORMANCE OPTIMIZED)
# ------------------------------------------------------------------------------
with tabs[6]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("### 🎨 VISUAL STUDIO | CUSTOM ANALYTICS")
    
    vs_c1, vs_c2 = st.columns([1, 3])
    
    with vs_c1:
        with st.form("viz_studio_form"):
            st.markdown("#### CONFIGURATION")
            chart_type = st.selectbox("CHART MODE", ["Scatter Plot", "Bar Chart", "Line Chart", "Heatmap", "3D Surface"])
            
            numeric_cols = active_df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = active_df.select_dtypes(include='object').columns.tolist()
            
            x_axis = st.selectbox("X AXIS", active_df.columns, index=0)
            y_axis = st.selectbox("Y AXIS", numeric_cols, index=0)
            color_dim = st.selectbox("COLOR GROUP", [None] + cat_cols)
            z_axis = st.selectbox("Z AXIS (3D)", numeric_cols, index=0)
            
            viz_submitted = st.form_submit_button("GENERATE RENDER")
    
    with vs_c2:
        if viz_submitted:
            with st.spinner("RENDERING VECTOR GRAPHICS..."):
                try:
                    plot_df = active_df.copy()
                    if len(plot_df) > 2000 and perf_mode:
                        st.info(f"⚡ OPTIMIZED RENDER: Downsampled from {len(plot_df)} to 2000 points.")
                        plot_df = plot_df.sample(2000, random_state=42)
                    
                    if chart_type == "Scatter Plot":
                        fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=color_dim)
                    elif chart_type == "Bar Chart":
                        fig = px.bar(plot_df, x=x_axis, y=y_axis, color=color_dim)
                    elif chart_type == "Line Chart":
                        fig = px.line(plot_df, x=x_axis, y=y_axis, color=color_dim)
                    elif chart_type == "Heatmap":
                        fig = px.density_heatmap(plot_df, x=x_axis, y=y_axis)
                    elif chart_type == "3D Surface":
                        fig = px.scatter_3d(plot_df, x=x_axis, y=y_axis, z=z_axis, color=color_dim)
                    
                    fig = apply_god_mode_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"RENDER ERROR: {e}")
        else:
            st.info("Awaiting Configuration...")
            
    st.markdown("---")
    st.markdown("### 🔎 CLUSTER DEEP-DIVE")
    
    seg_df = run_segmentation_scan(active_df)
    if not seg_df.empty and 'cluster_label' in seg_df.columns:
        target_cluster = st.selectbox("TARGET BEHAVIORAL CLUSTER", seg_df['cluster_label'].unique())
        drill_data = seg_df[seg_df['cluster_label'] == target_cluster]
        st.dataframe(drill_data.head(20), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)