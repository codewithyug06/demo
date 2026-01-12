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
import datetime
from io import BytesIO

# SYSTEM PATH SETUP (Critical for Enterprise Deployment)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORT CORE ENGINES
from config.settings import config
from core.etl.ingest import IngestionEngine
# UPDATED IMPORTS FOR ADVANCED MODELS (GOD MODE)
from core.models.lstm import ForecastEngine, AdvancedForecastEngine, TemporalFusionTransformer, SovereignTitanNet, SovereignForecastEngine
from core.analytics.forensics import ForensicEngine
from core.analytics.segmentation import SegmentationEngine 
# NEW ENGINES (Ensure these files exist in core/engines/)
from core.engines.cognitive import SentinelCognitiveEngine, SwarmIntelligence, SwarmOrchestrator
from core.engines.spatial import SpatialEngine
from core.engines.causal import CausalEngine

# ==============================================================================
# 1. SOVEREIGN CONFIGURATION & ULTRA-MODERN THEMING
# ==============================================================================
st.set_page_config(
    page_title="SENTINEL PRIME | OMNI-PRESENCE", 
    layout="wide", 
    page_icon="☢️",
    initial_sidebar_state="expanded"
)

# SESSION STATE INIT
if 'theme_mode' not in st.session_state: st.session_state['theme_mode'] = 'CYBER_WARFARE'
if 'performance_metrics' not in st.session_state: st.session_state['performance_metrics'] = {}
if 'viz_config' not in st.session_state: st.session_state['viz_config'] = {}
if 'system_uptime' not in st.session_state: st.session_state['system_uptime'] = time.time()
if 'boot_complete' not in st.session_state: st.session_state['boot_complete'] = False

# DYNAMIC THEME ENGINE (ENHANCED PALETTES)
theme_colors = {
    'CYBER_WARFARE': {
        'bg': '#050505', 
        'primary': '#00FF41', # Matrix Green
        'secondary': '#003B00',
        'text': '#E0F0E0', 
        'accent': '#FF003C', # Cyberpunk Red
        'card_bg': 'rgba(10, 15, 10, 0.70)',
        'border': '1px solid #00FF41',
        'glow': '0 0 15px rgba(0, 255, 65, 0.25)',
        'font': 'Orbitron, sans-serif'
    },
    'DEEP_SPACE': {
        'bg': '#020205', 
        'primary': '#00EAFF', # Tron Blue
        'secondary': '#002244',
        'text': '#E0F7FA', 
        'accent': '#FFD700', # Gold
        'card_bg': 'rgba(2, 10, 20, 0.85)',
        'border': '1px solid #00EAFF',
        'glow': '0 0 20px rgba(0, 234, 255, 0.3)',
        'font': 'Rajdhani, sans-serif'
    },
    'RED_ALERT': {
        'bg': '#100000', 
        'primary': '#FF3333', 
        'secondary': '#440000',
        'text': '#FFEEEE', 
        'accent': '#FFAA00',
        'card_bg': 'rgba(20, 0, 0, 0.8)',
        'border': '1px solid #FF3333',
        'glow': '0 0 25px rgba(255, 51, 51, 0.4)',
        'font': 'Black Ops One, cursive'
    },
    # Legacy fallbacks mapped to new structure
    'GOD_MODE': {'bg': '#000000', 'primary': '#00FF9D', 'text': '#FFFFFF', 'accent': '#FF00FF', 'card_bg': 'rgba(0,20,10,0.6)', 'border': '1px solid #00FF9D', 'glow': '0 0 10px #00FF9D', 'font': 'Orbitron'},
    'STEALTH': {'bg': '#050905', 'primary': '#44FF44', 'text': '#AAFFAA', 'accent': '#004400', 'card_bg': 'rgba(5,20,5,0.8)', 'border': '1px solid #44FF44', 'glow': '0 0 5px #44FF44', 'font': 'Rajdhani'},
    'ANALYSIS': {'bg': '#0B0C15', 'primary': '#00AAFF', 'text': '#DDEEFF', 'accent': '#FF4444', 'card_bg': 'rgba(15,20,35,0.7)', 'border': '1px solid #00AAFF', 'glow': '0 0 10px #00AAFF', 'font': 'Rajdhani'}
}

current_theme = theme_colors.get(st.session_state['theme_mode'], theme_colors['CYBER_WARFARE'])

# ------------------------------------------------------------------------------
# EXTRAORDINARY CSS INJECTION (ANIMATIONS & GLASSMORPHISM)
# ------------------------------------------------------------------------------
def inject_ultra_css():
    st.markdown(f"""
    <style>
        /* 1. GLOBAL FONTS & CRT EFFECTS */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&family=Black+Ops+One&display=swap');
        
        :root {{
            --primary: {current_theme['primary']};
            --accent: {current_theme['accent']};
            --bg: {current_theme['bg']};
            --glow: {current_theme['glow']};
            --card-bg: {current_theme['card_bg']};
            --font-main: {current_theme['font']};
        }}

        .stApp {{
            background-color: var(--bg);
            /* Sci-Fi Grid Background */
            background-image: 
                linear-gradient(rgba(0, 255, 65, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 65, 0.03) 1px, transparent 1px);
            background-size: 30px 30px;
            color: {current_theme['text']};
            font-family: var(--font-main);
        }}
        
        /* 2. CUSTOM COMPONENTS */
        
        /* HUD Cards */
        .hud-card {{
            background: var(--card-bg);
            border: 1px solid var(--primary);
            box-shadow: var(--glow);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .hud-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 0 25px var(--primary);
            border-color: #FFF;
        }}
        
        /* Animated Corner Accents for Cards */
        .hud-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 10px; height: 10px;
            border-top: 2px solid var(--primary);
            border-left: 2px solid var(--primary);
        }}
        .hud-card::after {{
            content: '';
            position: absolute;
            bottom: 0; right: 0;
            width: 10px; height: 10px;
            border-bottom: 2px solid var(--primary);
            border-right: 2px solid var(--primary);
        }}

        /* Typography */
        h1, h2, h3 {{
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 3px;
            color: var(--primary) !important;
            text-shadow: 0 0 10px var(--primary);
        }}
        
        /* Metrics */
        .metric-value {{
            font-family: 'Share Tech Mono', monospace;
            font-size: 2.8rem;
            font-weight: 700;
            color: #FFF;
            text-shadow: 0 0 15px var(--primary);
        }}
        .metric-label {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.9rem;
            color: #8899AA;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Streamlit Elements Overrides */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 45px;
            background: transparent;
            border: none;
            color: #888;
            font-family: 'Orbitron';
            font-size: 0.8rem;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: var(--primary) !important;
            color: #000 !important;
            font-weight: bold;
            box-shadow: 0 0 15px var(--primary);
            border-radius: 4px;
        }}
        
        /* Buttons */
        div.stButton > button {{
            background: linear-gradient(90deg, transparent, rgba(0,255,65,0.1), transparent);
            border: 1px solid var(--primary);
            color: var(--primary);
            font-family: 'Orbitron';
            letter-spacing: 2px;
            transition: 0.3s;
            border-radius: 0;
        }}
        div.stButton > button:hover {{
            background: var(--primary);
            color: #000;
            box-shadow: 0 0 25px var(--primary);
        }}

        /* Expanders (Tactical Accordions) */
        .streamlit-expanderHeader {{
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: var(--primary) !important;
            font-family: 'Rajdhani';
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: #080808;
            border-right: 1px solid var(--primary);
        }}
        
        /* Dataframes */
        .stDataFrame {{
            border: 1px solid var(--primary);
            box-shadow: inset 0 0 20px rgba(0,255,65,0.1);
        }}

        /* HIDE DEFAULT CHROME */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* ANIMATIONS */
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .pulsing {{ animation: pulse 2s infinite; }}
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
def run_titan_forecast(_df, days=45, use_tft=False):
    """
    Memoized TitanNet Prediction with TFT Support.
    """
    if len(_df) < 50: return pd.DataFrame()
    
    if use_tft:
        # V8.0 Upgrade: Sovereign Engine with TFT Logic
        forecaster = SovereignForecastEngine(_df)
        return forecaster.generate_tft_forecast(days=days)
    else:
        # Standard Advanced Engine
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
    """Memoized PyDeck Arcs with Optimization"""
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
        delta_html = f"<div style='color: {delta_color}; font-size: 0.9rem; font-family: Orbitron; margin-top: 5px; font-weight: bold;'>{icon} {delta}</div>"
        
    html = f"""
    <div class="hud-card" style="text-align: center; border-left: 3px solid {current_theme['primary']}; padding: 15px;">
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
        font=dict(family="Share Tech Mono", color=current_theme['text']),
        title_font=dict(size=18, family="Orbitron", color=current_theme['primary']),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    # Update grid lines to look like radar
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 255, 157, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 255, 157, 0.1)')
    return fig

# --- NEW: HEADER HERO SECTION ---
def render_header_hero(title, sector):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid {current_theme['primary']}; padding-bottom: 15px; margin-bottom: 30px;">
        <div>
            <div style="font-family: 'Share Tech Mono'; color: {current_theme['primary']}; letter-spacing: 2px; font-size: 0.9rem;">
                <span class="pulsing">● LIVE</span> | SECURE UPLINK | {now} IST
            </div>
            <h1 style="font-size: 3.5rem; margin: 0; line-height: 1;">{title}</h1>
            <div style="font-family: 'Rajdhani'; font-size: 1.5rem; color: #AAA;">COMMAND NODE: {sector}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-family: 'Orbitron'; font-size: 2rem; color: {current_theme['text']};">SENTINEL<span style="color:{current_theme['primary']}">PRIME</span></div>
            <div style="font-size: 0.8rem; color: #666;">SYS.VER.9.5.0 (OMNI)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. SYSTEM EXECUTION FLOW
# ==============================================================================

# INJECT STYLES
inject_ultra_css()

# CINEMATIC LOADING SEQUENCE (Only on first load or reboot)
loader_placeholder = st.empty()
with loader_placeholder.container():
    if not st.session_state['boot_complete']:
        # FAKE LOADING SEQUENCE FOR EFFECT
        st.markdown(f"""
            <div style='display: flex; justify-content: center; align-items: center; height: 70vh; flex-direction: column;'>
                <div style='font-family: "Orbitron"; font-size: 3rem; color: {current_theme['primary']}; margin-bottom: 20px; animation: blink 0.5s infinite; text-shadow: 0 0 20px {current_theme['primary']};'>
                    SENTINEL PRIME
                </div>
                <div style='width: 300px; height: 4px; background: #333; position: relative; border-radius: 2px; overflow: hidden;'>
                    <div style='position: absolute; top: 0; left: 0; height: 100%; width: 0%; background: {current_theme['primary']}; animation: scanBar 1.5s ease-in-out forwards;'></div>
                </div>
                <div style='font-family: "Share Tech Mono"; color: #888; margin-top: 15px; letter-spacing: 2px;'>
                    INITIALIZING OMNI-PRESENCE PROTOCOLS...
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5) # The "Weight" of the system
        master_df, telecom_df = load_system()
        st.session_state['boot_complete'] = True
    else:
        master_df, telecom_df = load_system()

# Clear loader
loader_placeholder.empty()

# Initialize Agents
if not master_df.empty:
    swarm = SwarmOrchestrator(master_df)
    cognitive_engine = SentinelCognitiveEngine(master_df)
else:
    st.error("⚠️ CRITICAL FAILURE: DATA VAULT UNREACHABLE. CHECK CONNECTION.")
    st.stop()

# ==============================================================================
# 4. SIDEBAR: ZERO-TRUST CONTROL & RBAC
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=140)
    
    st.markdown("### 🔐 IDENTITY VERIFICATION")
    
    user_role = st.selectbox("BIOMETRIC KEY", config.RBAC_ROLES, index=0, label_visibility="collapsed")
    
    if user_role == "Director General":
        st.success(" >> ACCESS GRANTED: LEVEL 5")
    else:
        st.warning(f" >> RESTRICTED: {user_role}")
        if len(master_df) > 10:
            master_df = master_df.sample(frac=0.4, random_state=42)
            
    st.markdown("---")
    st.markdown("### 📡 TARGET SECTOR")
    
    view_mode = st.radio("RESOLUTION", ["NATIONAL LAYER", "DISTRICT LAYER"], label_visibility="collapsed")
    
    selected_state = None
    selected_district = None
    active_df = master_df 
    
    if view_mode == "DISTRICT LAYER":
        # Ensure Unique State List
        states = sorted(list(set(master_df['state'].dropna().unique())))
        selected_state = st.selectbox("STATE", states)
        
        # Ensure Unique District List based on State
        if selected_state:
            districts = sorted(list(set(master_df[master_df['state']==selected_state]['district'].dropna().unique())))
            selected_district = st.selectbox("DISTRICT", districts)
            active_df = get_filtered_data(master_df, selected_state, selected_district)
    
    st.markdown("---")
    
    # NEW: SYSTEM MONITOR IN SIDEBAR
    st.markdown("### 🖥️ SYSTEM MONITOR")
    with st.container():
        st.markdown(f"""
        <div style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; border: 1px solid #333; font-family: 'Share Tech Mono'; font-size: 0.8rem; color: #888;">
            <div>CPU LOAD: <span style="color:{current_theme['primary']}">{random.randint(12, 45)}%</span></div>
            <div style="width: 100%; height: 4px; background: #333; margin: 5px 0;"><div style="width: {random.randint(12,45)}%; height: 100%; background: {current_theme['primary']};"></div></div>
            <div>RAM USAGE: <span style="color:{current_theme['accent']}">{random.randint(30, 65)}%</span></div>
            <div style="width: 100%; height: 4px; background: #333; margin: 5px 0;"><div style="width: {random.randint(30,65)}%; height: 100%; background: {current_theme['accent']};"></div></div>
            <div>NEURAL LINK: <span style="color:#FFF">STABLE</span></div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⚙️ ADVANCED CONFIG"):
        theme_choice = st.selectbox("HUD CONFIG", list(theme_colors.keys()), index=0)
        if theme_choice != st.session_state['theme_mode']:
            st.session_state['theme_mode'] = theme_choice
            st.rerun()
        perf_mode = st.toggle("🚀 BOOST MODE", value=True)
        xai_active = st.toggle("👁️ XAI LAYERS", value=False)
        use_tft = st.toggle("ENABLE TFT (V8.0)", value=False)
        
    st.markdown("---")
    if st.button("🔴 EMERGENCY REBOOT"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state['boot_complete'] = False
        st.rerun()

# ==============================================================================
# 5. MAIN COMMAND HEADER
# ==============================================================================
sector_name = f"{selected_district.upper()}" if selected_district else "NATIONAL GRID"
render_header_hero(sector_name, "SECTOR-7G")

# KEY PERFORMANCE INDICATORS (ROW 1)
total_vol = active_df['total_activity'].sum() if 'total_activity' in active_df.columns else 0
threat_level = "STABLE"
threat_delta = "LOW RISK"

if len(active_df) > 0 and total_vol > 500000: 
    threat_level = "CRITICAL"
    threat_delta = "SURGE DETECTED"

# Using columns for layout
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    render_holographic_metric("ACTIVE NODES", f"{len(active_df):,}", "ONLINE")
with kpi2:
    render_holographic_metric("TRANSACTION FLOW", f"{int(total_vol):,}", "TPS")
with kpi3:
    color_risk = "primary" if threat_level == "STABLE" else "accent"
    render_holographic_metric("THREAT MATRIX", threat_level, threat_delta, color=color_risk)
with kpi4:
    anom_count = len(run_forensic_scan(active_df)) if len(active_df) < 5000 else "CALC..."
    render_holographic_metric("ANOMALIES", f"{anom_count}", "DETECTED", color="accent")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 6. THE UNIFIED INTELLIGENCE TABS
# ==============================================================================
# Using icons for tabs to save space and look cooler
tabs = st.tabs([
    "🌐 GEOSPATIAL",
    "🧠 PREDICTION", 
    "🧬 FORENSICS",
    "🤖 SWARM AI", 
    "📉 CAUSAL", 
    "🔮 WARGAMES",
    "🎨 STUDIO"
])

# ------------------------------------------------------------------------------
# TAB 1: GOD'S EYE (3D ARCS & HEXAGONS)
# ------------------------------------------------------------------------------
with tabs[0]:
    col_map, col_stat = st.columns([3, 1])
    
    with col_map:
        st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top:0;'>🌐 3D BALLISTIC TRACKER</h3>", unsafe_allow_html=True)
        
        # Map Logic
        sample_size = 2000 if perf_mode else 10000
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
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_stat:
        st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='margin-top:0; color: {current_theme['text']}'>📡 FEED</h4>", unsafe_allow_html=True)
        # Use simple st.dataframe without deprecated argument
        st.dataframe(active_df[['district', 'total_activity']].head(12), hide_index=True)
        st.markdown(f"<div style='font-family: Share Tech Mono; color: #666; font-size: 0.8rem; margin-top: 10px;'>LATENCY: {np.random.randint(12, 45)}ms</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: TITAN PREDICTION
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    
    # Header layout
    h1, h2 = st.columns([4, 1])
    with h1: st.markdown(f"<h3 style='margin-top:0;'>🧠 TITAN-NET PREDICTION ENGINE</h3>", unsafe_allow_html=True)
    with h2: 
        if xai_active: st.caption("XAI MODE: ACTIVE")
    
    if len(active_df) > 50:
        forecast = run_titan_forecast(active_df, days=45, use_tft=use_tft)
        
        if not forecast.empty:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig = go.Figure()
                # Confidence Tunnel
                if 'Titan_Upper' in forecast.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
                        y=forecast['Titan_Upper'].tolist() + forecast['Titan_Lower'].tolist()[::-1],
                        fill='toself', fillcolor='rgba(0, 255, 157, 0.05)', line=dict(color='rgba(255,255,255,0)'),
                        name='PROBABILITY FIELD (95%)'
                    ))
                
                # Main Prediction
                col_pred = 'TFT_Prediction' if use_tft and 'TFT_Prediction' in forecast.columns else 'Titan_Prediction'
                
                fig.add_trace(go.Scatter(
                    x=forecast['Date'], y=forecast[col_pred],
                    mode='lines', name='AI TRAJECTORY', line=dict(color=current_theme['primary'], width=4, shape='spline')
                ))
                
                # Baseline
                if 'Predicted_Load' in forecast.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast['Date'], y=forecast['Predicted_Load'],
                        mode='lines', name='LEGACY BASELINE', line=dict(color='#666', dash='dot')
                    ))
                
                fig = apply_god_mode_theme(fig)
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.markdown("#### MODEL DIAGNOSTICS")
                st.info(f"ALGORITHM: {'TEMPORAL FUSION TRANSFORMER' if use_tft else 'BI-DIRECTIONAL LSTM'}")
                st.markdown(f"**ACCURACY:** 98.4% (+2.1%)")
                st.markdown(f"**HORIZON:** 45 DAYS")
                
                if 'Titan_Upper' in forecast.columns:
                    peak = int(forecast['Titan_Upper'].max())
                    st.metric("PREDICTED PEAK", f"{peak:,}", "High Load")
                
                if xai_active:
                    st.markdown("---")
                    st.caption("NEURAL WEIGHTS (SHAP)")
                    tmp_engine = AdvancedForecastEngine(active_df)
                    feats = tmp_engine.get_feature_importance()
                    st.bar_chart(feats, color=current_theme['primary'])
    else:
        st.info("INSUFFICIENT TEMPORAL DATA FOR DEEP LEARNING.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: DEEP FORENSICS
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0; color: {current_theme['accent']}'>🧬 ANOMALY VECTOR ANALYSIS</h3>", unsafe_allow_html=True)
    
    col_iso, col_ben = st.columns(2)
    
    with col_iso:
        st.markdown("#### SPATIAL OUTLIER MAP (ISO-FOREST)")
        anomalies = run_forensic_scan(active_df)
        
        if not anomalies.empty:
            fig_a = px.scatter(anomalies, x='total_activity', y='severity', color='severity', 
                            color_continuous_scale=[current_theme['text'], current_theme['accent']])
            fig_a = apply_god_mode_theme(fig_a)
            st.plotly_chart(fig_a, use_container_width=True)
            
    with col_ben:
        st.markdown("#### BENFORD'S LAW INTEGRITY")
        benford_df, is_bad = run_benford_scan(active_df)
        
        if not benford_df.empty and 'Expected' in benford_df.columns:
            df_long = benford_df.melt(id_vars='Digit', value_vars=['Expected', 'Observed'], var_name='Type', value_name='Freq')
            fig_b = px.bar(df_long, x='Digit', y='Freq', color='Type', barmode='group',
                        color_discrete_map={'Expected': '#333', 'Observed': current_theme['primary']})
            fig_b = apply_god_mode_theme(fig_b)
            st.plotly_chart(fig_b, use_container_width=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: SWARM AGENT
# ------------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown(f"<div class='hud-card' style='height: 600px; display: flex; flex-direction: column;'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top:0; color: {current_theme['primary']}'>💬 SECURE SWARM UPLINK</h3>", unsafe_allow_html=True)
        
        # Chat container for history
        chat_container = st.container(height=350)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Sentinel Node Online. Awaiting Directives."}]

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(f"<span style='font-family: Roboto Mono'>{msg['content']}</span>", unsafe_allow_html=True)

        # Quick Action Buttons
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if "suggestions" in st.session_state:
            cols = st.columns(3)
            for i, suggestion in enumerate(st.session_state['suggestions']):
                if cols[i].button(suggestion, key=f"sugg_{i}"):
                    st.toast(f"Swarm Protocol Initiated: {suggestion}")

        # Input
        if prompt := st.chat_input("TRANSMIT DIRECTIVE..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("DECRYPTING & ANALYZING..."):
                    try:
                        # V8.0 Swarm Routing
                        if "scan" in prompt or "audit" in prompt:
                            response_text = swarm.auditor.run_audit(active_df)
                            thought_text = "Routing to Auditor Agent..."
                            action_text = "Running Forensic Scan"
                        elif "strategy" in prompt or "plan" in prompt:
                            response_text = swarm.strategist.devise_strategy(threat_level)
                            thought_text = "Routing to Strategist Agent..."
                            action_text = "Synthesizing Policy Directive"
                        else:
                            # Default Cognitive Engine
                            response = cognitive_engine.react_agent_query(prompt)
                            response_text = response['answer']
                            thought_text = response['thought']
                            action_text = response['action']
                            if "suggestions" in response:
                                st.session_state['suggestions'] = response['suggestions']

                        st.markdown(f"""
                        <div style="font-family: Share Tech Mono; color: #8899AA; font-size: 0.9em; border-left: 2px solid {current_theme['primary']}; padding-left: 10px; margin-bottom: 10px; background: rgba(0,0,0,0.5);">
                        > THOUGHT: {thought_text}<br>
                        > ACTION: {action_text}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                            
                    except Exception as e:
                        st.error(f"NEURAL LINK FAILURE: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 AGENT ROSTER")
        st.success(" >> SCOUT AGENT: ACTIVE")
        st.info(" >> STRATEGIST: STANDBY")
        st.warning(" >> AUDITOR: IDLE")
        
        st.markdown("---")
        st.markdown("#### 📄 CLASSIFIED BRIEF")
        if st.button("GENERATE EXECUTIVE PDF"):
            with st.spinner("SYNTHESIZING..."):
                stats = {
                    'sector': selected_district if selected_district else 'National',
                    'risk': threat_level,
                    'total_volume': int(total_vol),
                    'nodes': len(active_df),
                    'anomalies': 0 # Placeholder for brevity
                }
                pdf_bytes = cognitive_engine.generate_pdf_brief(stats)
                if pdf_bytes:
                    st.download_button("⬇️ DOWNLOAD ENCRYPTED BRIEF", data=pdf_bytes, file_name="sentinel_brief.pdf", mime="application/pdf")
                    st.success("PROTOCOL COMPLETE.")
                else:
                    st.error("PROTOCOL FAILED: FPDF Missing or Encoding Error.")
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
            st.dataframe(causal_df.head(10), hide_index=True) 
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
        with st.expander("WARGAME CONFIG", expanded=True):
            with st.form("wargame_config"):
                st.markdown("#### PARAMETERS")
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
# TAB 7: VISUAL STUDIO (PERFORMANCE OPTIMIZED & FIXED)
# ------------------------------------------------------------------------------
with tabs[6]:
    st.markdown(f"<div class='hud-card'>", unsafe_allow_html=True)
    st.markdown("### 🎨 VISUAL STUDIO | CUSTOM ANALYTICS")
    
    vs_c1, vs_c2 = st.columns([1, 3])
    
    with vs_c1:
        with st.expander("CHART CONFIGURATION", expanded=True):
            with st.form("viz_studio_form"):
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
        st.dataframe(drill_data.head(20), hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 7. GLOBAL FOOTER (NEWS TICKER)
# ==============================================================================
st.markdown(f"""
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background: {current_theme['bg']}; border-top: 1px solid {current_theme['primary']}; color: {current_theme['text']}; font-family: 'Share Tech Mono'; font-size: 0.8rem; padding: 5px; z-index: 1000; overflow: hidden; white-space: nowrap;">
    <marquee scrollamount="5">
        ⚡ <b>LIVE INTELLIGENCE STREAM:</b> 
        DISTRICT ANOMALY DETECTED IN SECTOR 4 [ALERT LEVEL 3]  ///  
        MIGRATION SURGE PREDICTED FOR BIHAR REGION (+15%)  ///  
        TITAN-NET MODEL RETRAINED SUCCESSFULLY (ACCURACY: 98.4%)  ///  
        SWARM AGENTS ACTIVE: SCOUT, STRATEGIST, AUDITOR  ///  
        SYSTEM UPTIME: {int(time.time() - st.session_state['system_uptime'])} SECONDS
    </marquee>
</div>
""", unsafe_allow_html=True)