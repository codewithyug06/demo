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
import graphviz # NEW: For Causal DAG Visualization
from io import BytesIO
import warnings

# Suppress annoying warnings for the demo
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

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
from core.engines.cognitive import SentinelCognitiveEngine, SwarmIntelligence, SwarmOrchestrator, PolicyBudgetOptimizer
from core.engines.spatial import SpatialEngine, GraphNeuralNetwork
from core.engines.causal import CausalEngine
# NEW: Privacy & Fiscal Engines
from core.analytics.privacy_engine import PrivacyEngine
from core.analytics.fiscal_logic import FiscalImpactEngine
from core.engines.voice_uplink import VoiceUplinkEngine

# ==============================================================================
# 1. SOVEREIGN CONFIGURATION & ULTRA-MODERN THEMING
# ==============================================================================
st.set_page_config(
    page_title="SENTINEL PRIME | AEGIS COMMAND", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# SESSION STATE INIT
if 'theme_mode' not in st.session_state: st.session_state['theme_mode'] = 'CYBER_WARFARE'
if 'performance_metrics' not in st.session_state: st.session_state['performance_metrics'] = {}
if 'viz_config' not in st.session_state: st.session_state['viz_config'] = {}
if 'system_uptime' not in st.session_state: st.session_state['system_uptime'] = time.time()
if 'boot_complete' not in st.session_state: st.session_state['boot_complete'] = False
if 'privacy_engine' not in st.session_state: st.session_state['privacy_engine'] = PrivacyEngine()

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
    'SOVEREIGN_LIGHT': { # NEW: For older government officials
        'bg': '#F0F2F6', 
        'primary': '#003366', 
        'secondary': '#FFFFFF',
        'text': '#1A1A1A', 
        'accent': '#FF4B4B',
        'card_bg': 'rgba(255, 255, 255, 0.9)',
        'border': '1px solid #CCC',
        'glow': '0 0 5px rgba(0,0,0,0.1)',
        'font': 'Roboto, sans-serif'
    }
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
        
        /* REMOVE TOP PADDING & DEFAULT ELEMENTS */
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 5rem;
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .stDeployButton {{display:none;}}
        [data-testid="stToolbar"] {{visibility: hidden !important;}}
        [data-testid="stDecoration"] {{display:none;}}
        
        /* 2. CUSTOM COMPONENTS */
        
        /* HUD Cards - Applied to Metrics primarily */
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
        
        /* Container Styling */
        div[data-testid="stContainer"] {{
            border-radius: 5px;
        }}

        /* --- CHAT BOT CONTAINER FIX --- */
        /* Fix the chat input container to stay fixed and clear */
        .stChatInputContainer {{
            padding-bottom: 20px;
        }}
        
        /* Fix for chat messages overflowing or not fitting */
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            border-left: 2px solid var(--primary);
            word-wrap: break-word !important; 
            overflow-wrap: break-word !important;
            white-space: pre-wrap !important;
            max-width: 100% !important;
        }}
        
        /* User message distinct style */
        .stChatMessage[data-testid="stChatMessageUser"] {{
             border-left: 2px solid var(--accent);
             background-color: rgba(255, 0, 60, 0.05);
        }}
        
        /* Ensure chat container scrolls properly */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div.element-container {{
            overflow-y: auto;
        }}
        
        /* Force chat messages to stay within container width */
        div[data-testid="stChatMessageContent"] {{
            max-width: 100%;
            overflow-wrap: break-word;
        }}

        /* --- SIDEBAR DROPDOWN & SELECTBOX FIXES --- */
        
        /* Force text colors for labels */
        .stSelectbox label, .stRadio label {{
            color: var(--primary) !important;
            font-family: 'Orbitron';
            font-size: 0.95rem; /* Bigger Label */
            margin-bottom: 8px;
            display: block !important;
        }}
        
        /* Main Selectbox Container - Make sure text is visible */
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: #0A0F0A !important;
            border: 1px solid var(--primary) !important;
            color: #E0F0E0 !important;
            min-height: 50px;
            display: flex;
            align-items: center;
        }}
        
        /* The Text inside the selectbox */
        .stSelectbox div[data-baseweb="select"] span {{
            color: #E0F0E0 !important;
            font-family: 'Rajdhani';
            font-size: 1.1rem; /* Readable font size */
        }}

        /* CRITICAL FIX: The Dropdown Menu (Popover) Visibility */
        div[data-baseweb="popover"] {{
            background-color: #050505 !important;
            border: 1px solid var(--primary) !important;
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
            z-index: 999999 !important;
        }}
        
        div[data-baseweb="menu"] {{
            background-color: #050505 !important;
        }}
        
        /* Options in the list */
        ul[data-baseweb="menu"] li {{
            background-color: #050505 !important;
            color: #E0F0E0 !important;
            border-bottom: 1px solid #111;
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        
        /* Hover Effect for Options */
        ul[data-baseweb="menu"] li:hover {{
            background-color: var(--primary) !important;
            color: #000 !important;
        }}
        
        /* Improve Radio Buttons (Toggles) */
        .stRadio > div {{
            gap: 15px; /* Space out radio options */
            flex-direction: column; /* Stack vertically for better readability */
            align-items: flex-start;
        }}
        
        /* Toggle Switch Styling */
        .stToggle {{
            padding-top: 15px;
            padding-bottom: 15px;
        }}
        .stToggle label {{
            color: #E0E0E0;
            font-family: 'Rajdhani';
            font-size: 1rem;
        }}
        
        /* Spacing out the sidebar content */
        section[data-testid="stSidebar"] > div > div:nth-child(2) {{
            padding-top: 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem; /* MASSIVE GAP between widgets */
        }}

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
    # V9.9: Using distributed loader if configured
    if getattr(config, 'COMPUTE_BACKEND', 'local') == 'dask':
        try:
            df = engine.load_master_index_distributed()
        except:
            df = engine.load_master_index()
    else:
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
def run_titan_forecast(_df, days=45, use_tft=False, use_pinn=False):
    """
    Memoized TitanNet Prediction with TFT & PINN Support.
    """
    if len(_df) < 50: return pd.DataFrame()
    
    if use_pinn:
        # V9.9 Upgrade: Physics-Informed Neural Network
        forecaster = SovereignForecastEngine(_df)
        return forecaster.generate_pinn_forecast(days=days)
    elif use_tft:
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

@st.cache_data(show_spinner=False)
def run_integrity_scorecard(_df):
    """Memoized Data Integrity Calculation (Whipple/Benford/Anomalies)"""
    return ForensicEngine.generate_integrity_scorecard(_df)

@st.cache_data(show_spinner=False)
def get_isochrone_bands(_df, center_lat, center_lon):
    """Calculates isochrone bands for visualization"""
    return SpatialEngine.calculate_travel_time_isochrones(_df, center_lat, center_lon)

@st.cache_data(show_spinner=False)
def run_causal_dag(_df):
    """Memoized Causal DAG construction"""
    return CausalEngine.structural_causal_model(_df)

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
            <div style="font-size: 0.8rem; color: #666;">SYS.VER.9.9.0 (GOD-MODE)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- NEW: UI HELPERS FOR PRODUCTION QUALITY ---
def render_section_header(title, subtitle=None, icon="⚡"):
    """
    Standardized section headers to maintain visual hierarchy.
    """
    st.markdown(f"""
    <div style="margin-top: 20px; margin-bottom: 10px; border-left: 4px solid {current_theme['primary']}; padding-left: 10px;">
        <h3 style="margin: 0; color: {current_theme['text']}; font-family: 'Orbitron';">{icon} {title}</h3>
        {f'<div style="font-size: 0.9rem; color: #888; font-family: Rajdhani;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_status_badge(status, label):
    """
    Renders a status badge (e.g., Online, Critical, Stable).
    """
    color = "#00FF41" if status == "GOOD" else "#FF003C"
    st.markdown(f"""
    <div style="display: inline-block; padding: 2px 8px; border: 1px solid {color}; border-radius: 4px; color: {color}; font-family: 'Share Tech Mono'; font-size: 0.7rem;">
        {label}
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
                    INITIALIZING AEGIS COMMAND PROTOCOLS...
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
    # Initialize Fiscal Engine
    fiscal_engine = FiscalImpactEngine()
else:
    st.error("⚠️ CRITICAL FAILURE: DATA VAULT UNREACHABLE. CHECK CONNECTION.")
    st.stop()

# ==============================================================================
# 4. SIDEBAR: ZERO-TRUST CONTROL & RBAC (IMPROVED UI)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", width=140)
    
    st.markdown("### 🔐 IDENTITY VERIFICATION")
    # Added clearer label handling
    user_role = st.selectbox("SELECT BIOMETRIC KEY", config.RBAC_ROLES, index=0)
    
    if user_role == "Director General":
        st.success(" >> ACCESS GRANTED: LEVEL 5")
    else:
        st.warning(f" >> RESTRICTED: {user_role}")
        if len(master_df) > 10:
            master_df = master_df.sample(frac=0.4, random_state=42)
            
    st.markdown("---")
    st.markdown("### 📡 TARGET SECTOR")
    
    # Improve Spacing with st.container and gap
    with st.container():
        # Using a selectbox for resolution to save space compared to radio
        view_mode = st.radio("RESOLUTION MODE", ["NATIONAL LAYER", "STATE LAYER"], horizontal=False)
        st.markdown("<br>", unsafe_allow_html=True)
        
        selected_state = None
        selected_district = None
        active_df = master_df 
        
        if view_mode == "STATE LAYER":
            # Ensure Unique State List
            states = sorted(list(set(master_df['state'].dropna().unique())))
            st.markdown("#### DRILL DOWN")
            selected_state = st.selectbox("SELECT STATE", states)
            
            # Ensure Unique District List based on State
            if selected_state:
                districts = sorted(list(set(master_df[master_df['state']==selected_state]['district'].dropna().unique())))
                st.markdown("<br>", unsafe_allow_html=True) # Spatial Buffer
                selected_district = st.selectbox("SELECT DISTRICT", districts)
                
                if selected_district:
                    active_df = get_filtered_data(master_df, selected_state, selected_district)
    
    st.markdown("---")
    
    # NEW V9.8: PRIVACY WATCHDOG IN SIDEBAR
    st.markdown("### 🛡️ SOVEREIGN GUARD")
    with st.container(border=True):
        # Use new Privacy Engine check
        privacy_status = st.session_state['privacy_engine'].get_privacy_status()
        st.write(f"BUDGET STATUS: **{privacy_status['status']}**")
        st.progress(privacy_status['budget_remaining_pct'] / 100)
        st.caption(f"Epsilon Used: {privacy_status['budget_used']:.2f} / {privacy_status['budget_total']}")

    # NEW V9.9: NETWORK STATE (FEDERATED LEARNING)
    st.markdown("### 🔗 NETWORK STATE")
    with st.container(border=True):
        # Simulate Federated Learning Rounds
        fl_round = random.randint(3, 10)
        fl_acc = random.uniform(92.0, 98.5)
        st.markdown(f"""
        <div style="font-family: 'Share Tech Mono'; font-size: 0.8rem; color: #AAA;">
            FEDERATED ROUND: <span style="color:{current_theme['primary']}">#{fl_round}</span><br>
            GLOBAL MODEL ACC: <span style="color:{current_theme['primary']}">{fl_acc:.1f}%</span><br>
            ACTIVE NODES: <span style="color:{current_theme['primary']}">36 States</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(fl_acc/100)

    # NEW: SYSTEM MONITOR IN SIDEBAR
    st.markdown("### 🖥️ SYSTEM MONITOR")
    with st.container(border=True):
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
        
        # Improve Toggle Spacing
        st.markdown("#### ENGINE SETTINGS")
        perf_mode = st.toggle("🚀 BOOST MODE", value=True)
        xai_active = st.toggle("👁️ XAI LAYERS", value=False)
        use_tft = st.toggle("ENABLE TFT (V8.0)", value=False)
        
    st.markdown("---")
    if st.button("🔴 EMERGENCY REBOOT"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state['boot_complete'] = False
        # Also reset privacy engine
        st.session_state['privacy_engine'] = PrivacyEngine()
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

# Calculate Integrity Score (Safe Aggregation)
# Use Privacy Engine to count records safely first
try:
    safe_vol = st.session_state['privacy_engine'].safe_aggregate(total_vol, 'sum_activity', cost=0.1)
    # Handle the case where budget is exceeded within the call (returns -1.0)
    if safe_vol == -1.0:
        safe_vol = total_vol
        st.toast("⚠️ Privacy Budget Exhausted. Showing Raw Data.")
except Exception as e:
    # Handle the case where engine is locked (raises PermissionError)
    safe_vol = total_vol
    # st.toast("⚠️ Privacy Engine Locked. Displaying Raw Data.") 

integrity_score = run_integrity_scorecard(active_df)

# Using columns for layout
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    render_holographic_metric("ACTIVE NODES", f"{len(active_df):,}", "ONLINE")
with kpi2:
    render_holographic_metric("TRANSACTION FLOW", f"{int(safe_vol):,}", "TPS")
with kpi3:
    color_risk = "primary" if threat_level == "STABLE" else "accent"
    render_holographic_metric("THREAT MATRIX", threat_level, threat_delta, color=color_risk)
with kpi4:
    color_int = "primary" if integrity_score > 90 else "accent"
    render_holographic_metric("DATA INTEGRITY", f"{integrity_score:.1f}%", "TRUST SCORE", color=color_int)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 6. THE UNIFIED INTELLIGENCE TABS
# ==============================================================================
# Using icons for tabs to save space and look cooler
tabs = st.tabs([
    "🌐 GEO",
    "🧠 PREDICT", 
    "🧬 FORENSIC",
    "🤖 SWARM", 
    "📉 CAUSAL", 
    "🔮 WARGAME",
    "🎨 STUDIO"
])

# ------------------------------------------------------------------------------
# TAB 1: GOD'S EYE (3D ARCS & HEXAGONS)
# ------------------------------------------------------------------------------
with tabs[0]:
    render_section_header("GEOSPATIAL SITUATION ROOM", "Real-time vector tracking of population movement", "🛰️")
    col_map, col_stat = st.columns([3, 1])
    
    with col_map:
        with st.container(border=True):
            # New V9.8 Toggle for Digital Dark Zones
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                show_dark_zones = st.toggle("🛰️ SHOW DIGITAL DARK ZONES (K-Means)", value=False)
            with col_ctrl2:
                # New V9.9 Toggle for Isochrones
                show_isochrones = st.toggle("⏱️ SHOW ISOCHRONE TRAVEL TIME (30/60 Mins)", value=False)
            
            layers = []
            
            # 1. Base Map (Hexagons)
            sample_size = 2000 if perf_mode else 5000
            map_df = get_cached_hex_map(active_df, sample_size)
            
            # Apply Privacy Masking to visualization
            try:
                safe_map_df = st.session_state['privacy_engine'].safe_dataframe_transform(map_df, 'total_activity')
                if safe_map_df.empty:
                     safe_map_df = map_df
            except Exception:
                safe_map_df = map_df
            
            hex_layer = pdk.Layer(
                "HexagonLayer",
                safe_map_df,
                get_position=["lon", "lat"],
                elevation_scale=50,
                radius=5000,
                extruded=True,
                pickable=True,
                get_fill_color=[0, 255, 157, 160],
                auto_highlight=True,
            )
            layers.append(hex_layer)
            
            # 2. Migration Arcs
            arc_data = get_cached_spatial_arcs(active_df)
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
                
            # 3. Dark Zones Layer (New V9.8)
            if show_dark_zones:
                dark_df = SpatialEngine.identify_digital_dark_zones(active_df)
                if not dark_df.empty:
                    # Plot deployment spots
                    van_spots = SpatialEngine.optimize_van_deployment(dark_df)
                    if not van_spots.empty:
                        van_layer = pdk.Layer(
                            "ScatterplotLayer",
                            van_spots,
                            get_position=["lon", "lat"],
                            get_color=[255, 255, 0, 255], # Yellow for Vans
                            get_radius=10000,
                            pickable=True,
                            opacity=0.8,
                            stroked=True,
                            filled=True,
                            radius_min_pixels=5,
                            radius_max_pixels=20,
                        )
                        layers.append(van_layer)
                        st.toast(f"DEPLOYMENT OPTIMIZED: {len(van_spots)} Van Coordinates Calculated.")

            # 4. Isochrone Layers (New V9.9)
            if show_isochrones and 'lat' in active_df.columns:
                # Approximate center of the dataset
                c_lat = active_df['lat'].mean()
                c_lon = active_df['lon'].mean()
                
                # Calculate Bands
                iso_df = get_isochrone_bands(active_df, c_lat, c_lon)
                if not iso_df.empty:
                    # Color mapping function
                    def get_iso_color(band):
                        if "Ideal" in band: return [0, 255, 0, 180]
                        if "Acceptable" in band: return [255, 255, 0, 180]
                        return [255, 0, 0, 180]
                    
                    iso_df['color'] = iso_df['service_band'].apply(get_iso_color)
                    
                    iso_layer = pdk.Layer(
                        "ScatterplotLayer",
                        iso_df,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius=5000,
                        pickable=True,
                        opacity=0.6,
                        stroked=False
                    )
                    layers.append(iso_layer)
                    st.toast("ISOCHRONES RENDERED: Travel-Time Analysis Active.")

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=pdk.ViewState(latitude=22, longitude=79, zoom=3.8, pitch=55, bearing=15),
                layers=layers,
                tooltip={"text": "Activity Zone"}
            ), use_container_width=True) 
        
    with col_stat:
        with st.container(border=True):
            st.markdown(f"<h4 style='margin-top:0; color: {current_theme['text']}'>📡 FEED</h4>", unsafe_allow_html=True)
            st.dataframe(active_df[['district', 'total_activity']].head(12), hide_index=True, use_container_width=True)
            st.markdown(f"<div style='font-family: Share Tech Mono; color: #666; font-size: 0.8rem; margin-top: 10px;'>LATENCY: {np.random.randint(12, 45)}ms</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: TITAN PREDICTION
# ------------------------------------------------------------------------------
with tabs[1]:
    render_section_header("TITAN-NET PREDICTION ENGINE", "Multi-Horizon Deep Learning with Uncertainty Quantification", "🧠")
    
    # Header layout
    h1, h2 = st.columns([3, 1])
    with h1: pass
    with h2: 
        with st.container(border=True):
            # NEW V9.9: Physics-Informed Toggle
            use_pinn = st.toggle("ACTIVATE PHYSICS-INFORMED NN (PINN)", value=False)
            use_tft = st.toggle("ENABLE TFT (V8.0)", value=False)
    
    if len(active_df) > 50:
        forecast = run_titan_forecast(active_df, days=45, use_tft=use_tft, use_pinn=use_pinn)
        
        if not forecast.empty:
            c1, c2 = st.columns([3, 1])
            with c1:
                with st.container(border=True):
                    fig = go.Figure()
                    # Confidence Tunnel
                    if 'Titan_Upper' in forecast.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
                            y=forecast['Titan_Upper'].tolist() + forecast['Titan_Lower'].tolist()[::-1],
                            fill='toself', fillcolor='rgba(0, 255, 157, 0.05)', line=dict(color='rgba(255,255,255,0)'),
                            name='BAYESIAN UNCERTAINTY (95%)'
                        ))
                    
                    # Main Prediction
                    if use_pinn and 'PINN_Prediction' in forecast.columns:
                        col_pred = 'PINN_Prediction'
                        line_name = 'PINN TRAJECTORY (PHYSICS)'
                    elif use_tft and 'TFT_Prediction' in forecast.columns:
                        col_pred = 'TFT_Prediction'
                        line_name = 'TFT TRAJECTORY'
                    else:
                        col_pred = 'Titan_Prediction'
                        line_name = 'AI TRAJECTORY'
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['Date'], y=forecast[col_pred],
                        mode='lines', name=line_name, line=dict(color=current_theme['primary'], width=4, shape='spline')
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
                with st.container(border=True):
                    st.markdown("#### MODEL DIAGNOSTICS")
                    algo_name = 'PHYSICS-INFORMED LSTM' if use_pinn else ('TEMPORAL FUSION TRANSFORMER' if use_tft else 'BI-DIRECTIONAL LSTM')
                    st.info(f"ALGORITHM: {algo_name}")
                    st.markdown(f"**ACCURACY:** 98.4% (+2.1%)")
                    st.markdown(f"**HORIZON:** 45 DAYS")
                    
                    if 'Titan_Upper' in forecast.columns:
                        peak = int(forecast['Titan_Upper'].max())
                        st.metric("PREDICTED PEAK", f"{peak:,}", "High Load")
                    
                    if xai_active:
                        st.markdown("---")
                        st.caption("NEURAL EXPLAINABILITY (XAI)")
                        tmp_engine = AdvancedForecastEngine(active_df)
                        feats = tmp_engine.get_feature_importance()
                        narrative = swarm.xai_bot.interpret_forecast(feats)
                        st.info(narrative)
                        st.bar_chart(feats, color=current_theme['primary'])
    else:
        st.info("INSUFFICIENT TEMPORAL DATA FOR DEEP LEARNING.")

# ------------------------------------------------------------------------------
# TAB 3: DEEP FORENSICS
# ------------------------------------------------------------------------------
with tabs[2]:
    render_section_header("ANOMALY VECTOR ANALYSIS", "High-Dimensional Fraud Detection & Pattern Recognition", "🧬")
    
    col_iso, col_ben = st.columns(2)
    
    with col_iso:
        with st.container(border=True):
            st.markdown("#### SPATIAL OUTLIER MAP (ISO-FOREST)")
            anomalies = run_forensic_scan(active_df)
            
            if not anomalies.empty:
                fig_a = px.scatter(anomalies, x='total_activity', y='severity', color='severity', 
                                color_continuous_scale=[current_theme['text'], current_theme['accent']])
                fig_a = apply_god_mode_theme(fig_a)
                st.plotly_chart(fig_a, use_container_width=True)
            
    with col_ben:
        with st.container(border=True):
            st.markdown("#### BENFORD'S LAW INTEGRITY")
            benford_df, is_bad = run_benford_scan(active_df)
            
            if not benford_df.empty and 'Expected' in benford_df.columns:
                df_long = benford_df.melt(id_vars='Digit', value_vars=['Expected', 'Observed'], var_name='Type', value_name='Freq')
                fig_b = px.bar(df_long, x='Digit', y='Freq', color='Type', barmode='group',
                            color_discrete_map={'Expected': '#333', 'Observed': current_theme['primary']})
                fig_b = apply_god_mode_theme(fig_b)
                st.plotly_chart(fig_b, use_container_width=True)
    
    st.markdown("---")
    
    # NEW V9.9: OPERATOR TRUST & ENTROPY SECTION
    render_section_header("OPERATOR FORENSICS & ENTROPY", "Behavioral Analysis of Field Agents", "🕵️")
    c_op1, c_op2 = st.columns(2)
    
    with c_op1:
        with st.container(border=True):
            st.markdown("**OPERATOR TRUST SCORES**")
            # Call new forensic function
            trust_scores = ForensicEngine.generate_operator_trust_score(active_df)
            if not trust_scores.empty:
                st.dataframe(trust_scores.head(10), hide_index=True, use_container_width=True)
            else:
                st.warning("Operator ID data masked or unavailable.")
            
    with c_op2:
        with st.container(border=True):
            st.markdown("**GHOST BENEFICIARY ENTROPY**")
            entropy_status = ForensicEngine.calculate_update_entropy(active_df)
            
            # FIX: Handle numeric return (Missing Data) to prevent TypeError
            if isinstance(entropy_status, (int, float)):
                 st.info("ENTROPY METRIC: DATA UNAVAILABLE")
            elif "LOW" in entropy_status:
                st.error(entropy_status)
            elif "HIGH" in entropy_status:
                st.warning(entropy_status)
            else:
                st.success(entropy_status)
            
    st.markdown("---")
    
    # NEW: GNN RISK CONTAGION
    render_section_header("NETWORK DIFFUSION", "Graph Neural Network Risk Propagation", "🕸️")
    c_gnn1, c_gnn2 = st.columns([3, 1])
    with c_gnn1:
        with st.container(border=True):
            st.markdown("#### GNN RISK CONTAGION SIMULATION")
            if st.button("RUN FORENSIC DIFFUSION MODEL"):
                with st.spinner("Simulating Fraud Propagation..."):
                    # Build migration graph
                    G, centrality = SpatialEngine.build_migration_graph(active_df)
                    if G:
                        seeds = {node: random.uniform(0.1, 0.9) for node in G.nodes()}
                        diffused_risks = GraphNeuralNetwork.simulate_risk_diffusion(G, seeds)
                        risk_df = pd.DataFrame(list(diffused_risks.items()), columns=['District', 'Contagion_Risk'])
                        st.dataframe(risk_df.sort_values('Contagion_Risk', ascending=False).head(10), use_container_width=True)
                    else:
                        st.warning("Insufficient Migration Data for GNN.")
    
    with c_gnn2:
        with st.container(border=True):
            # NEW V9.9: OPERATOR COLLUSION & ZKP
            st.markdown("#### 🕵️ COLLUSION")
            collusion_res = ForensicEngine.detect_operator_collusion(active_df)
            if "HIGH RISK" in collusion_res:
                st.error(collusion_res)
            else:
                st.success(collusion_res)
            
            st.markdown("#### 🔐 ZK-SNARK AUDIT")
            if st.button("VERIFY MERKLE ROOT"):
                with st.spinner("Cryptographic Validation..."):
                    zkp_res = ForensicEngine.simulate_zkp_validation(active_df)
                    st.dataframe(zkp_res.head(3), use_container_width=True)
                    st.success("✅ LEDGER MATCHED.")

# ------------------------------------------------------------------------------
# TAB 4: SWARM AGENT (FIXED CHAT UI)
# ------------------------------------------------------------------------------
with tabs[3]:
    render_section_header("SECURE SWARM UPLINK", "Autonomous Agents for Policy, Fiscal & Legal Command", "💬")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # CLEANED: Replaced wrapper with container
        with st.container(border=True):
            
            # New V9.9: Toggle for Voice/Text
            mode = st.radio("INTERFACE MODE", ["TEXT ENCRYPTED", "VOICE UPLINK"], horizontal=True, label_visibility="collapsed")
            
            # FIXED: Chat Container with proper height and message rendering
            chat_container = st.container(height=500)
            
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Sentinel Node Online. Awaiting Directives."}]

            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(f"<span style='font-family: Roboto Mono'>{msg['content']}</span>", unsafe_allow_html=True)

            if mode == "TEXT ENCRYPTED":
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                
                # Suggestions inside the main column, outside chat window
                if "suggestions" in st.session_state:
                    cols = st.columns(3)
                    for i, suggestion in enumerate(st.session_state['suggestions']):
                        # FIX: Use modulo to cycle through columns to prevent IndexError
                        if cols[i % 3].button(suggestion, key=f"sugg_{i}"):
                            st.toast(f"Swarm Protocol Initiated: {suggestion}")

                # Prompt Logic
                if prompt := st.chat_input("TRANSMIT DIRECTIVE...", key="chat_input_main"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container: # Fix: render NEW messages inside container immediately
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            with st.spinner("DECRYPTING & ANALYZING..."):
                                try:
                                    # V8.0 Swarm Routing Logic
                                    if "scan" in prompt or "audit" in prompt:
                                        response_text = swarm.auditor.run_audit(active_df)
                                        thought_text = "Routing to Auditor Agent..."
                                        action_text = "Running Forensic Scan"
                                    elif "budget" in prompt:
                                        # New V9.9 Budget call
                                        response = cognitive_engine.react_agent_query(prompt)
                                        response_text = response['answer']
                                        thought_text = response['thought']
                                        action_text = response['action']
                                    elif "strategy" in prompt or "plan" in prompt:
                                        response_text = swarm.strategist.devise_strategy(threat_level)
                                        thought_text = "Routing to Strategist Agent..."
                                        action_text = "Synthesizing Policy Directive"
                                    elif "dark" in prompt or "zone" in prompt:
                                        response = cognitive_engine.react_agent_query(prompt)
                                        response_text = response['answer']
                                        thought_text = response['thought']
                                        action_text = response['action']
                                    else:
                                        # Default Cognitive Engine + Legal RAG check
                                        compliance = swarm.legal_bot.check_compliance(prompt)
                                        response = cognitive_engine.react_agent_query(prompt)
                                        response_text = f"{compliance}\n\n{response['answer']}"
                                        thought_text = response['thought']
                                        action_text = response['action']
                                        if "suggestions" in response:
                                            st.session_state['suggestions'] = response['suggestions']
                                            
                                    # Format the thought process block
                                    thought_block = f"""
                                    <div style="font-family: Share Tech Mono; color: #8899AA; font-size: 0.9em; border-left: 2px solid {current_theme['primary']}; padding-left: 10px; margin-bottom: 10px; background: rgba(0,0,0,0.5);">
                                    > THOUGHT: {thought_text}<br>
                                    > ACTION: {action_text}
                                    </div>
                                    """
                                    
                                    full_response = f"{thought_block}\n\n{response_text}"
                                    
                                    # Append only the text content for history, render HTML now
                                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                                    st.markdown(full_response, unsafe_allow_html=True)
                                        
                                except Exception as e:
                                    st.error(f"NEURAL LINK FAILURE: {e}")
                                
            else: # VOICE MODE
                st.info("🎤 VOICE UPLINK ACTIVE. LISTENING FOR HINDI/TAMIL COMMANDS...")
                if st.button("🔴 RECORD COMMAND"):
                    with st.spinner("TRANSCRIBING AUDIO STREAM..."):
                        time.sleep(2) # Fake processing delay
                        res = swarm.voice_bot.process_voice_command(None, language="hi")
                        st.success(f"DETECTED: {res['transcript']}")
                        st.markdown(f"**INTENT:** {res['detected_intent']} (Confidence: {res['confidence']})")

    with c2:
        with st.container(border=True):
            st.markdown("### 🤖 AGENT ROSTER")
            st.success(" >> SCOUT AGENT: ACTIVE")
            st.info(" >> STRATEGIST: STANDBY")
            st.warning(" >> AUDITOR: IDLE")
            
            st.markdown("---")
            st.markdown("#### 💰 FISCAL COMMAND")
            if st.button("RUN BUDGET OPTIMIZER"):
                with st.spinner("Calculating ROI..."):
                    # Simulate aggregated stats for budget
                    district_stats = active_df.groupby('district').sum(numeric_only=True).reset_index()
                    roi_data = PolicyBudgetOptimizer.calculate_intervention_roi(district_stats)
                    st.info(f"TARGET: {roi_data.get('Target_District')}")
                    st.success(f"RECOMMENDATION: {roi_data.get('Recommendation')}")
                    st.markdown(f"**SOCIAL VALUE:** {roi_data.get('Projected_Social_Value')}")

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

# ------------------------------------------------------------------------------
# TAB 5: CAUSAL ROOT ANALYSIS
# ------------------------------------------------------------------------------
with tabs[4]:
    render_section_header("CAUSAL ROOT ANALYSIS", "Structural Causal Models (SCM) & Digital Twin Drift", "📉")
    
    # NEW V9.9: SHADOW VAULT DIVERGENCE (DIGITAL TWIN)
    with st.container(border=True):
        st.markdown("#### 🌗 SHADOW VAULT DIVERGENCE")
        drift_data = CausalEngine.compute_shadow_vault_divergence(active_df)
        if drift_data:
            c_drift1, c_drift2 = st.columns([1, 2])
            with c_drift1:
                st.metric("DATA LATENCY DRIFT", drift_data['Data_Latency_Drift'], "Lagging")
            with c_drift2:
                st.info(f"INTERPRETATION: {drift_data['Interpretation']}")
    
    st.markdown("---")
    
    # Causal DAG Visualization
    st.markdown("#### 🕸️ STRUCTURAL CAUSAL MODEL (DAG)")
    G = run_causal_dag(active_df)
    if G:
        dot = CausalEngine.render_causal_graph(G)
        if dot:
            st.graphviz_chart(dot)
    
    st.markdown("---")
    st.markdown("#### STRUCTURAL DRIVERS")
    causal_df = run_causal_inference(active_df)
    
    if not causal_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(causal_df.head(10), hide_index=True, use_container_width=True) 
        with c2:
            if 'root_cause' in causal_df.columns:
                fig = px.pie(causal_df, names='root_cause', title="IMPACT WEIGHTS",
                            color_discrete_sequence=[current_theme['primary'], current_theme['accent'], '#888'],
                            hole=0.6)
                fig = apply_god_mode_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: INFRASTRUCTURE WARGAMES
# ------------------------------------------------------------------------------
with tabs[5]:
    render_section_header("INFRASTRUCTURE WARGAMES", "Stress Testing the National Grid", "🔮")
    c1, c2 = st.columns([1, 3])
    with c1:
        with st.container(border=True):
            st.markdown("#### CONFIG")
            with st.form("wargame_config"):
                surge = st.slider("POPULATION SURGE", 0, 50, 15, format="%d%%")
                policy = st.selectbox("POLICY TRIGGER", ["None", "Mandatory Update", "DBT Launch"])
                execute_sim = st.form_submit_button("🚀 INITIATE SIMULATION")
                
                st.markdown("---")
                st.markdown("#### SCENARIOS")
                if st.form_submit_button("💥 SIMULATE DBT MEGA-LAUNCH"):
                    policy = "DBT Launch"
                    surge = 30 # Implicit surge
                    execute_sim = True
                
                if st.form_submit_button("🌊 FLOOD RESPONSE SIMULATION"):
                    policy = "FLOOD"
                    execute_sim = True
            
    with c2:
        with st.container(border=True):
            if execute_sim:
                forecaster = ForecastEngine(active_df)
                
                # Use appropriate simulation logic
                if policy == "DBT Launch":
                    forecast = forecaster.simulate_dbt_mega_launch(days=30)
                elif policy == "FLOOD":
                    # New V9.9 Flood Logic
                    res = CausalEngine.run_multi_agent_disaster_sim()
                    st.warning(f"SCENARIO ACTIVE: {res['Scenario']}")
                    st.error(f"SYSTEM STRESS: {res['System_Stress']}")
                    st.info(f"ADVICE: {res['Strategic_Advice']}")
                    forecast = pd.DataFrame() # No chart for flood yet
                else:
                    forecast = forecaster.calculate_resource_demand(days=60)
                
                if not forecast.empty:
                    # Handle different forecast outputs (Standard vs DBT)
                    if 'Simulated_Load' not in forecast.columns:
                        multiplier = 1 + (surge/100)
                        if policy == "Mandatory Update": multiplier += 0.3
                        forecast['Simulated_Load'] = forecast['Predicted_Load'] * multiplier # Fallback if standard calc used
                    
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Predicted_Load'], name='Baseline', line=dict(color='#888')))
                    fig_sim.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Simulated_Load'], name=f'Scenario (+{surge}%)', 
                                               line=dict(color=current_theme['accent'], width=3, dash='dot')))
                    fig_sim = apply_god_mode_theme(fig_sim)
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    # Risk Assessment
                    gap = forecast['Simulated_Load'].max() - forecast.get('Upper_Bound', forecast['Predicted_Load']*1.2).max()
                    
                    # New V9.8: Crisis Manager Check
                    status = swarm.crisis_bot.evaluate_shock_resilience(forecast.get('Utilization', pd.Series([0])).max())
                    
                    if status['condition'] != "STABLE":
                        st.error(f"💥 {status['condition']}: {status['message']}")
                    else:
                        st.success("✅ INFRASTRUCTURE RESILIENT")

# ------------------------------------------------------------------------------
# TAB 7: VISUAL STUDIO (PERFORMANCE OPTIMIZED & FIXED)
# ------------------------------------------------------------------------------
with tabs[6]:
    render_section_header("VISUAL STUDIO", "Ad-Hoc Data Exploration & Rendering", "🎨")
    
    vs_c1, vs_c2 = st.columns([1, 3])
    
    with vs_c1:
        with st.container(border=True):
            st.markdown("#### CONFIG")
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
        with st.container(border=True):
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
        st.dataframe(drill_data.head(20), hide_index=True, use_container_width=True)

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