import os
from pathlib import Path

class Config:
    """
    CENTRAL CONFIGURATION MATRIX | SENTINEL PRIME V9.5 [AEGIS]
    Controls Paths, Themes, AI Hyperparameters, and Security Protocols.
    Acts as the central nervous system for the Sovereign Digital Twin.
    """
    
    # ==========================================================================
    # 1. SYSTEM ARCHITECTURE & PATHS
    # ==========================================================================
    BASE_DIR = Path(__file__).parent.parent
    
    # Data Lake Paths
    DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    # Sovereign Storage (Local RAG & Logs)
    VECTOR_DB_PATH = BASE_DIR / "data" / "chromadb_store"
    LOGS_DIR = BASE_DIR / "logs"
    REPORTS_DIR = BASE_DIR / "reports"  # For generated Executive PDFs
    
    # Create critical directories if they don't exist
    for _dir in [PROCESSED_DIR, LOGS_DIR, REPORTS_DIR]:
        _dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 2. SOVEREIGN VISUAL IDENTITY (OMNI-PRESENCE THEME)
    # ==========================================================================
    # Core Palette
    THEME_BG = "#050505"       # Void Black
    THEME_PRIMARY = "#00FF9D"  # Neon Mint
    THEME_SECONDARY = "#1A1A1A"
    THEME_ALERT = "#FF2A2A"    # Cyber Red for Fraud
    THEME_TEXT = "#E0F0E0"     # Phosphor White
    
    # Extended Visuals for PyDeck & Plotly
    MAP_STYLE = "mapbox://styles/mapbox/dark-v10"
    DEFAULT_MAP_ZOOM = 3.8
    DEFAULT_MAP_PITCH = 55
    
    # UI Layout Constants
    SIDEBAR_WIDTH = "expanded" # Options: "auto", "expanded", "collapsed"
    
    # ==========================================================================
    # 3. AI HYPERPARAMETERS & NEURAL CONFIG
    # ==========================================================================
    # Legacy LSTM
    LSTM_HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    
    # V8.0 Temporal Fusion Transformer (TFT)
    TFT_HIDDEN_SIZE = 128
    TFT_ATTENTION_HEADS = 4
    TFT_DROPOUT = 0.1
    QUANTILE_LEVELS = [0.1, 0.5, 0.9] # Probabilistic Forecasting Bounds
    
    # Graph Neural Network (GNN)
    GNN_NEIGHBOR_DEPTH = 2
    PAGERANK_ALPHA = 0.85
    
    # Anomaly Detection Thresholds
    ANOMALY_THRESHOLD = 0.01        # Isolation Forest Contamination
    BENFORD_TOLERANCE = 0.05        # Max allowed deviation from Benford's Law
    WHIPPLE_INDEX_THRESHOLD = 125   # Age Heaping Alert Level (125+ is "Rough")
    
    FORECAST_HORIZON = 30
    
    # ==========================================================================
    # 4. SECURITY & RBAC PROTOCOLS (Zero-Trust)
    # ==========================================================================
    RBAC_ROLES = ["Director General", "State Secretary", "District Magistrate", "Auditor"]
    DEFAULT_ROLE = "Director General"
    
    # Sovereign Privacy Flags
    MASK_PII = True                 # Auto-mask Aadhaar/Mobile numbers in logs
    LOCAL_COMPUTE_ONLY = True       # Prevent accidental cloud uploads
    
    # ==========================================================================
    # 5. EXTERNAL API KEYS & INTEGRATIONS
    # ==========================================================================
    # CRITICAL FIX: Added default fallback to empty string to prevent AttributeError
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
    
    # Local LLM Endpoint (Ollama)
    OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
    DEFAULT_LLM_MODEL = "llama3"

config = Config()