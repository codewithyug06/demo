import os
from pathlib import Path

class Config:
    """
    CENTRAL CONFIGURATION MATRIX | SENTINEL PRIME V9.7 [AEGIS COMMAND]
    Controls Paths, Themes, AI Hyperparameters, and Security Protocols.
    Acts as the central nervous system for the Sovereign Digital Twin.
    
    COMPLIANCE: UIDAI Data Security Guidelines 2026
    ARCH: Zero-Trust / Local-First / Sovereign
    """
    
    # ==========================================================================
    # 1. SYSTEM ARCHITECTURE & PATHS
    # ==========================================================================
    BASE_DIR = Path(__file__).parent.parent
    
    # Data Lake Paths (Ingestion Layer)
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
    # Core Palette - Cyber Warfare Aesthetic
    THEME_BG = "#050505"       # Void Black
    THEME_PRIMARY = "#00FF9D"  # Neon Mint (Matrix Green)
    THEME_SECONDARY = "#1A1A1A"
    THEME_ALERT = "#FF2A2A"    # Cyber Red for Fraud/Risk
    THEME_TEXT = "#E0F0E0"     # Phosphor White
    THEME_ACCENT = "#FF003C"   # Critical Alert Red
    
    # Extended Visuals for PyDeck & Plotly
    MAP_STYLE = "mapbox://styles/mapbox/dark-v10"
    DEFAULT_MAP_ZOOM = 3.8
    DEFAULT_MAP_PITCH = 55
    DEFAULT_MAP_BEARING = 15
    
    # UI Layout Constants
    SIDEBAR_WIDTH = "expanded" # Options: "auto", "expanded", "collapsed"
    NEWS_TICKER_SPEED = 6      # Scrolling speed for the "Live Intel" footer
    
    # ==========================================================================
    # 3. AI HYPERPARAMETERS & NEURAL CONFIG
    # ==========================================================================
    # Legacy LSTM (Bi-Directional)
    LSTM_HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    
    # V8.0 Temporal Fusion Transformer (TFT) - Explainable AI
    TFT_HIDDEN_SIZE = 128
    TFT_ATTENTION_HEADS = 4
    TFT_DROPOUT = 0.1
    
    # Probabilistic Forecasting (Confidence Tunnel)
    QUANTILE_LEVELS = [0.1, 0.5, 0.9] # p10 (Lower), p50 (Median), p90 (Upper)
    
    # Graph Neural Network (GNN) - Risk Contagion
    GNN_NEIGHBOR_DEPTH = 2
    PAGERANK_ALPHA = 0.85
    
    # ==========================================================================
    # 4. FORENSIC & INTEGRITY THRESHOLDS (WINNING CRITERIA)
    # ==========================================================================
    # Anomaly Detection
    ANOMALY_THRESHOLD = 0.01        # Isolation Forest Contamination Rate
    
    # Benford's Law (Digit Frequency Analysis)
    BENFORD_TOLERANCE = 0.05        # Max allowed deviation (5%) before flagging
    
    # Whipple's Index (Age Heaping/Demographic Quality)
    # United Nations Standard for Age Accuracy
    WHIPPLE_INDEX_THRESHOLD = 125   # Global Alert Level
    WHIPPLE_RANGES = {
        "HIGHLY_ACCURATE": (0, 105),
        "FAIR_DATA": (105, 110),
        "APPROXIMATE": (110, 125),
        "ROUGH": (125, 175),
        "VERY_ROUGH": (175, 999)    # Indicates massive manual entry error/fraud
    }
    
    # Forecast Horizon
    FORECAST_HORIZON = 30
    
    # ==========================================================================
    # 5. WARGAME SIMULATOR PARAMETERS (DBT MEGA-LAUNCH)
    # ==========================================================================
    # Stress Testing Constants for Infrastructure
    DBT_LAUNCH_TRAFFIC_MULTIPLIER = 5.0  # Simulate 5x load during PM-Kisan launch
    INFRA_FAILURE_POINT = 0.95           # Server crashes at 95% utilization
    LATENCY_PENALTY_FACTOR = 0.4         # 40% slowdown per 10% overload
    
    # ==========================================================================
    # 6. SECURITY & RBAC PROTOCOLS (Zero-Trust)
    # ==========================================================================
    RBAC_ROLES = ["Director General", "State Secretary", "District Magistrate", "Auditor"]
    DEFAULT_ROLE = "Director General"
    
    # Sovereign Privacy Flags (GDPR/Data Protection Bill Compliant)
    MASK_PII = True                 # Force-mask Aadhaar/Mobile numbers in ingestion
    LOCAL_COMPUTE_ONLY = True       # Prevent accidental cloud uploads
    
    # Regex Patterns for PII Sanitization
    PII_REGEX_AADHAAR = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    PII_REGEX_MOBILE = r'\b[6-9]\d{9}\b'
    
    # ==========================================================================
    # 7. EXTERNAL API KEYS & INTEGRATIONS
    # ==========================================================================
    # CRITICAL FIX: Added default fallback to empty string to prevent AttributeError
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
    
    # Local LLM Endpoint (Ollama - Sovereign AI)
    # This enables RAG without sending data to OpenAI
    OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
    DEFAULT_LLM_MODEL = "llama3"
    
    # Vector DB Config
    VECTOR_DB_COLLECTION = "uidai_policy_docs"

config = Config()