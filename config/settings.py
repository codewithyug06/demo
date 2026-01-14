import os
from pathlib import Path

class Config:
    """
    CENTRAL CONFIGURATION MATRIX | SENTINEL PRIME V9.9 [AEGIS COMMAND]
    Controls Paths, Themes, AI Hyperparameters, Security Protocols, and Agent Behavior.
    Acts as the central nervous system for the Sovereign Digital Twin.
    
    COMPLIANCE: UIDAI Data Security Guidelines 2026 (Internal Only)
    ARCH: Zero-Trust / Local-First / Sovereign / Cloud-Native
    """
    
    # ==========================================================================
    # 1. SYSTEM ARCHITECTURE & PATHS
    # ==========================================================================
    BASE_DIR = Path(__file__).parent.parent
    
    # Data Lake Paths (Ingestion Layer)
    # Allow overriding via env vars for containerized deployments
    DATA_DIR = Path(os.getenv("SENTINEL_DATA_DIR", BASE_DIR / "data" / "raw"))
    PROCESSED_DIR = Path(os.getenv("SENTINEL_PROCESSED_DIR", BASE_DIR / "data" / "processed"))
    
    # Sovereign Storage (Local RAG & Logs)
    VECTOR_DB_PATH = Path(os.getenv("SENTINEL_VECTOR_DB", BASE_DIR / "data" / "chromadb_store"))
    LOGS_DIR = Path(os.getenv("SENTINEL_LOGS_DIR", BASE_DIR / "logs"))
    REPORTS_DIR = Path(os.getenv("SENTINEL_REPORTS_DIR", BASE_DIR / "reports"))  # For generated Executive PDFs
    ASSETS_DIR = Path(os.getenv("SENTINEL_ASSETS_DIR", BASE_DIR / "assets"))    # For Logos/Fonts in PDF generation
    
    # Create critical directories if they don't exist
    for _dir in [PROCESSED_DIR, LOGS_DIR, REPORTS_DIR, ASSETS_DIR, VECTOR_DB_PATH]:
        try:
            _dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {_dir}: {e}")

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
    
    # Chart Specific Palettes (Plotly)
    COLOR_SEQUENCE_FORECAST = [THEME_PRIMARY, "#0088FF", "#FFCC00"]
    COLOR_SEQUENCE_RISK = ["#00FF00", "#FFFF00", "#FF0000"] # Green to Red
    
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
    LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", 64))
    LSTM_LAYERS = int(os.getenv("LSTM_LAYERS", 2))
    
    # V8.0 Temporal Fusion Transformer (TFT) - Explainable AI
    TFT_HIDDEN_SIZE = int(os.getenv("TFT_HIDDEN_SIZE", 128))
    TFT_ATTENTION_HEADS = int(os.getenv("TFT_HEADS", 4))
    TFT_DROPOUT = float(os.getenv("TFT_DROPOUT", 0.1))
    
    # Probabilistic Forecasting (Confidence Tunnel)
    QUANTILE_LEVELS = [0.1, 0.5, 0.9] # p10 (Lower), p50 (Median), p90 (Upper)
    
    # Graph Neural Network (GNN) - Risk Contagion
    GNN_NEIGHBOR_DEPTH = 2
    PAGERANK_ALPHA = 0.85
    
    # NEW V9.8: Bayesian Neural Network (BNN) - Uncertainty Quantification
    BNN_CONFIDENCE_INTERVAL = 0.95  # 95% Confidence required for auto-approval
    BNN_MC_DROPOUT_SAMPLES = int(os.getenv("BNN_SAMPLES", 50))     # Number of Monte Carlo samples for uncertainty
    
    # NEW V9.8: Spatiotemporal GCN (ST-GCN)
    STGCN_TEMPORAL_WINDOW = 7       # Look back 7 days for spatial contagion
    
    # ==========================================================================
    # 4. FORENSIC & INTEGRITY THRESHOLDS (WINNING CRITERIA)
    # ==========================================================================
    # Anomaly Detection
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.01))        # Isolation Forest Contamination Rate
    
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
    
    # Integrity Scorecard Weights (Used in Forensics Engine)
    # How much each factor contributes to the total 100% Trust Score
    SCORECARD_WEIGHTS = {
        "BENFORD_PENALTY": 15,
        "WHIPPLE_ROUGH_PENALTY": 20,
        "WHIPPLE_BAD_PENALTY": 40,
        "ANOMALY_FACTOR": 50        # Multiplier for % of anomalous nodes
    }
    
    # NEW V9.8: Zero-Knowledge Proof (ZKP) Parameters
    ZKP_PROTOCOL_SEED = int(os.getenv("ZKP_SEED", 42))          # Seed for cryptographic simulation
    ZKP_VALIDATION_STRENGTH = "SHA-256"
    
    # NEW V9.8: Adversarial Robustness
    ADVERSARIAL_ATTACK_MAGNITUDE = 0.05 # 5% noise injection to test robustness
    
    # Forecast Horizon
    FORECAST_HORIZON = 30
    
    # ==========================================================================
    # 5. WARGAME SIMULATOR PARAMETERS (DBT MEGA-LAUNCH)
    # ==========================================================================
    # Stress Testing Constants for Infrastructure
    DBT_LAUNCH_TRAFFIC_MULTIPLIER = 5.0  # Simulate 5x load during PM-Kisan launch
    INFRA_FAILURE_POINT = 0.95           # Server crashes at 95% utilization
    LATENCY_PENALTY_FACTOR = 0.4         # 40% slowdown per 10% overload
    
    # Strategy Mitigation Constants
    OFFLINE_MODE_LATENCY_REDUCTION = 0.3 # Moving to offline reduces latency by 30%
    MOBILE_VAN_DEPLOYMENT_CAPACITY = 2000 # Each van handles 2000 txns/day
    
    # NEW V9.8: Kubernetes (K8s) Auto-Scaling Simulation
    K8S_AUTOSCALE_THRESHOLD = 0.85       # Spin up pods at 85% load
    K8S_POD_CAPACITY = 5000              # Transactions per pod
    K8S_SPINUP_TIME = 45                 # Seconds to spin up a new pod
    
    # ==========================================================================
    # 6. SECURITY & RBAC PROTOCOLS (Zero-Trust)
    # ==========================================================================
    RBAC_ROLES = ["Director General", "State Secretary", "District Magistrate", "Auditor"]
    DEFAULT_ROLE = "Director General"
    
    # Sovereign Privacy Flags (GDPR/Data Protection Bill Compliant)
    MASK_PII = os.getenv("MASK_PII", "True").lower() == "true" # Force-mask Aadhaar/Mobile numbers in ingestion
    LOCAL_COMPUTE_ONLY = True       # Prevent accidental cloud uploads
    
    # Regex Patterns for PII Sanitization (High-Speed Filtering)
    PII_REGEX_AADHAAR = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    PII_REGEX_MOBILE = r'\b[6-9]\d{9}\b'
    
    # NEW V9.8: Hardware-Accelerated Encryption (TPM Simulation)
    TPM_ENABLED = True
    ENCRYPTION_STANDARD = "AES-256-GCM"
    
    # ==========================================================================
    # 7. EXTERNAL API KEYS & INTEGRATIONS
    # ==========================================================================
    # CRITICAL FIX: Added default fallback to empty string to prevent AttributeError
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
    
    # Local LLM Endpoint (Ollama - Sovereign AI)
    # This enables RAG without sending data to OpenAI
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    DEFAULT_LLM_MODEL = "llama3"
    
    # Vector DB Config (Legal-RAG)
    VECTOR_DB_COLLECTION = "aadhaar_act_legal_docs"
    
    # NEW V9.8: Cross-Lingual Voice Support
    SUPPORTED_LANGUAGES = ["en", "hi", "ta", "te", "bn"]
    
    # ==========================================================================
    # 8. GNN & SPATIAL PHYSICS (NEW)
    # ==========================================================================
    # Controls how fast fraud/risk spreads in the GNN model
    RISK_DIFFUSION_DECAY = 0.6      # 60% of risk is passed to neighbors per step
    RISK_DIFFUSION_STEPS = 3        # How many 'hops' to simulate
    
    # Digital Dark Zone Definition
    DARK_ZONE_ACTIVITY_THRESHOLD = 500 # Districts below this activity are suspect
    DARK_ZONE_ISOLATION_THRESHOLD = 0.3 # Normalized distance from nearest hub
    
    # NEW V9.8: Dynamic Isochrone Analysis
    ISOCHRONE_TRAVEL_MODES = ["walking", "driving", "boat"]
    RIVER_BARRIER_PENALTY = 0.8     # Travel speed reduced by 80% if river crossing needed
    
    # ==========================================================================
    # 9. PERFORMANCE & CACHING
    # ==========================================================================
    # Streamlit Cache TTL (Time To Live) in seconds
    CACHE_TTL_DATA = 3600           # Keep raw data for 1 hour
    CACHE_TTL_MODELS = 7200         # Keep trained models for 2 hours
    CACHE_TTL_PLOTS = 600           # Redraw plots every 10 mins
    
    # ==========================================================================
    # 10. REGIONAL LINGUISTICS (ADVANCED NLP)
    # ==========================================================================
    # NEW V9.8: Regional Phonetic Mapping for Fuzzy Matching
    # Handles variations like "Mohd" vs "Mohammed" or "C." vs "Chellapandian"
    REGIONAL_PHONETIC_MAPPING = {
        "NORTH": {"mohd": "mohammed", "md": "mohammed", "kr": "kumar"},
        "SOUTH": {"c": "chellapandian", "m": "murugan", "k": "karuppasamy"},
        "EAST": {"sk": "shaikh", "moni": "monisha"},
        "WEST": {"patel": "patil", "v": "vasant"}
    }

    # ==========================================================================
    # 11. FISCAL & BUDGETARY OPTIMIZATION (NEW V9.9)
    # ==========================================================================
    # Cost modeling for the Fiscal Optimizer Agent
    FISCAL_UNIT_COST_ENROLMENT_KIT = 50000  # Cost per kit in INR
    FISCAL_UNIT_COST_MOBILE_VAN = 1200000   # Cost per van per year
    FISCAL_UNIT_COST_OPERATOR_TRAINING = 5000
    
    # ROI Calculation Factors
    FISCAL_VALUE_PER_SATURATION_POINT = 1000000 # Abstract value of 1% saturation increase
    FISCAL_FRAUD_PREVENTION_VALUE = 500 # Saved per fraudulent enrolment blocked

    # ==========================================================================
    # 12. LEGAL & COMPLIANCE GUARDRAILS (NEW V9.9)
    # ==========================================================================
    # Vector DB paths for specific acts
    LEGAL_DOCS_PATH = BASE_DIR / "data" / "legal_docs"
    COMPLIANCE_MANDATES = {
        "AADHAAR_ACT_2016": ["Section 7", "Section 29", "Section 33"],
        "DPDP_ACT_2023": ["Section 8", "Section 9"]
    }
    
    # ==========================================================================
    # 13. ADVANCED PHYSICS & PINN PARAMETERS (NEW V9.9)
    # ==========================================================================
    # Physics-Informed Neural Network (PINN) friction coefficients
    # Used to model population spread like fluid dynamics
    FRICTION_COEFFICIENTS = {
        "PLAINS": 0.1,
        "HILLS": 0.6,
        "FOREST": 0.8,
        "URBAN": 0.05
    }
    
    # ==========================================================================
    # 14. FEDERATED LEARNING & DIFFERENTIAL PRIVACY (NEW V9.9)
    # ==========================================================================
    # Simulation settings for State-to-National weight aggregation
    FEDERATED_ROUNDS = 10
    FEDERATED_MIN_CLIENTS = 3
    DIFFERENTIAL_PRIVACY_EPSILON = 1.0  # Privacy budget
    
    # ==========================================================================
    # 15. ENTROPY & GHOST DETECTION (NEW V9.9)
    # ==========================================================================
    # Information Entropy thresholds to detect machine-generated vs human data
    ENTROPY_THRESHOLD_LOW = 0.5  # Too regular (Suspicious)
    ENTROPY_THRESHOLD_HIGH = 4.5 # Too chaotic (Random noise)
    
    # ==========================================================================
    # 16. DISTRIBUTED COMPUTE CLUSTER (NEW V9.9)
    # ==========================================================================
    # Dask / Ray Cluster Configuration
    COMPUTE_BACKEND = os.getenv("COMPUTE_BACKEND", "local") # Options: "dask", "ray", "local"
    DASK_SCHEDULER_PORT = 8786
    RAY_DASHBOARD_PORT = 8265
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4)) # Default for local simulation
    
    # ==========================================================================
    # 17. OPERATOR TRUST SCORING (NEW V9.9)
    # ==========================================================================
    # Penalties applied to Operator Trust Score (starts at 100)
    TRUST_PENALTY_WHIPPLE = 10  # If age heaping detected
    TRUST_PENALTY_BENFORD = 15  # If digit bias detected
    TRUST_PENALTY_GEO_TRIANGLE = 25 # If collusion detected
    TRUST_RECOVERY_RATE = 0.5   # Points recovered per month of clean behavior

config = Config()