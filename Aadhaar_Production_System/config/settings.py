import os
from pathlib import Path

class Config:
    # PATHS
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    MODEL_DIR = BASE_DIR / "core" / "models"
    
    # VISUAL IDENTITY (Cyberpunk/Sovereign)
    THEME_BG = "#0e1117"
    THEME_PRIMARY = "#00FFC2" # Neon Cyan
    THEME_WARNING = "#FF4B4B"
    
    # ML PARAMETERS
    LSTM_LOOKBACK = 7
    ANOMALY_THRESHOLD = 0.01
    WHIPPLE_THRESHOLD = 125 # Index > 125 indicates bad data quality
    
    # TELECOM MAPPING (Simplified for Demo if no common key exists)
    STATE_MAPPING = {
        "Delhi": "Delhi",
        "Uttar Pradesh": "U.P. (West)", # Example mapping logic
        "Maharashtra": "Maharashtra"
    }

config = Config()