import os
from pathlib import Path

class Config:
    # PATHS
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    # VISUAL IDENTITY (Sovereign Style)
    THEME_BG = "#0e1117"
    THEME_PRIMARY = "#00FFC2" # Neon Cyan
    THEME_SECONDARY = "#6c757d"
    
    # ANALYTICS THRESHOLDS
    WHIPPLE_BAD_QUALITY = 125  # >125 implies rough/bad data
    ANOMALY_CONTAMINATION = 0.01

config = Config()