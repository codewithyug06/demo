import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    
    # SOVEREIGN VISUAL IDENTITY
    THEME_BG = "#050505"       # Void Black
    THEME_PRIMARY = "#00FF9D"  # Neon Mint
    THEME_SECONDARY = "#1A1A1A"
    
    # AI PARAMETERS
    LSTM_HIDDEN_SIZE = 64
    ANOMALY_THRESHOLD = 0.01

config = Config()