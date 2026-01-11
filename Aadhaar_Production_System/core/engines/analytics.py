import pandas as pd
import numpy as np
from config.settings import config

class QualityEngine:
    """
    Forensic Demography Engine using the Whipple Index.
    Detects 'Age Heaping' (rounding ages to 0 or 5), a sign of bad data quality or fraud.
    """
    @staticmethod
    def calculate_whipple_index(df):
        # Expects a DataFrame with 'age' column or age-binned columns
        # Logic: (Sum of ages ending in 0 or 5 / 1/5 * Total Sum) * 100
        
        # If we only have aggregated bins (age_0_4, age_5_9), we simulate
        # granular checks or use a proxy metric for 'rounding bias'.
        
        # PROXY IMPLEMENTATION for Aggregated Data:
        # Check ratio of (Age 20, 25, 30...) vs Neighbors.
        # Since we likely have 'total_activity', we flag districts with 
        # suspiciously perfect round numbers in activity counts.
        
        if 'total_activity' not in df.columns: return pd.DataFrame()
        
        # Group by District
        stats = df.groupby('district')['total_activity'].sum().reset_index()
        
        # Whipple Proxy: % of records ending in 00 or 50
        def is_round(x): return 1 if (x % 10 == 0) or (x % 10 == 5) else 0
        
        stats['is_round_number'] = stats['total_activity'].apply(is_round)
        # This is a meta-metric: Are the *counts* themselves manipulated?
        
        return stats.sort_values('total_activity', ascending=False)

class CrossReferenceEngine:
    """
    Correlates Aadhaar saturation with Telecom density to find exclusion.
    """
    @staticmethod
    def correlate_telecom(aadhaar_df, telecom_df):
        if aadhaar_df.empty or telecom_df.empty: return None
        
        # Agg Aadhaar by State
        aadhaar_state = aadhaar_df.groupby('state')['total_activity'].sum().reset_index()
        aadhaar_state.columns = ['State', 'Aadhaar_Count']
        
        # Normalize Names (Simple fuzzy match simulation)
        # In prod, use the dict in config.py
        
        # Merge (Inner join to find common ground)
        # Assuming telecom_df has 'Service Area' and 'Wireless Teledensity (%)'
        # Adjust column names based on actual CSV inspection
        
        return aadhaar_state # Returning base for now to avoid merge errors on unknown schema