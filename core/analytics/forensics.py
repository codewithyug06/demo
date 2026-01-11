import pandas as pd
import numpy as np
import math

class ForensicEngine:
    """
    ADVANCED FORENSIC SUITE v2.0
    Modules:
    1. Whipple Index (Age Heaping Detection)
    2. Benford's Law (Data Fabrication Detection) [NEW]
    """
    
    @staticmethod
    def calculate_whipple(df):
        """
        EXISTING FUNCTION: Detects Age Heaping.
        (Preserved as requested)
        """
        if 'total_activity' not in df.columns: return pd.DataFrame()
        
        # Group by district to get granular stats
        stats = df.groupby(['state', 'district'])['total_activity'].sum().reset_index()
        
        # Logic: Check if the total activity count ends in 0 or 5 (Heaping Proxy)
        stats['is_suspicious'] = stats['total_activity'].apply(lambda x: 1 if x % 5 == 0 else 0)
        
        return stats.sort_values('total_activity', ascending=False)

    @staticmethod
    def calculate_benfords_law(df):
        """
        NEW FUNCTION: BENFORD'S LAW ANALYSIS
        Detects if the dataset looks 'naturally occurring' or 'manually invented'.
        Natural data follows a specific distribution of leading digits (1 is most common, 9 is least).
        """
        if 'total_activity' not in df.columns: 
            return pd.DataFrame(), False

        # 1. Extract Leading Digit
        # Convert to string, take first char, convert back to int. 0 is ignored.
        def get_leading_digit(x):
            s = str(int(x))
            return int(s[0]) if s[0] != '0' else None

        digits = df['total_activity'].apply(get_leading_digit).dropna()
        
        if len(digits) < 100:
            return pd.DataFrame(), False # Not enough data for stat significance

        # 2. Calculate Observed Frequency
        observed_counts = digits.value_counts(normalize=True).sort_index()
        
        # 3. Calculate Expected Frequency (Benford's Law: P(d) = log10(1 + 1/d))
        expected_counts = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        
        # 4. Create Comparison DataFrame
        analysis_df = pd.DataFrame({
            'Digit': range(1, 10),
            'Expected_Benford': [expected_counts[d] for d in range(1, 10)],
            'Observed_Real': [observed_counts.get(d, 0) for d in range(1, 10)]
        })
        
        # Calculate Deviation
        analysis_df['Deviation'] = abs(analysis_df['Expected_Benford'] - analysis_df['Observed_Real'])
        
        # Flag if deviation is high (simple threshold for demo)
        is_anomalous = analysis_df['Deviation'].mean() > 0.05
        
        return analysis_df, is_anomalous