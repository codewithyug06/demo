import pandas as pd
import numpy as np
import math
from sklearn.ensemble import IsolationForest
from config.settings import config

class ForensicEngine:
    """
    ADVANCED FORENSIC SUITE v9.7 (GOD MODE)
    Includes: Whipple (UN Standard), Benford, Digit Fingerprint, ISOLATION FOREST,
    and Multi-Dataset Correlation Logic.
    """
    
    @staticmethod
    def calculate_whipple(df):
        """
        Detects Age Heaping (rounding to 0 or 5).
        UPDATED: Uses the official UN Formula for Whipple's Index.
        Range 23-62 is the standard demographic window for this test.
        """
        if 'age' not in df.columns:
            # Fallback to total_activity if age column missing (Legacy support)
            if 'total_activity' not in df.columns: return 0.0
            # Simple heuristic if no age data: Check modulo 5 of activity counts
            suspicious = df['total_activity'].apply(lambda x: 1 if x % 5 == 0 else 0).sum()
            return (suspicious / len(df)) * 500 if len(df) > 0 else 0

        # Filter relevant ages (23 to 62)
        target_ages = df[(df['age'] >= 23) & (df['age'] <= 62)]
        if target_ages.empty: return 0.0
        
        total_pop = len(target_ages)
        
        # Count ages ending in 0 or 5
        heaping_count = target_ages[target_ages['age'] % 5 == 0].shape[0]
        
        # Formula: (Sum of Age 25,30...60 / 1/5 * Sum of all ages 23-62) * 100
        whipple_index = (heaping_count / (total_pop / 5)) * 100
        
        return whipple_index

    @staticmethod
    def calculate_benfords_law(df):
        """
        Detects data fabrication.
        Returns standardized columns: ['Digit', 'Expected', 'Observed', 'Deviation']
        """
        if 'total_activity' not in df.columns: return pd.DataFrame(), False
        
        def get_leading_digit(x):
            try:
                s = str(int(x))
                return int(s[0]) if s[0] != '0' else None
            except: return None
            
        digits = df['total_activity'].apply(get_leading_digit).dropna()
        if len(digits) < 50: return pd.DataFrame(), False
        
        observed = digits.value_counts(normalize=True).sort_index()
        expected = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        
        analysis = pd.DataFrame({
            'Digit': range(1, 10),
            'Expected': [expected[d] for d in range(1, 10)],
            'Observed': [observed.get(d, 0) for d in range(1, 10)]
        })
        analysis['Deviation'] = abs(analysis['Expected'] - analysis['Observed'])
        
        return analysis, analysis['Deviation'].mean() > config.BENFORD_TOLERANCE

    @staticmethod
    def calculate_digit_fingerprint(df):
        """
        Detects manual data entry bias (Last Digit Analysis).
        """
        if 'total_activity' not in df.columns: return 0.0
        
        def get_last_digit(x): 
            try: return int(str(int(x))[-1])
            except: return -1
        
        last_digits = df['total_activity'].apply(get_last_digit)
        last_digits = last_digits[last_digits != -1]
        
        if len(last_digits) == 0: return 0.0
        
        counts = last_digits.value_counts(normalize=True).sort_index()
        fingerprint_score = sum([abs(counts.get(d, 0) - 0.1) for d in range(10)])
        
        return fingerprint_score

    # ==========================================================================
    # NEW GOD-LEVEL FEATURE: MULTIVARIATE ANOMALY DETECTION (UNSUPERVISED AI)
    # ==========================================================================
    @staticmethod
    def detect_high_dimensional_fraud(df):
        """
        Uses Isolation Forest to detect anomalies based on multi-dimensional features:
        (Activity Volume, Latitude, Longitude).
        
        Finds 'Spatial Outliers' - centers that have high activity in low-density zones.
        """
        if len(df) < 50: return pd.DataFrame()
        
        # Prepare Features
        # Using Lat/Lon acts as a proxy for "Location Context"
        # In a real scenario, we would add 'Time of Day', 'Operator ID', etc.
        features = df[['total_activity', 'lat', 'lon']].fillna(0)
        
        # Model: Isolation Forest
        # Designed to detect anomalies that are 'few and different'
        clf = IsolationForest(contamination=config.ANOMALY_THRESHOLD, random_state=42)
        df = df.copy() 
        df['anomaly_score'] = clf.fit_predict(features)
        
        # -1 indicates anomaly, 1 indicates normal.
        anomalies = df[df['anomaly_score'] == -1].copy()
        
        # Calculate Severity (How far from the mean activity)
        mean_activity = df['total_activity'].mean()
        anomalies['severity'] = anomalies['total_activity'] / (mean_activity + 1e-5)
        
        return anomalies.sort_values('severity', ascending=False)

    # ==========================================================================
    # NEW V9.7 FEATURES: TELEDENSITY & SCORECARDS
    # ==========================================================================
    @staticmethod
    def generate_integrity_scorecard(df):
        """
        Aggregates multiple forensic tests into a single 'Trust Score' (0-100).
        Used by the Strategist Agent for Policy Briefs.
        """
        score = 100.0
        
        # 1. Benford Penalty
        _, is_bad_benford = ForensicEngine.calculate_benfords_law(df)
        if is_bad_benford: score -= 15
        
        # 2. Whipple Penalty (Demographic Quality)
        whipple = ForensicEngine.calculate_whipple(df)
        if whipple > 125: score -= 20 # Rough Data
        if whipple > 175: score -= 40 # Very Rough (Fraud Likely)
        
        # 3. Anomaly Penalty
        anomalies = ForensicEngine.detect_high_dimensional_fraud(df)
        if not anomalies.empty:
            score -= (len(anomalies) / len(df)) * 50
            
        return max(0, min(100, score))

    @staticmethod
    def cross_correlate_teledensity(aadhaar_df, telecom_df):
        """
        Bivariate Analysis: Correlates Aadhaar Activity with Telecom Density.
        Goal: Identify 'Digital Dark Zones' where Aadhaar updates lag due to connectivity.
        """
        if telecom_df.empty or 'teledensity' not in telecom_df.columns:
            return "TELECOM DATA MISSING"
            
        # Merge on District (Fuzzy match recommended in prod, strict here for speed)
        # Assumes normalized headers from IngestionEngine
        if 'district' not in aadhaar_df.columns or 'district' not in telecom_df.columns:
            return "SCHEMA MISMATCH"
            
        merged = pd.merge(aadhaar_df, telecom_df, on='district', how='inner')
        if len(merged) < 10: return "INSUFFICIENT OVERLAP"
        
        correlation = merged['total_activity'].corr(merged['teledensity'])
        
        if correlation < 0.3:
            return "WEAK CORRELATION: Infrastructure deployment issue detected."
        return f"STRONG CORRELATION ({correlation:.2f}): Digital access drives enrolment."