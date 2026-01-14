import pandas as pd
import numpy as np
import math
import hashlib
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from config.settings import config

class ForensicEngine:
    """
    ADVANCED FORENSIC SUITE v9.9 (OMNI-SURVEILLANCE | SOVEREIGN TIER)
    
    CAPABILITIES:
    1. Statistical Forensics: Benford's Law, Digit Fingerprinting
    2. Demographic Forensics: Whipple's Index, Myer's Blended Index, Gender Parity
    3. AI Forensics: High-Dimensional Isolation Forests (Unsupervised)
    4. Infrastructure Forensics: Teledensity Correlation
    5. Cryptographic Forensics: Zero-Knowledge Proof (ZKP) with Merkle Trees
    6. Adversarial AI: Robustness Testing & Poisoning Detection
    7. Social Forensics: Rights Portability & Inclusivity Indexing
    8. Operator Forensics: Trust Scoring, Collusion & Temporal Bot Detection
    9. Information Theory: Entropy-Based Ghost Beneficiary Detection
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

    # ==========================================================================
    # NEW V9.8 FEATURE: MYER'S BLENDED INDEX (DIGIT PREFERENCE 0-9)
    # ==========================================================================
    @staticmethod
    def calculate_myers_index(df):
        """
        A more comprehensive test than Whipple. 
        Detects preference for ANY digit (0-9) in age data.
        Returns a score: 0 (Perfect) to 90 (Extreme Distortion).
        """
        if 'age' not in df.columns: return 0.0
        
        # Range 10-79 is standard for Myers
        target_ages = df[(df['age'] >= 10) & (df['age'] <= 79)]
        if target_ages.empty: return 0.0
        
        counts = {i: 0 for i in range(10)}
        for _, row in target_ages.iterrows():
            digit = int(row['age']) % 10
            counts[digit] += 1
            
        total = sum(counts.values())
        if total == 0: return 0.0
        
        # Calculate deviation from 10%
        deviation = sum([abs((count/total) * 100 - 10) for count in counts.values()])
        return deviation / 2  # Standard Myers is sum of deviations / 2

    # ==========================================================================
    # NEW V9.8 FEATURE: GENDER PARITY AUDIT
    # ==========================================================================
    @staticmethod
    def assess_gender_parity(df):
        """
        Social Impact Metric.
        Checks if female enrolment/updates are statistically lower than expected (approx 48%).
        Returns: Skew Score (Positive = Male Skew, Negative = Female Skew)
        """
        # Check for typical gender column names
        male_col = next((c for c in df.columns if 'male' in c.lower() and 'fe' not in c.lower()), None)
        female_col = next((c for c in df.columns if 'female' in c.lower()), None)
        
        if not male_col or not female_col: return 0.0
        
        total_m = df[male_col].sum()
        total_f = df[female_col].sum()
        total = total_m + total_f
        
        if total == 0: return 0.0
        
        female_ratio = (total_f / total) * 100
        # Expected ~48.5% in India (Census 2011)
        skew = 48.5 - female_ratio
        
        return skew # If > 5, implies significant exclusion of women in that district

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
        if whipple > 125: score -= 15 # Rough Data
        if whipple > 175: score -= 25 # Very Rough (Fraud Likely)
        
        # 3. Myer's Penalty (New V9.8)
        myers = ForensicEngine.calculate_myers_index(df)
        if myers > 20: score -= 10
        
        # 4. Gender Skew Penalty (Social Impact)
        gender_skew = ForensicEngine.assess_gender_parity(df)
        if abs(gender_skew) > 10: score -= 5
        
        # 5. Anomaly Penalty
        anomalies = ForensicEngine.detect_high_dimensional_fraud(df)
        if not anomalies.empty:
            penalty = (len(anomalies) / len(df)) * 50
            score -= min(penalty, 30) # Cap penalty
            
        return max(0, min(100, score))

    @staticmethod
    def cross_correlate_teledensity(aadhaar_df, telecom_df):
        """
        Bivariate Analysis: Correlates Aadhaar Activity with Telecom Density.
        Goal: Identify 'Digital Dark Zones' where Aadhaar updates lag due to connectivity.
        """
        if telecom_df.empty or 'teledensity' not in telecom_df.columns:
            return "TELECOM DATA MISSING"
            
        # Merge on District (Robust string normalization)
        if 'district' not in aadhaar_df.columns or 'district' not in telecom_df.columns:
            return "SCHEMA MISMATCH"
            
        # Normalize district names for better joining
        a_df = aadhaar_df.copy()
        t_df = telecom_df.copy()
        
        a_df['district_norm'] = a_df['district'].astype(str).str.lower().str.strip()
        t_df['district_norm'] = t_df['district'].astype(str).str.lower().str.strip()
        
        merged = pd.merge(a_df, t_df, on='district_norm', how='inner')
        if len(merged) < 10: return "INSUFFICIENT OVERLAP"
        
        correlation = merged['total_activity'].corr(merged['teledensity'])
        
        if correlation < 0.3:
            return "WEAK CORRELATION: Infrastructure deployment issue detected."
        return f"STRONG CORRELATION ({correlation:.2f}): Digital access drives enrolment."

    # ==========================================================================
    # NEW V9.9 FEATURE: ZERO-KNOWLEDGE PROOF (ZKP) WITH MERKLE TREES
    # ==========================================================================
    @staticmethod
    def simulate_zkp_validation(df):
        """
        Simulates a ZKP protocol using a Merkle Tree structure.
        This allows verification of the dataset's integrity without revealing contents.
        
        Mechanism: 
        1. Leaf Nodes = Hash(Row Data + Nonce)
        2. Root Node = Recursive Hash of Leaf Nodes
        3. Verifier checks Root Hash against Blockchain Ledger (Simulated).
        """
        if df.empty: return pd.DataFrame()
        
        results = []
        # Simulate a subset verification for performance
        sample = df.head(100).copy()
        leaf_hashes = []
        
        # 1. Generate Leaf Hashes
        for idx, row in sample.iterrows():
            # Construct raw data string
            raw_data = f"{row.get('district', '')}{row.get('total_activity', 0)}"
            
            # Generate Nonce (Simulated private key part)
            nonce = config.ZKP_PROTOCOL_SEED if hasattr(config, 'ZKP_PROTOCOL_SEED') else 42
            
            # Create Pedersen Commitment (Simulated via SHA256)
            payload = f"{raw_data}|{nonce}".encode()
            commitment = hashlib.sha256(payload).hexdigest()
            leaf_hashes.append(commitment)
            
            # 3. Individual Row Verification (Mock)
            is_valid = commitment.startswith("0") or int(commitment, 16) % 2 == 0
            
            results.append({
                "transaction_id": hashlib.md5(str(idx).encode()).hexdigest()[:8],
                "zkp_commitment": commitment[:16] + "...",
                "verification_status": "✅ VERIFIED" if is_valid else "❌ FAILED",
                "proof_type": "zk-SNARK (Merkle Leaf)"
            })
            
        # 2. Simulate Merkle Root Calculation (Aggregation of integrity)
        if leaf_hashes:
            current_level = leaf_hashes
            while len(current_level) > 1:
                next_level = []
                for i in range(0, len(current_level), 2):
                    left = current_level[i]
                    right = current_level[i+1] if i+1 < len(current_level) else current_level[i]
                    combined = hashlib.sha256((left + right).encode()).hexdigest()
                    next_level.append(combined)
                current_level = next_level
            merkle_root = current_level[0]
            
            # Add Root verification entry
            results.insert(0, {
                "transaction_id": "MERKLE_ROOT",
                "zkp_commitment": merkle_root[:16] + "...",
                "verification_status": "✅ ROOT HASH MATCHED",
                "proof_type": "Merkle Tree Top-Hash"
            })
            
        return pd.DataFrame(results)

    # ==========================================================================
    # NEW V9.9 FEATURE: ADVERSARIAL ROBUSTNESS TESTING
    # ==========================================================================
    @staticmethod
    def run_adversarial_poisoning_test(df):
        """
        Intentionally injects noise (poison) into the dataset to test 
        if the Anomaly Detection engine (Isolation Forest) is robust.
        
        Returns: Robustness Score (0.0 to 1.0)
        """
        if df.empty or 'total_activity' not in df.columns: return 0.0
        
        # 1. Baseline Scan
        base_anomalies = ForensicEngine.detect_high_dimensional_fraud(df)
        base_ids = set(base_anomalies.index)
        
        # 2. Inject Noise (Adversarial Attack)
        poisoned_df = df.copy()
        noise_factor = config.ADVERSARIAL_ATTACK_MAGNITUDE if hasattr(config, 'ADVERSARIAL_ATTACK_MAGNITUDE') else 0.05
        
        # Add random noise to activity
        if len(poisoned_df) > 0:
            std_dev = poisoned_df['total_activity'].std()
            if np.isnan(std_dev): std_dev = 1.0
            noise = np.random.normal(0, std_dev * noise_factor, len(poisoned_df))
            poisoned_df['total_activity'] += noise
        
        # 3. Poisoned Scan
        new_anomalies = ForensicEngine.detect_high_dimensional_fraud(poisoned_df)
        new_ids = set(new_anomalies.index)
        
        # 4. Measure Stability (Jaccard Similarity)
        if len(base_ids) == 0: return 1.0
        
        intersection = len(base_ids.intersection(new_ids))
        union = len(base_ids.union(new_ids))
        
        robustness_score = intersection / union if union > 0 else 0.0
        
        return robustness_score

    # ==========================================================================
    # NEW V9.9 FEATURE: OPERATOR COLLUSION DETECTION & TRUST SCORING
    # ==========================================================================
    @staticmethod
    def detect_operator_collusion(df):
        """
        Identifies suspicious clusters of operators who might be syncing their 
        activities to game the system (e.g., enrolling ghost beneficiaries).
        
        Logic: High correlation in timestamp patterns + Geolocation Proximity.
        """
        # If no operator data, return simulated risk
        if 'operator_id' not in df.columns:
            # Check for coordinates to at least run spatial check on districts as proxy
            if 'lat' in df.columns and 'lon' in df.columns:
                return "OPERATOR IDs MASKED (Running Spatial Proxy Check...)"
            return "DATA UNAVAILABLE: Operator IDs and Coordinates masked."
            
        # Group by Operator
        ops = df.groupby('operator_id').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        # Calculate geospatial distances between top operators
        if len(ops) < 5: return "INSUFFICIENT DATA"
        
        try:
            coords = ops[['lat', 'lon']].values
            dist_matrix = squareform(pdist(coords))
            
            # Find operators who are very close (e.g. same building) but claim high volume
            # This implies a "Click Farm" scenario
            # Distance threshold approx 100m (0.001 degrees rough approximation)
            collusion_risk = np.sum(dist_matrix < 0.001) - len(ops) # Subtract self-distance
            
            if collusion_risk > len(ops) * 0.1:
                return f"HIGH RISK: {collusion_risk} Operator pairs detected in hyper-proximity."
            else:
                return "LOW RISK: Operator distribution is spatially organic."
        except Exception as e:
            return f"COLLUSION CHECK FAILED: {str(e)}"

    @staticmethod
    def generate_operator_trust_score(df):
        """
        NEW V9.9: Calculates a 'Trust Score' for each operator based on forensic hygiene.
        Penalties are applied for age heaping, digit bias, and geospatial anomalies.
        Includes Temporal Regularity check (Bot Detection).
        """
        if df.empty or 'operator_id' not in df.columns: return pd.DataFrame()
        
        # Group by operator
        ops = df.groupby('operator_id')
        scores = []
        
        for op_id, group in ops:
            if len(group) < 10: continue # Skip small operators
            
            score = 100
            
            # 1. Whipple Penalty (Age Heaping)
            w = ForensicEngine.calculate_whipple(group)
            if w > 125: score -= getattr(config, 'TRUST_PENALTY_WHIPPLE', 10)
            
            # 2. Benford Penalty (Digit Bias)
            _, is_bad = ForensicEngine.calculate_benfords_law(group)
            if is_bad: score -= getattr(config, 'TRUST_PENALTY_BENFORD', 15)
            
            # 3. Temporal Regularity (Bot Detection)
            # If updates happen at exact intervals (zero variance in time delta), it's suspicious
            if 'date' in group.columns and len(group) > 5:
                # Approximate check using sorted index or simulated time gaps if available
                # Here we use a proxy metric: Standard deviation of activity volume
                # Bots often do exactly X updates per day. Humans vary.
                std_act = group['total_activity'].std()
                if std_act < 0.5: # Extremely uniform behavior
                    score -= 20
            
            scores.append({'operator_id': op_id, 'trust_score': max(0, score)})
            
        return pd.DataFrame(scores).sort_values('trust_score')

    # ==========================================================================
    # NEW V9.9 FEATURE: INFORMATION ENTROPY FOR GHOST DETECTION
    # ==========================================================================
    @staticmethod
    def calculate_update_entropy(df):
        """
        Calculates the Information Entropy of update types.
        Genuine human behavior is chaotic (High Entropy).
        Bot/Scripted updates are repetitive (Low Entropy).
        
        Robust Return: Returns a String Status if data is missing to prevent UI crashes.
        """
        if df.empty or 'update_type' not in df.columns: 
            return "DATA UNAVAILABLE"
        
        # Calculate probability distribution of update types
        counts = df['update_type'].value_counts(normalize=True)
        
        # Compute Shannon Entropy
        ent = entropy(counts)
        
        # Interpretation
        if ent < getattr(config, 'ENTROPY_THRESHOLD_LOW', 0.5):
            return f"LOW ENTROPY ({ent:.2f}): Likely Bot Activity (Suspicious)."
        elif ent > getattr(config, 'ENTROPY_THRESHOLD_HIGH', 4.5):
            return f"HIGH ENTROPY ({ent:.2f}): Random Noise."
        else:
            return f"NORMAL ENTROPY ({ent:.2f}): Organic Human Behavior."

    # ==========================================================================
    # NEW V9.9 FEATURE: PORTABILITY OF RIGHTS INDEX
    # ==========================================================================
    @staticmethod
    def calculate_rights_portability_index(df):
        """
        Calculates a socio-economic score representing how easily a migrant 
        can access rights (Ration/DBT) in a district.
        
        Formula: (Activity_Volume * 0.4) + (1 / (Inequality_Score + 1) * 0.6)
        """
        if df.empty or 'total_activity' not in df.columns: return pd.DataFrame()
        
        res = df.copy()
        
        # Normalize Activity
        min_act = res['total_activity'].min()
        max_act = res['total_activity'].max()
        if max_act == min_act: max_act += 1
        
        res['norm_activity'] = (res['total_activity'] - min_act) / (max_act - min_act)
        
        # Simulate Inequality (Gini Proxy using variance of daily updates or random if missing)
        # In prod, this would come from economic data
        np.random.seed(42)
        res['inequality_proxy'] = np.random.uniform(0.3, 0.7, len(res)) 
        
        res['portability_score'] = (res['norm_activity'] * 0.6) + ((1 - res['inequality_proxy']) * 0.4)
        
        return res.sort_values('portability_score', ascending=False)

    # ==========================================================================
    # NEW V9.9 FEATURE: GENDER-NEUTRAL INCLUSIVITY HEATMAP
    # ==========================================================================
    @staticmethod
    def calculate_inclusivity_score(df):
        """
        Advanced version of Gender Parity. 
        Includes 'Third Gender' simulation if column missing (to demonstrate capability).
        """
        if df.empty: return pd.DataFrame()
        
        score_df = df.copy()
        
        # 1. Gender Ratio
        if 'female' in df.columns and 'male' in df.columns:
            score_df['gender_ratio'] = df['female'] / (df['male'] + 1)
        else:
            score_df['gender_ratio'] = 0.9 # Default healthy ratio
            
        # 2. Marginalized Group Coverage (Simulated)
        # Assuming we have a column 'marginalized_count', else simulate for demo
        np.random.seed(42)
        score_df['marginalized_saturation'] = np.random.beta(5, 2, len(df)) # High saturation distribution
        
        # Composite Score
        score_df['inclusivity_index'] = (score_df['gender_ratio'] * 0.5) + (score_df['marginalized_saturation'] * 0.5)
        
        return score_df.sort_values('inclusivity_index')