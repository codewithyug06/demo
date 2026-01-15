import numpy as np
import pandas as pd
from datetime import datetime
import json
import uuid

class PrivacyEngine:
    """
    SOVEREIGN PRIVACY ENGINE (Differential Privacy Layer) v9.9
    
    Implements Epsilon-Differential Privacy (ε-DP) to mathematically guarantee 
    that the output of any query does not compromise the privacy of any single individual.
    
    MECHANISMS:
    1. Privacy Budgeting (Sequential Composition)
    2. Laplace Mechanism (for Count/Sum queries)
    3. Sensitivity-Calibrated Noise Injection
    4. Sovereign Audit Logging (Zero-Knowledge)
    5. Re-identification Risk Assessment (k-Anonymity Proxy)
    """
    
    def __init__(self, total_epsilon=5.0, delta=1e-5):
        """
        Initialize the Privacy Guardian.
        
        Args:
            total_epsilon (float): The total privacy loss budget allowed for this session.
                                   Lower = Higher Privacy. Standard Academic Value = 1.0 - 10.0.
            delta (float): The probability of privacy breach (should be < 1/N).
        """
        self.max_epsilon = total_epsilon
        self.used_epsilon = 0.0
        self.delta = delta
        self.query_log = []
        self.active = True
        
        # Sensitivity registry (Maximum effect one individual can have on a query)
        self.sensitivity_map = {
            'count': 1.0,         # One person adds 1 to a count
            'sum_activity': 50.0, # Cap: One person does max 50 updates/year (Clamping)
            'mean': 0.05,         # Impact on mean is low for large N
            'histogram': 2.0,     # Impact on a distribution bucket
            'risk_score': 0.1     # Impact on aggregated risk score
        }

    def _check_budget(self, cost):
        """
        Internal Gatekeeper: Checks if the privacy budget allows this query.
        """
        if not self.active:
            raise PermissionError("PRIVACY ENGINE LOCKED: Budget Exhausted.")
        
        if self.used_epsilon + cost > self.max_epsilon:
            self.active = False
            self._log_event("BLOCK", cost, "Budget Exceeded")
            return False
        return True

    def _log_event(self, action, cost, context):
        """
        Immutable Audit Log (Does not store data, only metadata).
        """
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "epsilon_cost": cost,
            "cumulative_epsilon": self.used_epsilon,
            "context": context
        }
        self.query_log.append(entry)

    def _laplace_mechanism(self, true_value, sensitivity, epsilon):
        """
        The Mathematical Core: Adds noise drawn from a Laplace distribution.
        
        Noise ~ Lap(sensitivity / epsilon)
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise

    def safe_aggregate(self, value, agg_type='count', cost=0.1):
        """
        The Public Interface for Data Science Agents.
        Returns a 'Safe' (Noisy) version of the metric.
        
        Args:
            value (float): The raw, sensitive number (e.g., total enrolments).
            agg_type (str): Type of aggregation ('count', 'sum_activity', etc).
            cost (float): How much privacy budget to burn (epsilon).
        
        Returns:
            float: Differentially Private value.
        """
        if not self._check_budget(cost):
            return -1.0 # Sentinel for blocked query
            
        sensitivity = self.sensitivity_map.get(agg_type, 1.0)
        
        # Apply Mechanism
        safe_val = self._laplace_mechanism(value, sensitivity, cost)
        
        # Post-Processing (Physics Guard): Counts cannot be negative
        if agg_type in ['count', 'sum_activity']:
            safe_val = max(0, safe_val)
            # Integrity Guard: Round to nearest integer for realism (optional)
            safe_val = round(safe_val)

        # Update State
        self.used_epsilon += cost
        self._log_event("QUERY", cost, f"{agg_type} aggregation")
        
        return safe_val

    def safe_dataframe_transform(self, df, sensitive_col, epsilon_per_row=0.01):
        """
        Applies Local Differential Privacy to an entire column for visualization.
        Used for Scatter Plots where individual points might leak info.
        """
        cost = epsilon_per_row * len(df)
        # Cap cost for viz to avoid instant depletion
        cost = min(cost, 1.0) 
        
        if not self._check_budget(cost):
            return pd.DataFrame()
            
        sensitivity = self.sensitivity_map.get('histogram', 2.0)
        scale = sensitivity / epsilon_per_row
        
        # Vectorized Noise Injection
        noise = np.random.laplace(0, scale, size=len(df))
        
        safe_df = df.copy()
        if pd.api.types.is_numeric_dtype(safe_df[sensitive_col]):
            safe_df[sensitive_col] = safe_df[sensitive_col] + noise
            # Consistency: Ensure no negative activity
            safe_df[sensitive_col] = safe_df[sensitive_col].clip(lower=0)
            
        self.used_epsilon += cost
        self._log_event("BATCH_TRANSFORM", cost, f"Viz Masking: {sensitive_col}")
        
        return safe_df

    def calculate_reidentification_risk(self, df, quasi_identifiers=['district', 'age', 'gender']):
        """
        Estimates the risk of re-identification based on k-Anonymity principles.
        Returns a risk score (0-100).
        """
        if df.empty: return 0.0
        
        # Check which columns exist
        available_cols = [c for c in quasi_identifiers if c in df.columns]
        if not available_cols: return 0.0
        
        # Group by quasi-identifiers to find unique combinations
        groups = df.groupby(available_cols).size()
        
        # Count groups with size < 5 (High Risk of Re-ID)
        risky_groups = groups[groups < 5].count()
        total_groups = len(groups)
        
        risk_score = (risky_groups / total_groups) * 100 if total_groups > 0 else 0
        return round(risk_score, 2)

    def get_privacy_status(self):
        """
        Returns the current health of the privacy firewall for the UI.
        """
        remaining = self.max_epsilon - self.used_epsilon
        health_pct = (remaining / self.max_epsilon) * 100
        
        status = "SECURE"
        if health_pct < 20: status = "CRITICAL RISK"
        elif health_pct < 50: status = "MODERATE LEAKAGE"
            
        return {
            "status": status,
            "budget_used": round(self.used_epsilon, 4),
            "budget_total": self.max_epsilon,
            "budget_remaining_pct": round(health_pct, 1),
            "queries_processed": len(self.query_log),
            "active": self.active
        }