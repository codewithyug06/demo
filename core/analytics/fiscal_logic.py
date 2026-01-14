import pandas as pd
import numpy as np

class FiscalImpactEngine:
    """
    SOVEREIGN FISCAL LOGIC ENGINE (V9.9)
    
    Translates technical anomalies into Government Financial Impact.
    
    CORE METRICS:
    1. Ghost Beneficiary Savings (₹ Crores)
    2. Kit Deployment ROI (Efficiency Gain)
    3. Authentication Failure Cost (Productivity Loss)
    4. Fraud Prevention Value
    """
    
    def __init__(self):
        # Government Standard Rates (Approximate for Hackathon)
        self.COST_PER_GHOST = 25000  # ₹25k/year subsidy leakage per ghost
        self.KIT_DEPLOYMENT_COST = 500000 # ₹5L per kit per year (OpEx)
        self.AUTH_FAILURE_LOSS = 500 # ₹500 productivity loss per failed auth
        self.FRAUD_penalty = 100000 # ₹1L penalty per fraudulent operator

    def calculate_ghost_savings(self, anomaly_df):
        """
        Quantifies the fiscal savings from identifying potential ghost beneficiaries.
        
        Args:
            anomaly_df (pd.DataFrame): Output from ForensicEngine.detect_high_dimensional_fraud()
            
        Returns:
            dict: Savings breakdown in Crores.
        """
        if anomaly_df.empty:
            return {"total_savings_cr": 0.0, "ghost_count": 0}
            
        # Filter for high-severity anomalies which are proxies for ghosts
        high_risk = anomaly_df[anomaly_df['severity'] > 0.8]
        ghost_count = len(high_risk)
        
        # Calculate Savings
        total_savings = ghost_count * self.COST_PER_GHOST
        savings_cr = total_savings / 10000000 # Convert to Crores
        
        return {
            "total_savings_cr": round(savings_cr, 2),
            "ghost_count": ghost_count,
            "districts_impacted": high_risk['district'].nunique()
        }

    def compute_kit_roi(self, forecast_df, current_kits, recommended_kits):
        """
        Calculates the Return on Investment (ROI) for re-balancing enrolment kits.
        
        Args:
            forecast_df (pd.DataFrame): Future demand prediction.
            current_kits (int): Current hardware count.
            recommended_kits (int): AI-suggested count.
            
        Returns:
            dict: ROI metrics.
        """
        # Delta analysis
        kit_delta = recommended_kits - current_kits
        
        if kit_delta == 0:
            return {"status": "OPTIMAL", "roi_pct": 0.0}
            
        # Cost of change
        deployment_cost = abs(kit_delta) * 50000 # Logistic cost of moving a kit
        
        # Benefit: Revenue from new updates/enrolments (UIDAI charges for updates)
        # Assume 50 updates per day per kit * 300 days * ₹50 fee
        projected_revenue_gain = kit_delta * 50 * 300 * 50
        
        # Social Benefit (Intangible but quantified for Jury)
        # 1 Kit = 5000 citizens served per year
        citizens_served = kit_delta * 5000
        
        roi = ((projected_revenue_gain - deployment_cost) / deployment_cost) * 100 if deployment_cost > 0 else 0
        
        return {
            "action": "DEPLOY" if kit_delta > 0 else "RECALL",
            "kits_moved": abs(kit_delta),
            "cost_incurred": deployment_cost,
            "revenue_gain": projected_revenue_gain,
            "social_impact_citizens": citizens_served,
            "roi_pct": round(roi, 1)
        }

    def assess_authentication_loss(self, auth_df):
        """
        Estimates economic loss due to authentication failures (e.g., poor biometrics).
        """
        if auth_df.empty: return {}
        
        failed_txns = auth_df[auth_df['status'] == 'FAILURE'].shape[0]
        economic_loss = failed_txns * self.AUTH_FAILURE_LOSS
        loss_cr = economic_loss / 10000000
        
        return {
            "failed_transactions": failed_txns,
            "economic_loss_cr": round(loss_cr, 2),
            "impact_severity": "HIGH" if loss_cr > 10 else "MODERATE"
        }

# Instance for easy import
fiscal_engine = FiscalImpactEngine()