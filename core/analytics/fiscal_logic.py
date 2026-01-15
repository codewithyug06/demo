import pandas as pd
import numpy as np

class FiscalImpactEngine:
    """
    SOVEREIGN FISCAL LOGIC ENGINE (V9.9)
    
    Translates technical anomalies into Government Financial Impact.
    Now includes advanced ROI models for Mobile Vans, Training, and Fraud Prevention.
    
    CORE METRICS:
    1. Ghost Beneficiary Savings (₹ Crores)
    2. Kit Deployment ROI (Efficiency Gain)
    3. Authentication Failure Cost (Productivity Loss)
    4. Fraud Prevention Value
    5. Mobile Van Efficiency Index (New)
    6. Operator Training ROI (New)
    """
    
    def __init__(self):
        # Government Standard Rates (Approximate for Hackathon)
        self.COST_PER_GHOST = 25000  # ₹25k/year subsidy leakage per ghost
        self.KIT_DEPLOYMENT_COST = 500000 # ₹5L per kit per year (OpEx)
        self.AUTH_FAILURE_LOSS = 500 # ₹500 productivity loss per failed auth
        self.FRAUD_PENALTY = 100000 # ₹1L penalty per fraudulent operator
        
        # New Fiscal Constants
        self.MOBILE_VAN_COST = 1200000 # ₹12L per van per year
        self.TRAINING_COST_PER_OP = 5000 # ₹5k per operator
        self.SUBSIDY_ACCURACY_GAIN = 0.02 # 2% gain in subsidy efficiency per 1% saturation increase

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

    # ==========================================================================
    # NEW V9.9: ADVANCED ROI MODELS (WINNING CRITERIA)
    # ==========================================================================
    def calculate_mobile_van_efficiency(self, dark_zones_df):
        """
        Calculates the ROI of deploying Mobile Vans vs Static Centers in dark zones.
        Vans are expensive but have higher reach in remote areas.
        """
        if dark_zones_df.empty: return {}
        
        target_pop = len(dark_zones_df) * 5000 # Approx unserved pop
        
        # Option A: Static Centers
        static_centers_needed = max(1, int(target_pop / 3000))
        cost_static = static_centers_needed * self.KIT_DEPLOYMENT_COST
        coverage_static = 0.6 # Static centers miss 40% of remote pop
        
        # Option B: Mobile Vans
        vans_needed = max(1, int(target_pop / 10000)) # Vans cover more ground
        cost_mobile = vans_needed * self.MOBILE_VAN_COST
        coverage_mobile = 0.95 # Vans reach 95%
        
        # ROI Logic: Cost per citizen reached
        cpp_static = cost_static / (target_pop * coverage_static)
        cpp_mobile = cost_mobile / (target_pop * coverage_mobile)
        
        recommendation = "MOBILE VANS" if cpp_mobile < cpp_static else "STATIC CENTERS"
        
        return {
            "Target_Population": target_pop,
            "Cost_Per_Person_Static": round(cpp_static, 2),
            "Cost_Per_Person_Mobile": round(cpp_mobile, 2),
            "Recommendation": recommendation,
            "Efficiency_Gain": f"{abs(cpp_static - cpp_mobile)/cpp_static * 100:.1f}%"
        }

    def calculate_training_program_roi(self, operator_df):
        """
        Calculates if retraining operators with low trust scores is cheaper than banning them.
        """
        if operator_df.empty: return {}
        
        # Identify low performing operators
        low_trust_ops = operator_df[operator_df['trust_score'] < 50]
        count = len(low_trust_ops)
        
        if count == 0: return {"status": "NO TRAINING NEEDED"}
        
        # Cost of Training
        training_cost = count * self.TRAINING_COST_PER_OP
        
        # Cost of Banning (Recruitment + Setup of new operator)
        replacement_cost = count * 25000 # Cost to onboard new agency
        
        savings = replacement_cost - training_cost
        
        return {
            "Operators_At_Risk": count,
            "Training_Cost": training_cost,
            "Replacement_Cost": replacement_cost,
            "Net_Savings_By_Training": savings,
            "Recommendation": "RETRAIN" if savings > 0 else "REPLACE"
        }

# Instance for easy import
fiscal_engine = FiscalImpactEngine()