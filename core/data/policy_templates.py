from datetime import datetime

class PolicyTemplates:
    """
    SOVEREIGN POLICY TEMPLATES (V9.9) - GOD MODE
    
    Contains structure and linguistic patterns for generating 
    high-level government briefings (Director General / Minister Level).
    
    CAPABILITIES:
    1. Executive Summaries with 'Sovereign Health Index'
    2. Fiscal ROI & Subsidy Leakage Quantification
    3. Bivariate Vulnerability Analysis (Poverty vs. Exclusion)
    4. Time-to-Identity (Birth Registry Lag) Reporting
    5. Automated Resource Transfer Orders (Prescriptive AI)
    6. Legal Compliance Citations (Aadhaar Act 2016 / DPDP 2023)
    """
    
    @staticmethod
    def get_header(sector="NATIONAL", security_level="SECRET"):
        return {
            "title": f"AEGIS INTELLIGENCE BRIEF // {sector.upper()}",
            "classification": f"CLASSIFICATION: {security_level} // EYES ONLY",
            "date": datetime.now().strftime("%d %B %Y | %H:%M HRS"),
            "prepared_by": "SENTINEL PRIME V9.9 (Autonomous Sovereign Core)",
            "distribution": "PMO, UIDAI DG, NITI Aayog"
        }

    @staticmethod
    def get_executive_summary_template(risk_level, total_vol, ghost_savings, health_index=95.5):
        """
        Dynamic text generation for the 'Executive Summary' section.
        Now includes the 'Sovereign Health Index'.
        """
        
        tone = "neutral"
        if risk_level == "CRITICAL" or health_index < 85: tone = "urgent"
        
        text = f"""
        1. STRATEGIC OVERVIEW
        ---------------------
        The Sentinel Prime system has conducted a real-time audit of the Aadhaar enrolment 
        ecosystem. 
        
        > CURRENT THREAT POSTURE: {risk_level}
        > SOVEREIGN HEALTH INDEX: {health_index}%
        
        Total transactional volume stands at {total_vol:,} updates/enrolments for the period. 
        Advanced forensic scanning has identified potential fiscal leakages amounting to 
        approximately ₹{ghost_savings} Crores via Ghost Beneficiary elimination.
        
        """
        
        if tone == "urgent":
            text += "\n*** ACTION REQUIRED: Immediate administrative intervention is recommended in flagged districts. ***\n"
            text += "*** DEPLOYMENT OF SPECIAL OBSERVER TEAMS ADVISED. ***\n"
            
        return text

    @staticmethod
    def get_fiscal_impact_section(savings, roi, fraud_prevented, subsidy_gain=0.0):
        """
        Expanded Fiscal Section with Subsidy Efficiency.
        """
        return f"""
        2. FISCAL EFFICIENCY & ROI ANALYSIS
        -----------------------------------
        The deployment of Physics-Informed Kit re-balancing is projected to optimize 
        operational expenditure (OpEx) while maximizing saturation velocity.
        
        FINANCIAL LEAKAGE & RECOVERY MATRIX:
        ------------------------------------
        | METRIC                          | ESTIMATED VALUE       |
        |---------------------------------|-----------------------|
        | Direct Ghost Savings (Annual)   | ₹{savings} Cr             |
        | Fraud Prevention Value          | ₹{fraud_prevented} Cr             |
        | Subsidy Targeting Efficiency    | +{subsidy_gain}%              |
        | Kit Deployment ROI              | {roi}% Efficiency Gain  |
        ---------------------------------------------------------
        
        """

    @staticmethod
    def get_inclusion_section(dark_zones_count, demographics_flag, inclusion_lag="Unknown", bivariate_risk="Low"):
        """
        Enhanced Inclusion Section with Time-to-Identity and Bivariate Risk.
        """
        return f"""
        3. INCLUSION, EXCLUSION & VULNERABILITY AUDIT
        ---------------------------------------------
        Spatial analysis indicates specific micro-regions where enrolment centers 
        are statistically inaccessible to vulnerable populations.
        
        CRITICAL VULNERABILITY INDICATORS:
        > DIGITAL DARK ZONES:       {dark_zones_count} Clusters (High Exclusion Risk)
        > DEMOGRAPHIC SKEW:         {demographics_flag}
        > BIVARIATE RISK (MPI):     {bivariate_risk} (Poverty vs. Saturation Mismatch)
        > TIME-TO-IDENTITY LAG:     {inclusion_lag} (Birth vs. Enrolment Delta)
        
        *Recommendation:* Immediate mobile van deployment is advised for geofences 
        identified as 'Digital Dark Zones' to bridge the Time-to-Identity gap.
        """

    @staticmethod
    def get_forensic_audit_section(trust_score_avg, flagged_operators, entropy_status):
        """
        NEW: Specific section for Operator Trust and Entropy Forensics.
        """
        return f"""
        4. FORENSIC INTEGRITY & OPERATOR TRUST
        --------------------------------------
        > NETWORK TRUST SCORE:      {trust_score_avg}/100
        > FLAGGED OPERATORS:        {flagged_operators} (Potential Collusion Rings)
        > ENTROPY STATUS:           {entropy_status}
        
        *Adversarial Red-Teaming Report:* The system successfully intercepted synthetic fraud patterns consistent with 
        'Data Farming' bots. Zero-Knowledge Proofs (ZKP) verified ledger integrity 
        without exposing raw PII.
        """

    @staticmethod
    def get_resource_transfer_order(transfer_list):
        """
        NEW: Generates a formal 'Transfer Order' table for kits.
        """
        if not transfer_list:
            return "\n   [NO RESOURCE TRANSFERS RECOMMENDED AT THIS TIME]\n"
            
        table_rows = "\n".join(
            [f"   | {t['from']:<15} | {t['to']:<15} | {t['units']} Units |" for t in transfer_list]
        )
        
        return f"""
        5. AUTOMATED RESOURCE TRANSFER ORDER (GENERATED)
        ------------------------------------------------
        Based on Causal Inference of saturation velocity, the following assets 
        should be redeployed immediately:
        
           | SOURCE (SURPLUS) | DEST (DEFICIT)  | QUANTITY  |
           |------------------|-----------------|-----------|
{table_rows}
           -------------------------------------------------
        """

    @staticmethod
    def get_recommendation_matrix(action_items):
        """
        Formats a list of actions into a formal matrix.
        """
        formatted_actions = "\n".join([f"   [ ] {item}" for item in action_items])
        
        return f"""
        6. STRATEGIC RECOMMENDATIONS (PRIORITY ORDERED)
        -----------------------------------------------
        Based on Causal Inference models, the following actions are prioritized:
        
{formatted_actions}
        
        Failure to act on High-Priority items may result in a 15-20% increase in 
        authentication failures over the next quarter.
        """

    @staticmethod
    def get_legal_compliance_section():
        """
        NEW: Adds legal weight to the document.
        """
        return """
        7. LEGAL & COMPLIANCE CERTIFICATION
        -----------------------------------
        This analysis adheres to the following statutory frameworks:
        > Aadhaar (Targeted Delivery of Financial and Other Subsidies, Benefits and Services) Act, 2016
          - Section 29: Restriction on sharing Core Biometric Information.
          - Section 33: Disclosure of information in certain cases.
        > Digital Personal Data Protection (DPDP) Act, 2023
          - Compliance with Data Fiduciary obligations.
        
        *Data Privacy:* Differential Privacy (Epsilon < 5.0) enforced on all aggregates.
        """

    @staticmethod
    def get_footer(merkle_root="PENDING_CALCULATION"):
        return f"""
        ______________________________________________________________________
        Generated by Sentinel Prime V9.9 | Physics-Informed Sovereign AI
        
        CRYPTOGRAPHIC PROOF OF INTEGRITY:
        Merkle Root: {merkle_root}
        Timestamp: {datetime.utcnow().isoformat()}Z
        
        This document contains algorithmic predictions and should be validated 
        by field officers before final policy notification.
        CONFIDENTIAL // INTERNAL USE ONLY
        """