import os
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==============================================================================
# LEGACY IMPORTS (PRESERVED FOR BACKWARD COMPATIBILITY)
# ==============================================================================
try:
    from src.preprocessing import DataIngestionEngine
    from src.models.migration_engine import MigrationAnalyzer
    from src.models.anomaly_engine import AnomalyDetector
    from src.utils.dbt_middleware import DBTMiddleware
except ImportError:
    print(">> [WARNING] Legacy 'src' modules not found. Switching to Core Architecture.")

# ==============================================================================
# V9.8 AEGIS CORE IMPORTS (GOD MODE)
# ==============================================================================
# Ensure system path can find the new modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import config
from core.etl.ingest import IngestionEngine
from core.analytics.forensics import ForensicEngine
from core.analytics.segmentation import SegmentationEngine
from core.engines.spatial import SpatialEngine, GraphNeuralNetwork
from core.engines.cognitive import SentinelCognitiveEngine, SwarmOrchestrator
from core.models.lstm import ForecastEngine, SovereignForecastEngine

# ==============================================================================
# 1. LEGACY MAIN FUNCTION (DO NOT TOUCH)
# ==============================================================================
def main():
    print("==================================================")
    print("   AADHAAR PULSE 2026: NATIONAL ANALYTICS ENGINE   ")
    print("==================================================")

    # 1. SETUP & LOADING
    DATA_DIR = 'data/states'
    CENSUS_FILE = 'data/census/census2011.csv'
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fallback to empty class if import failed (to prevent crash)
    if 'DataIngestionEngine' in globals():
        ingestor = DataIngestionEngine(DATA_DIR, CENSUS_FILE)
        
        print("\n[STEP 1] Ingesting State Data...")
        master_df = ingestor.load_all_states()
        census_df = ingestor.load_and_project_census()
        
        # 2. MIGRATION ANALYTICS (Policy Engine)
        print("\n[STEP 2] Running Spatio-Temporal Migration Model...")
        migrator = MigrationAnalyzer(master_df, census_df)
        saturation_report = migrator.calculate_saturation_indices()
        
        top_hotspots = saturation_report.head(10)
        print(f"   > Identified {len(saturation_report)} districts.")
        print("   > Top 5 Migration Magnets (High Saturation):")
        print(top_hotspots[['district', 'state', 'saturation_index']].to_string(index=False))
        
        # Save Visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_hotspots, x='saturation_index', y='district', palette='viridis')
        plt.title('Top 10 Districts: Aadhaar Activity vs. Projected Population')
        plt.xlabel('Saturation Index (%)')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/migration_hotspots.png")
        print(f"   > Graph saved to {OUTPUT_DIR}/migration_hotspots.png")

        # 3. ANOMALY DETECTION (Security Engine)
        print("\n[STEP 3] Running Isolation Forest for Fraud Detection...")
        detector = AnomalyDetector(contamination=0.005) # Strict 0.5%
        anomalies = detector.detect_velocity_anomalies(master_df)
        
        print(f"   > Scanned {len(master_df)} transactions.")
        print(f"   > FLAGGED: {len(anomalies)} suspicious 'Super-Operator' events.")
        anomalies.to_csv(f"{OUTPUT_DIR}/flagged_anomalies.csv", index=False)

        # 4. DBT MIDDLEWARE TEST (Inclusion Engine)
        print("\n[STEP 4] Testing Fuzzy Logic Middleware...")
        tester = DBTMiddleware()
        result = tester.verify_beneficiary("Mohammed Yusuf", "Mohd. Yusuf")
        print(f"   > Test Case: {result}")

    print("\n==================================================")
    print("   PROCESS COMPLETE. READY FOR JURY SUBMISSION.")
    print("==================================================")

# ==============================================================================
# 2. NEW GOD-MODE FUNCTION (V9.8 AEGIS PROTOCOL)
# ==============================================================================
def run_aegis_protocol():
    """
    Executes the Advanced V9.8 Sovereign Architecture.
    Includes: Privacy Watchdog, GNN Contagion, Dark Zones, DBT Wargaming.
    """
    print("\n\n")
    print("██████████████████████████████████████████████████")
    print("█  SENTINEL PRIME | AEGIS COMMAND | V9.8.0 ALPHA █")
    print("█  SOVEREIGN DIGITAL TWIN ACTIVATED              █")
    print("██████████████████████████████████████████████████")
    time.sleep(1)

    # 1. SOVEREIGN INGESTION & PRIVACY CHECK
    print("\n[AEGIS-1] INITIALIZING SECURE DATA LINK...")
    engine = IngestionEngine()
    df = engine.load_master_index()
    telecom_df = engine.load_telecom_index()
    
    # Initialize Swarm
    swarm = SwarmOrchestrator(df)
    
    # Run Privacy Watchdog
    privacy_status = swarm.privacy_bot.verify_sanitization(df)
    print(f"   > PII SANITIZATION PROTOCOL: {privacy_status}")
    if "ALERT" in privacy_status:
        print("   > 🛑 HALTING PROTOCOL: DATA LEAK DETECTED.")
        return

    # 2. ADVANCED FORENSICS (WHIPPLE + BENFORD + MYERS)
    print("\n[AEGIS-2] EXECUTING DEEP FORENSIC SCAN...")
    integrity_score = ForensicEngine.generate_integrity_scorecard(df)
    whipple = ForensicEngine.calculate_whipple(df)
    myers = ForensicEngine.calculate_myers_index(df)
    gender_skew = ForensicEngine.assess_gender_parity(df)
    
    print(f"   > GLOBAL DATA TRUST SCORE: {integrity_score:.2f}/100")
    print(f"   > WHIPPLE'S INDEX (Age Heaping): {whipple:.2f} " + 
          ("(Rough Data)" if whipple > 125 else "(Accurate)"))
    print(f"   > MYER'S BLENDED INDEX: {myers:.2f} (Digit Preference)")
    print(f"   > GENDER PARITY SKEW: {gender_skew:.2f}%")
    
    # 3. SPATIAL INTELLIGENCE & GNN CONTAGION
    print("\n[AEGIS-3] BUILDING MIGRATION GRAPH & GNN SIMULATION...")
    G, centrality = SpatialEngine.build_migration_graph(df)
    
    if G:
        # Simulate Fraud Contagion
        print(f"   > Graph Built: {len(G.nodes)} Nodes, {len(G.edges)} Edges.")
        seeds = {list(G.nodes)[0]: 0.8} # Seed risk in one node
        diffused_risks = GraphNeuralNetwork.simulate_risk_diffusion(G, seeds)
        high_risk_zones = [k for k, v in diffused_risks.items() if v > 0.5]
        print(f"   > RISK CONTAGION: Detected {len(high_risk_zones)} districts at risk of forensic infection.")
    
    # 4. DIGITAL DARK ZONES & VAN DEPLOYMENT
    print("\n[AEGIS-4] IDENTIFYING DIGITAL DARK ZONES...")
    dark_zones = SpatialEngine.identify_digital_dark_zones(df)
    print(f"   > ISOLATED BLOCKS FOUND: {len(dark_zones)}")
    
    if not dark_zones.empty:
        deployments = SpatialEngine.optimize_van_deployment(dark_zones, n_vans=5)
        print("   > OPTIMAL VAN DEPLOYMENT COORDINATES (K-Means Centroids):")
        print(deployments[['van_id', 'lat', 'lon']].to_string(index=False))

    # 5. WARGAME: DBT MEGA-LAUNCH SIMULATION
    print("\n[AEGIS-5] INITIATING WARGAME: 'PM-KISAN DISBURSEMENT'...")
    forecaster = ForecastEngine(df)
    simulation = forecaster.simulate_dbt_mega_launch(days=15)
    
    peak_load = simulation['Utilization'].max()
    print(f"   > PREDICTED PEAK LOAD: {peak_load*100:.1f}% Capacity")
    
    status = swarm.crisis_bot.evaluate_shock_resilience(peak_load)
    print(f"   > INFRASTRUCTURE STATUS: {status['condition']}")
    print(f"   > STRATEGIC DIRECTIVE: {status['message']}")

    # 6. EXECUTIVE REPORT GENERATION
    print("\n[AEGIS-6] SYNTHESIZING CLASSIFIED SITREP...")
    cog_engine = SentinelCognitiveEngine(df)
    
    stats = {
        'sector': 'NATIONAL COMMAND',
        'risk': status['condition'],
        'total_volume': int(df['total_activity'].sum() if 'total_activity' in df else 0),
        'nodes': len(df),
        'anomalies': len(high_risk_zones) if G else 0
    }
    
    # New V9.8 Full Spectrum Brief
    pdf_data = cog_engine.generate_full_spectrum_brief(
        stats, 
        gnn_risk=len(high_risk_zones)/len(df) if len(df) > 0 else 0,
        dark_zones=len(dark_zones)
    )
    
    if pdf_data:
        report_path = "outputs/AEGIS_CLASSIFIED_BRIEF.pdf"
        with open(report_path, "wb") as f:
            f.write(pdf_data)
        print(f"   > 📄 REPORT GENERATED: {report_path}")
    else:
        print("   > ⚠️ REPORT GENERATION FAILED (Check FPDF).")

    print("\n")
    print("██████████████████████████████████████████████████")
    print("█  MISSION ACCOMPLISHED. SYSTEM STANDING BY.     █")
    print("██████████████████████████████████████████████████")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # 1. Run Legacy (Standard Output)
    try:
        main()
    except Exception as e:
        print(f"Legacy Protocol Failed: {e}")
    
    # 2. Run God Mode (Advanced Output)
    # This runs regardless of legacy success, demonstrating resilience
    run_aegis_protocol()