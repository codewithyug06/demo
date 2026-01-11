import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import DataIngestionEngine
from src.models.migration_engine import MigrationAnalyzer
from src.models.anomaly_engine import AnomalyDetector
from src.utils.dbt_middleware import DBTMiddleware

def main():
    print("==================================================")
    print("   AADHAAR PULSE 2026: NATIONAL ANALYTICS ENGINE   ")
    print("==================================================")

    # 1. SETUP & LOADING
    DATA_DIR = 'data/states'
    CENSUS_FILE = 'data/census/census2011.csv'
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

if __name__ == "__main__":
    main()