import pandas as pd
import glob
import os
from config.settings import config

class DataLoader:
    """
    Enterprise Data Ingestion Layer.
    Auto-detects Aadhaar, Census, and Telecom datasets.
    """
    
    def __init__(self):
        self.raw_path = config.DATA_DIR

    def load_aadhaar_master(self):
        # Recursive search for State CSVs (avoiding census/telecom)
        all_files = glob.glob(str(self.raw_path / "**/*.csv"), recursive=True)
        aadhaar_files = [f for f in all_files if "census" not in f.lower() and "trai" not in f.lower()]
        
        if not aadhaar_files:
            return pd.DataFrame()
            
        df_list = []
        for f in aadhaar_files:
            try:
                temp = pd.read_csv(f)
                # Normalize Headers
                temp.columns = [c.lower().strip() for c in temp.columns]
                
                # Auto-Calc Total Activity if missing
                if 'total_activity' not in temp.columns:
                    numeric_cols = temp.select_dtypes(include='number').columns
                    # Sum columns that likely represent counts
                    cnt_cols = [c for c in numeric_cols if 'age' in c or 'count' in c]
                    if cnt_cols:
                        temp['total_activity'] = temp[cnt_cols].sum(axis=1)
                    else:
                        temp['total_activity'] = 0
                
                df_list.append(temp)
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                
        if not df_list: return pd.DataFrame()
        
        master = pd.concat(df_list, ignore_index=True)
        if 'date' in master.columns:
            master['date'] = pd.to_datetime(master['date'], errors='coerce')
        return master.dropna(subset=['date']) if 'date' in master.columns else master

    def load_telecom_data(self):
        # Look for TRAI/Telecom data
        files = glob.glob(str(self.raw_path / "*trai*.csv"))
        if files:
            try:
                df = pd.read_csv(files[0])
                # Standardization logic would go here
                return df
            except: return pd.DataFrame()
        return pd.DataFrame()