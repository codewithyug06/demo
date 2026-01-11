import pandas as pd
import glob
import os
import numpy as np
from config.settings import config

class IngestionEngine:
    """
    Enterprise ETL Layer.
    Auto-detects massive datasets, normalizes headers, and handles missing geo-data.
    """
    def __init__(self):
        self.raw_path = config.DATA_DIR

    def load_master_index(self):
        all_files = glob.glob(str(self.raw_path / "*.csv"))
        target_files = [f for f in all_files if "trai" not in f and "census" not in f]
        
        if not target_files:
            return pd.DataFrame()
            
        df_list = []
        for f in target_files:
            try:
                # Skip empty or tiny files
                if os.path.getsize(f) < 100: 
                    continue

                # 1. READ AS STRING (Type Safety)
                # Prevent ArrowTypeError by forcing text columns to string immediately
                temp = pd.read_csv(f, dtype={'state': str, 'district': str, 'sub_district': str, 'pincode': str})
                
                # 2. NORMALIZE HEADERS
                temp.columns = [c.lower().strip().replace(" ", "_") for c in temp.columns]
                
                # 3. NUMERIC CONVERSION
                for col in temp.columns:
                    if 'age' in col or 'count' in col or 'total' in col or 'activity' in col:
                        temp[col] = pd.to_numeric(temp[col], errors='coerce').fillna(0)

                # 4. TOTAL ACTIVITY CALCULATION
                if 'total_activity' not in temp.columns:
                    num_cols = temp.select_dtypes(include='number').columns
                    sum_cols = [c for c in num_cols if 'age' in c or 'count' in c]
                    if sum_cols:
                        temp['total_activity'] = temp[sum_cols].sum(axis=1)
                    else:
                        temp['total_activity'] = 0
                
                # 5. DATE PARSING
                if 'date' in temp.columns:
                    temp['date'] = pd.to_datetime(temp['date'], dayfirst=True, errors='coerce')
                
                # 6. SANITIZATION & BLACKLIST (CRITICAL FIX)
                # Removes rows where 'state' or 'district' is "10000", "0", or numeric noise
                for col in ['state', 'district']:
                    if col in temp.columns:
                        # Fill NaNs
                        temp[col] = temp[col].fillna('Unknown')
                        
                        # Remove explicit bad values requested by user
                        blacklist = ["10000", "0", "1", "nan", "null"]
                        temp = temp[~temp[col].astype(str).isin(blacklist)]
                        
                        # Remove any row where State/District is purely numeric (e.g. "12345")
                        temp = temp[~temp[col].astype(str).str.isnumeric()]

                df_list.append(temp)
            except Exception as e:
                # Log but don't crash
                print(f"Skipping corrupt asset: {f} ({e})")
                
        if not df_list: return pd.DataFrame()
        
        master = pd.concat(df_list, ignore_index=True)
        
        # Filter out rows with bad dates if date column exists
        if 'date' in master.columns:
            master = master.dropna(subset=['date'])
        
        # 7. GEO-SIMULATION
        if 'lat' not in master.columns:
            master['lat'] = np.random.uniform(8.4, 37.6, len(master))
        if 'lon' not in master.columns:
            master['lon'] = np.random.uniform(68.7, 97.2, len(master))
            
        return master

    def load_telecom_index(self):
        files = glob.glob(str(self.raw_path / "*trai*.csv"))
        if files:
            try:
                return pd.read_csv(files[0])
            except: return pd.DataFrame()
        return pd.DataFrame()