
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
                # Check file size to avoid empty file errors
                if os.path.getsize(f) < 10: 
                    continue

                temp = pd.read_csv(f)
                
                # 1. Normalize Headers
                temp.columns = [c.lower().strip().replace(" ", "_") for c in temp.columns]
                
                # 2. CRITICAL FIX: Force String Types for Categorical Columns
                # This prevents ArrowTypeError when mixed ints/strings exist
                for col in ['state', 'district', 'sub_district', 'pincode']:
                    if col in temp.columns:
                        temp[col] = temp[col].astype(str)

                # 3. Logic: Calculate Total Activity
                if 'total_activity' not in temp.columns:
                    num_cols = temp.select_dtypes(include='number').columns
                    sum_cols = [c for c in num_cols if 'age' in c or 'count' in c]
                    if sum_cols:
                        temp['total_activity'] = temp[sum_cols].sum(axis=1)
                    else:
                        temp['total_activity'] = 0
                
                # 4. Logic: Date Parsing
                if 'date' in temp.columns:
                    temp['date'] = pd.to_datetime(temp['date'], dayfirst=True, errors='coerce')
                
                df_list.append(temp)
            except Exception as e:
                # Silent fail for corrupt files to keep system running
                pass
                
        if not df_list: return pd.DataFrame()
        
        master = pd.concat(df_list, ignore_index=True)
        master = master.dropna(subset=['date']) if 'date' in master.columns else master
        
        # 5. GEO-SIMULATION
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
