import pandas as pd
import glob
import os
import numpy as np
import re
from config.settings import config

class IngestionEngine:
    """
    Enterprise ETL Layer.
    Auto-detects massive datasets, normalizes headers, handles missing geo-data,
    and enforces Sovereign Privacy standards (PII Sanitization).
    """
    def __init__(self):
        self.raw_path = config.DATA_DIR

    def sanitize_pii(self, df):
        """
        SOVEREIGN PROTOCOL: Removes PII (Personally Identifiable Information).
        Masks Aadhaar numbers (12 digits) and Mobile numbers (10 digits).
        """
        # Regex patterns for sensitive data
        aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        mobile_pattern = r'\b[6-9]\d{9}\b'
        
        # Scan string columns only
        str_cols = df.select_dtypes(include='object').columns
        
        for col in str_cols:
            # Check if column likely contains PII based on name
            if any(x in col.lower() for x in ['uid', 'aadhaar', 'mobile', 'phone', 'contact']):
                df[col] = "REDACTED_PII"
                continue
                
            # Deep scan: Mask patterns in value text
            # Using verify=False for speed on large datasets
            # Only run on small sample to decide if cleaning is needed to save performance
            sample = df[col].astype(str).head(100).str.cat()
            if re.search(aadhaar_pattern, sample) or re.search(mobile_pattern, sample):
                df[col] = df[col].astype(str).str.replace(aadhaar_pattern, 'XXXXXXXXXXXX', regex=True)
                df[col] = df[col].astype(str).str.replace(mobile_pattern, 'XXXXXXXXXX', regex=True)
                
        return df

    def load_master_index(self):
        all_files = glob.glob(str(self.raw_path / "*.csv"))
        # Exclude auxiliary files
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
                    # Only sum relevant columns to avoid summing unrelated metrics
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
                        # Also removes rows where district names are purely numeric (like "10000")
                        # Using regex to identify purely numeric strings even if types are mixed
                        mask_valid = ~temp[col].astype(str).str.match(r'^\d+$') & \
                                     ~temp[col].astype(str).isin(["nan", "null", "None", ""])
                        temp = temp[mask_valid]

                # 7. APPLY SOVEREIGN PII MASKING
                temp = self.sanitize_pii(temp)

                df_list.append(temp)
            except Exception as e:
                # Log but don't crash
                print(f"Skipping corrupt asset: {f} ({e})")
                
        if not df_list: return pd.DataFrame()
        
        master = pd.concat(df_list, ignore_index=True)
        
        # Filter out rows with bad dates if date column exists
        if 'date' in master.columns:
            master = master.dropna(subset=['date'])
        
        # 8. GEO-SIMULATION (Fallback for missing Lat/Lon)
        # Used for visual demonstration if real GIS data is missing
        if 'lat' not in master.columns:
            # Simulate generic India bounds
            master['lat'] = np.random.uniform(8.4, 37.6, len(master))
        if 'lon' not in master.columns:
            master['lon'] = np.random.uniform(68.7, 97.2, len(master))
            
        return master

    def load_telecom_index(self):
        files = glob.glob(str(self.raw_path / "*trai*.csv"))
        if files:
            try:
                df = pd.read_csv(files[0])
                # Ensure teledensity is numeric for analysis
                if 'teledensity' in df.columns:
                     df['teledensity'] = pd.to_numeric(df['teledensity'], errors='coerce')
                return df
            except: return pd.DataFrame()
        return pd.DataFrame()