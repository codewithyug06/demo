import pandas as pd
import glob
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestionEngine:
    def __init__(self, states_dir, census_file):
        self.states_dir = states_dir
        self.census_file = census_file

    def load_all_states(self):
        """
        Iterates through ALL files in data/states/ regardless of filename.
        Validates schema before merging to ensure robustness.
        """
        all_files = glob.glob(os.path.join(self.states_dir, "*.csv"))
        df_list = []
        
        required_cols = {'state', 'district', 'pincode'}
        
        logging.info(f"Found {len(all_files)} files in {self.states_dir}. Beginning ingestion...")

        for filename in all_files:
            try:
                # Read only header first to check validity
                df_iter = pd.read_csv(filename, nrows=1)
                if not required_cols.issubset(df_iter.columns):
                    logging.warning(f"Skipping {filename}: Missing required columns.")
                    continue
                
                # Load full file if valid
                df = pd.read_csv(filename)
                
                # Standardize column names (handle case sensitivity)
                df.columns = [c.lower() for c in df.columns]
                
                # Create 'total_activity' if not exists (handling Enrolment vs Update files)
                if 'age_18_greater' in df.columns:
                    df['total_activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
                elif 'demo_age_17_' in df.columns:
                     df['total_activity'] = df['demo_age_5_17'] + df['demo_age_17_']
                
                df_list.append(df)
            except Exception as e:
                logging.error(f"Error reading {filename}: {str(e)}")

        if not df_list:
            raise ValueError("No valid Aadhaar data files found!")

        master_df = pd.concat(df_list, ignore_index=True)
        master_df['date'] = pd.to_datetime(master_df['date'], dayfirst=True, errors='coerce')
        
        logging.info(f"Successfully aggregated {len(master_df)} records from {len(df_list)} files.")
        return master_df

    def load_and_project_census(self):
        """
        Loads Census 2011 and projects population to 2025 using Growth Rate.
        Formula: Pop_2025 = Pop_2011 * (1 + Growth_Rate)^1.4
        """
        try:
            df = pd.read_csv(self.census_file)
            
            # Cleaning Data
            df['Growth'] = df['Growth'].astype(str).str.replace(' %', '').str.replace(',', '').astype(float) / 100
            df['Population'] = df['Population'].astype(str).str.replace(',', '').astype(float)
            
            # Projection Logic (1.4 decades from 2011 to 2025)
            df['Projected_Pop_2025'] = df['Population'] * ((1 + df['Growth']) ** 1.4)
            
            # Standardization for Joining
            df['District'] = df['District'].str.strip().str.title()
            df['State'] = df['State'].str.strip().str.title()
            
            return df[['State', 'District', 'Projected_Pop_2025']]
        except Exception as e:
            logging.error(f"Census Data Error: {e}")
            return pd.DataFrame()