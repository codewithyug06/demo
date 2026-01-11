import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MigrationAnalyzer:
    def __init__(self, aadhaar_df, census_df):
        self.aadhaar = aadhaar_df
        self.census = census_df

    def analyze_patterns(self):
        """
        1. Calculates Saturation.
        2. Uses K-Means Clustering to classify districts into:
           - Cluster 0: Low Activity (Stable Rural)
           - Cluster 1: High Outflow (Source)
           - Cluster 2: High Inflow (Destination/Metro)
        """
        # Group & Merge (Same as before)
        aadhaar_grouped = self.aadhaar.groupby(['state', 'district'])['total_activity'].sum().reset_index()
        aadhaar_grouped['district'] = aadhaar_grouped['district'].str.strip().str.title()
        
        merged = pd.merge(aadhaar_grouped, self.census, 
                          left_on='district', right_on='District', how='inner')
        
        # Feature Engineering
        merged['saturation_index'] = (merged['total_activity'] / merged['Projected_Pop_2025']) * 100
        
        # --- ADVANCED: UNSUPERVISED CLUSTERING ---
        scaler = StandardScaler()
        X = merged[['saturation_index', 'total_activity']]
        X_scaled = scaler.fit_transform(X)
        
        # Force 3 Clusters (Low, Medium, High intensity)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        merged['cluster_id'] = kmeans.fit_predict(X_scaled)
        
        # Label Clusters automatically based on mean saturation
        cluster_means = merged.groupby('cluster_id')['saturation_index'].mean().sort_values()
        label_map = {
            cluster_means.index[0]: 'Stable/Low-Mobility',
            cluster_means.index[1]: 'Active Transit Zone',
            cluster_means.index[2]: 'High-Migration Hub'
        }
        merged['classification'] = merged['cluster_id'].map(label_map)
        
        return merged[['state', 'district', 'classification', 'saturation_index']]