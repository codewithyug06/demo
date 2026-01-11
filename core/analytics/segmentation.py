import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SegmentationEngine:
    """
    BEHAVIORAL SEGMENTATION ENGINE v1.1 (Robust)
    Uses Unsupervised Learning (K-Means) to cluster districts based on their operational behavior.
    """
    
    @staticmethod
    def segment_districts(df, n_clusters=4):
        """
        Segments districts into clusters.
        SAFEGUARD: Ensures 'cluster_label' column always exists.
        """
        if df.empty or 'total_activity' not in df.columns:
            return pd.DataFrame()

        # Aggregate data by District
        district_stats = df.groupby(['state', 'district']).agg({
            'total_activity': ['sum', 'std', 'mean']
        }).reset_index()
        
        # Flatten columns
        district_stats.columns = ['state', 'district', 'total_volume', 'volatility', 'daily_avg']
        district_stats = district_stats.fillna(0)
        
        # --- FAIL-SAFE LOGIC ---
        # If not enough data for K-Means, assign a default label and return immediately
        if len(district_stats) < n_clusters:
            district_stats['cluster'] = 0
            district_stats['cluster_label'] = "⚠️ Insufficient Data for Clustering"
            return district_stats

        # Normalize Features
        scaler = StandardScaler()
        features = district_stats[['total_volume', 'volatility']]
        try:
            scaled_features = scaler.fit_transform(features)
            
            # K-Means Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            district_stats['cluster'] = kmeans.fit_predict(scaled_features)
            
            # Map Clusters to Human-Readable Labels
            cluster_map = {}
            for c in range(n_clusters):
                cluster_data = district_stats[district_stats['cluster'] == c]
                avg_vol = cluster_data['total_volume'].mean()
                
                # Dynamic Naming
                q75 = district_stats['total_volume'].quantile(0.75)
                q25 = district_stats['total_volume'].quantile(0.25)
                
                if avg_vol >= q75:
                    label = "🔥 High-Velocity Hub"
                elif avg_vol <= q25:
                    label = "💤 Low-Activity Zone"
                else:
                    label = "⚖️ Steady-State"
                cluster_map[c] = label
                
            district_stats['cluster_label'] = district_stats['cluster'].map(cluster_map)
            
        except Exception as e:
            # Fallback if Sklearn fails
            district_stats['cluster_label'] = "Error in Clustering"
            print(f"Clustering Error: {e}")
        
        return district_stats