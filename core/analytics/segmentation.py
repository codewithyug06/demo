import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

class SegmentationEngine:
    """
    BEHAVIORAL SEGMENTATION ENGINE v9.9 (GOD MODE)
    
    CAPABILITIES:
    1. K-Means (Centroid-based Partitioning)
    2. DBSCAN (Density-based Spatial Clustering)
    3. Hierarchical (Agglomerative Connectivity)
    4. Service Saturation Indexing
    5. Policy Action Mapping (Automated Directives)
    6. Vulnerable Group Micro-Routing (Elderly/Divyang)
    """
    
    @staticmethod
    def segment_districts(df, n_clusters=4):
        """
        Segments districts into clusters using K-Means (Legacy/Standard).
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
        
        # FIX: Fill NaNs ONLY in numeric columns to avoid Categorical type errors
        numeric_cols = ['total_volume', 'volatility', 'daily_avg']
        district_stats[numeric_cols] = district_stats[numeric_cols].fillna(0)
        
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

    # ==========================================================================
    # NEW V9.7 FEATURE: DENSITY-BASED SPATIAL CLUSTERING (DBSCAN)
    # ==========================================================================
    @staticmethod
    def perform_density_clustering(df, eps=0.5, min_samples=5):
        """
        Uses DBSCAN to find arbitrary shaped clusters and outliers (Noise).
        Excellent for identifying 'Organic' growth patterns vs. 'Forced' administrative groupings.
        
        Returns: DataFrame with 'density_label' (Core, Border, or Noise).
        """
        if df.empty or 'total_activity' not in df.columns: return pd.DataFrame()
        
        # Aggregation
        stats = df.groupby(['district']).agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        # FIX: Fill NaNs in numeric columns only
        cols_to_fix = ['total_activity', 'lat', 'lon']
        stats[cols_to_fix] = stats[cols_to_fix].fillna(0)
        
        if len(stats) < 10: return stats # Not enough for density scan
        
        # Scale features (Critical for DBSCAN)
        # Using Lat/Lon + Activity allows us to find "Geographic Hotspots"
        features = stats[['lat', 'lon', 'total_activity']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        try:
            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
            stats['dbscan_cluster'] = db.labels_
            
            # Labeling Logic
            def label_density(c):
                if c == -1: return "🚨 SPATIAL ANOMALY (Noise)"
                return f"CORE CLUSTER {c+1}"
                
            stats['density_group'] = stats['dbscan_cluster'].apply(label_density)
            
            # Calculate Silhouette Score to measure cluster quality (Internal Metric)
            # Useful for the 'Strategist Agent' to know if the grouping is tight or loose
            if len(set(db.labels_)) > 1:
                score = silhouette_score(X_scaled, db.labels_)
                stats['cluster_quality_score'] = score
            else:
                stats['cluster_quality_score'] = 0.0
                
            return stats.sort_values('total_activity', ascending=False)
            
        except Exception as e:
            print(f"DBSCAN Error: {e}")
            return stats

    # ==========================================================================
    # NEW V9.7 FEATURE: SERVICE SATURATION INDEXING
    # ==========================================================================
    @staticmethod
    def calculate_saturation_index(df):
        """
        Mathematically determines if a district is 'Saturated' (High Volume / Low Volatility)
        or 'Volatile' (Spiky Volume).
        
        Formula: Saturation = (Normalized_Volume * 0.7) + ((1 - Normalized_Volatility) * 0.3)
        High Score (>0.8) = Mature Market (Service Saturation).
        Low Score (<0.3) = Underserved / Chaotic Market.
        """
        if df.empty: return pd.DataFrame()
        
        # 1. Prep Data
        stats = df.groupby('district').agg({
            'total_activity': ['sum', 'std']
        }).reset_index()
        stats.columns = ['district', 'vol', 'std']
        
        # FIX: Numeric Fill
        stats[['vol', 'std']] = stats[['vol', 'std']].fillna(0)
        
        # 2. Normalization (MinMax to 0-1 range)
        scaler = MinMaxScaler()
        stats[['vol_norm', 'std_norm']] = scaler.fit_transform(stats[['vol', 'std']])
        
        # 3. Compute Index
        # We invert volatility because Low Volatility is a sign of Saturation/Stability
        stats['saturation_score'] = (stats['vol_norm'] * 0.7) + ((1 - stats['std_norm']) * 0.3)
        
        # 4. Strategic Categorization
        def categorize(x):
            if x > 0.75: return "🟢 FULLY SATURATED (Optimal)"
            elif x > 0.4: return "🟡 DEVELOPING (Growth Phase)"
            else: return "🔴 UNDERSERVED (Infrastructure Gap)"
            
        stats['market_status'] = stats['saturation_score'].apply(categorize)
        
        return stats.sort_values('saturation_score', ascending=False)

    # ==========================================================================
    # NEW V9.7 FEATURE: HIERARCHICAL AFFINITY GROUPING
    # ==========================================================================
    @staticmethod
    def perform_hierarchical_clustering(df, n_clusters=3):
        """
        Uses Agglomerative Clustering to build a 'Family Tree' of districts.
        Finds districts that are 'siblings' in terms of behavior, even if geographically distant.
        Example: Finding that 'Pune' behaves like 'Bangalore' despite being in different states.
        """
        if df.empty: return pd.DataFrame()
        
        stats = df.groupby('district').agg({
            'total_activity': 'mean',
            'total_activity': 'max' # Peak load
        }).reset_index()
        
        if len(stats) < n_clusters: return stats
        
        # Normalize
        features = stats.iloc[:, 1:] # Skip district name
        X_scaled = StandardScaler().fit_transform(features)
        
        try:
            # Ward linkage minimizes variance within clusters
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            stats['affinity_group'] = hc.fit_predict(X_scaled)
            
            return stats
        except:
            return stats

    # ==========================================================================
    # NEW V9.8: AUTOMATED POLICY MAPPING & EMERGING HOTSPOTS
    # ==========================================================================
    @staticmethod
    def generate_policy_labels(cluster_label):
        """
        Translates Cluster Labels (e.g., 'High-Velocity Hub') into 
        Actionable Administrative Directives.
        """
        directives = {
            "🔥 High-Velocity Hub": "ACTION: Deploy 5 extra Static Enrolment Kits. Enable High-Bandwidth Sync.",
            "💤 Low-Activity Zone": "ACTION: Initiate Awareness Camp. Deploy 1 Mobile Van for weekly rounds.",
            "⚖️ Steady-State": "ACTION: Maintain status quo. Quarterly audit recommended.",
            "🚨 SPATIAL ANOMALY (Noise)": "ACTION: URGENT FORENSIC AUDIT. Check for data entry fraud."
        }
        return directives.get(cluster_label, "ACTION: Monitor")

    @staticmethod
    def detect_emerging_hotspots(df):
        """
        Identifies 'Sleeping Giants' - districts with low current volume but 
        extremely high recent acceleration (Volatility).
        """
        if df.empty: return pd.DataFrame()
        
        stats = df.groupby('district').agg({
            'total_activity': ['sum', 'std']
        }).reset_index()
        stats.columns = ['district', 'vol', 'std']
        
        # Emerging = Low Volume (< 25th percentile) AND High Volatility (> 75th percentile)
        q25_vol = stats['vol'].quantile(0.25)
        q75_std = stats['std'].quantile(0.75)
        
        hotspots = stats[(stats['vol'] < q25_vol) & (stats['std'] > q75_std)].copy()
        hotspots['status'] = "🚀 EMERGING HOTSPOT"
        
        return hotspots

    # ==========================================================================
    # NEW V9.9: VULNERABLE GROUP MICRO-ROUTING (SOCIAL IMPACT)
    # ==========================================================================
    @staticmethod
    def optimize_doorstep_service_routes(df, target_group="elderly"):
        """
        Identifies micro-clusters of vulnerable citizens (Age > 80 or Disability)
        and calculates optimal service routes.
        
        Returns: DataFrame with Cluster Centroids for 'Doorstep Service Agents'.
        """
        if df.empty or 'age' not in df.columns or 'lat' not in df.columns:
            return pd.DataFrame()
            
        # Filter for Vulnerable Population
        # Assuming 'disability_flag' exists, else use Age > 80 proxy
        if target_group == "elderly":
            vulnerable = df[df['age'] >= 80].copy()
        else:
            vulnerable = df.copy() # Fallback
            
        if len(vulnerable) < 5: return pd.DataFrame()
        
        # Use K-Means to find service centroids
        coords = vulnerable[['lat', 'lon']].values
        n_agents = max(1, len(vulnerable) // 50) # 1 Agent per 50 elderly citizens
        
        kmeans = KMeans(n_clusters=n_agents, random_state=42)
        vulnerable['service_cluster'] = kmeans.fit_predict(coords)
        
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['lat', 'lon'])
        centroids['service_id'] = [f"DOORSTEP-UNIT-{i+1}" for i in range(len(centroids))]
        centroids['priority'] = "HIGH (Vulnerable Group)"
        
        return centroids
    
    @staticmethod
    def calculate_cluster_roi(df):
        """
        NEW V9.9: Helper to calculate potential ROI for a cluster.
        Used by the Fiscal Optimizer Agent.
        """
        if df.empty or 'saturation_score' not in df.columns: return pd.DataFrame()
        
        # Lower saturation = Higher Potential ROI for intervention
        # ROI proxy = (1 - Saturation) * Population_Density_Proxy (Volume)
        df['potential_roi_index'] = (1 - df['saturation_score']) * df['vol_norm']
        
        return df.sort_values('potential_roi_index', ascending=False)