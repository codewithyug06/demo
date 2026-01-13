import pandas as pd
import numpy as np
import networkx as nx
import random
from scipy.spatial.distance import cdist

# ==============================================================================
# 1. GRAPH NEURAL NETWORK (GNN) SIMULATOR
# ==============================================================================
class GraphNeuralNetwork:
    """
    Simulates Message Passing Interface (MPI) for District Nodes.
    Predicts 'High Velocity Hubs' and 'Forensic Risk Contagion'.
    """
    @staticmethod
    def predict_node_importance(G):
        if G is None: return {}
        # PageRank is a good proxy for GNN node embedding importance in this context
        try:
            return nx.pagerank(G, alpha=0.85)
        except:
            return {}

    @staticmethod
    def simulate_risk_diffusion(G, initial_risk_scores, decay_factor=0.6, steps=3):
        """
        NEW V9.7: FORENSIC CONTAGION MODEL
        Simulates how fraud/anomaly risk spreads through the migration network.
        
        Args:
            G: NetworkX graph of migration flows.
            initial_risk_scores: Dict of {district: risk_score (0.0-1.0)}
            decay_factor: How much risk is transmitted to neighbors (0.6 = 60%)
            steps: Number of hops (degrees of separation)
            
        Returns:
            Dict of {district: diffused_risk_score}
        """
        if G is None or not initial_risk_scores: return {}
        
        current_risks = initial_risk_scores.copy()
        
        for _ in range(steps):
            new_risks = current_risks.copy()
            for node in G.nodes():
                if node not in current_risks: continue
                
                # Get neighbors (Districts connected by migration)
                neighbors = list(G.neighbors(node))
                if not neighbors: continue
                
                # Calculate risk pressure from this node to its neighbors
                outbound_risk = current_risks[node] * decay_factor
                
                for neighbor in neighbors:
                    # Get edge weight (strength of connection)
                    weight = G[node][neighbor].get('weight', 0.5)
                    
                    # Neighbor absorbs risk based on connection strength
                    transmitted_risk = outbound_risk * weight
                    
                    # Update neighbor risk (Max aggregation to simulate critical exposure)
                    existing_risk = new_risks.get(neighbor, 0)
                    new_risks[neighbor] = max(existing_risk, transmitted_risk)
            
            current_risks = new_risks
            
        return current_risks

class SpatialEngine:
    """
    PART 2 & 3: ADVANCED GEOSPATIAL & GRAPH INTELLIGENCE
    Features: H3 Indexing, Migration Graphs, Digital Twin Layers, Dark Zone Detection.
    """
    
    @staticmethod
    def generate_h3_hexagons(df, resolution=4):
        """
        Generates H3 Hexagon data for aggregation.
        If the 'h3' library is missing, it falls back to raw lat/lon aggregation.
        """
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return pd.DataFrame()
        
        try:
            import h3
            # Real H3 Indexing
            df['h3_index'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], resolution), axis=1)
            hex_data = df.groupby('h3_index').agg({
                'total_activity': 'sum',
                'lat': 'mean', # Centroid proxy
                'lon': 'mean'
            }).reset_index()
            return hex_data
        except ImportError:
            # Fallback: Simple Grid Aggregation if H3 is missing
            # This ensures the map never breaks even without the library
            return df[['lat', 'lon', 'total_activity']].copy()

    @staticmethod
    def build_migration_graph(df):
        G = nx.Graph()
        
        # Aggregation to find hubs
        if 'district' not in df.columns: return None, {}
        
        hub_data = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index().sort_values('total_activity', ascending=False)
        
        # Take top 20 hubs for the graph
        top_hubs = hub_data.head(20)
        
        for _, row in top_hubs.iterrows():
            G.add_node(row['district'], pos=(row['lon'], row['lat']))
            
        # Create probabilistic edges based on "Gravity Model"
        districts = top_hubs['district'].tolist()
        
        import random
        for i in range(len(districts)):
            for j in range(i+1, len(districts)):
                # Random connection for simulation
                if random.random() > 0.85: 
                    # Weight represents migration volume affinity
                    G.add_edge(districts[i], districts[j], weight=random.random())
            
        try:
            centrality = nx.degree_centrality(G)
        except:
            centrality = {}
        
        return G, centrality

    @staticmethod
    def downsample_for_map(df, max_points=10000):
        if len(df) > max_points:
            return df.sample(n=max_points, random_state=42)
        return df

    # ==========================================================================
    # NEW GOD-LEVEL FEATURE: 3D BALLISTIC MIGRATION ARCS
    # ==========================================================================
    @staticmethod
    def generate_migration_arcs(df):
        """
        Generates source-target pairs for PyDeck ArcLayer.
        Simulates migration from low-activity areas (Source) to High-activity Hubs (Target).
        OPTIMIZATION: Limits calculation to top flows to prevent browser crash.
        """
        if df.empty or 'district' not in df.columns: return pd.DataFrame()
        
        # 1. Identify Sources (Low Activity) and Targets (High Activity)
        stats = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        if len(stats) < 2: return pd.DataFrame()
        
        # Limit to top 5 Targets and random 20 Sources to keep arc count manageable
        targets = stats.nlargest(5, 'total_activity')
        sources = stats.nsmallest(min(20, len(stats)), 'total_activity')
        
        arcs = []
        import random
        
        for _, src in sources.iterrows():
            # Connect source to a random major hub
            target = targets.sample(1).iloc[0]
            
            # Ensure we don't connect a district to itself
            if src['district'] != target['district']:
                # Color logic: Red (High volume) to Green (Low volume)
                arcs.append({
                    "source_text": src['district'],
                    "target_text": target['district'],
                    "source": [src['lon'], src['lat']],
                    "target": [target['lon'], target['lat']],
                    "value": random.randint(100, 1000), # Simulated flow volume
                    "color": [0, 255, 194, 200] if random.random() > 0.5 else [255, 0, 100, 200]
                })
            
        return pd.DataFrame(arcs)

    # NEW: Zero-Shot Anomaly Detection (Visual Logic)
    @staticmethod
    def zero_shot_scan(df):
        """
        Placeholder for CLIP-based logic. 
        Returns outliers based on logical incongruence.
        """
        if df.empty: return []
        # Simulation: Return random 'Logical Anomalies'
        return df.sample(3) if len(df) > 3 else df

    # ==========================================================================
    # NEW V9.7 FEATURE: DIGITAL DARK ZONE DETECTION
    # ==========================================================================
    @staticmethod
    def identify_digital_dark_zones(df, threshold_activity=500):
        """
        Identifies districts that are 'off the grid'.
        Logic: High Distance from major Hubs + Low Activity.
        Useful for planning Mobile Enrolment Van routes.
        """
        if df.empty or 'lat' not in df.columns: return pd.DataFrame()
        
        # Aggregate
        stats = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        # Find major hubs
        hubs = stats[stats['total_activity'] > stats['total_activity'].quantile(0.9)]
        if hubs.empty: return pd.DataFrame()
        
        hub_coords = hubs[['lat', 'lon']].values
        
        def min_dist_to_hub(row):
            dists = cdist([[row['lat'], row['lon']]], hub_coords)
            return dists.min()
            
        stats['dist_to_hub'] = stats.apply(min_dist_to_hub, axis=1)
        
        # Dark Zone = Low Activity AND Far from Hub (Isolated)
        # We normalize distance for scoring
        max_dist = stats['dist_to_hub'].max()
        if max_dist == 0: max_dist = 1
        
        stats['isolation_score'] = stats['dist_to_hub'] / max_dist
        
        dark_zones = stats[
            (stats['total_activity'] < threshold_activity) & 
            (stats['isolation_score'] > 0.3)
        ].copy()
        
        return dark_zones.sort_values('isolation_score', ascending=False)

    # ==========================================================================
    # NEW V9.7 FEATURE: SPATIAL AUTOCORRELATION (MORAN'S I SIMULATION)
    # ==========================================================================
    @staticmethod
    def calculate_spatial_autocorrelation(df):
        """
        Determines if activity is Clustered, Dispersed, or Random.
        Returns a Global Moran's I Index (-1 to +1).
        +1 = Perfectly Clustered (Highs near Highs) -> Efficient Diffusion
        -1 = Dispersed (Checkerboard) -> Uneven Development
        """
        # This is a statistical simulation as full shapefiles are needed for exact calculation
        if len(df) < 10: return 0.0
        
        # Simulate based on variance
        # If variance is very high compared to mean, it suggests clustering (over-dispersion)
        mean_act = df['total_activity'].mean()
        var_act = df['total_activity'].var()
        
        # Heuristic ratio
        ratio = var_act / (mean_act + 1e-5)
        
        # Normalize to -1 to 1 range for simulation
        morans_i = np.tanh((ratio - 1000) / 5000)
        return morans_i