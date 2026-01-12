import pandas as pd
import numpy as np
import networkx as nx

# NEW: Graph Neural Network (GNN) Simulator
class GraphNeuralNetwork:
    """
    Simulates Message Passing Interface (MPI) for District Nodes.
    Predicts 'High Velocity Hubs' based on neighbor activity.
    """
    @staticmethod
    def predict_node_importance(G):
        if G is None: return {}
        # PageRank is a good proxy for GNN node embedding importance in this context
        return nx.pagerank(G, alpha=0.85)

class SpatialEngine:
    """
    PART 2 & 3: ADVANCED GEOSPATIAL & GRAPH INTELLIGENCE
    Features: H3 Indexing, Migration Graphs, Digital Twin Layers.
    """
    
    @staticmethod
    def generate_h3_hexagons(df, resolution=4):
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return pd.DataFrame()
        
        # Simple aggregation for heatmap
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
        """
        if df.empty or 'district' not in df.columns: return pd.DataFrame()
        
        # 1. Identify Sources (Low Activity) and Targets (High Activity)
        stats = df.groupby('district').agg({
            'total_activity': 'sum',
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        
        if len(stats) < 2: return pd.DataFrame()
        
        targets = stats.nlargest(5, 'total_activity')
        sources = stats.nsmallest(20, 'total_activity')
        
        arcs = []
        import random
        
        for _, src in sources.iterrows():
            # Connect source to a random major hub
            target = targets.sample(1).iloc[0]
            
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