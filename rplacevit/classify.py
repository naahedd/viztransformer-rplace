import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from .dataset import RPlaceDataset

def analyze_user_patterns(dataset, viz_dims=2, cluster_count=100, viz_path='user_patterns.png'):
    """
    Analyze and visualize user behavior patterns using clustering.
    
    Args:
        dataset (RPlaceDataset): Input dataset with user data
        viz_dims (int): Visualization dimensions (2 or 3)
        cluster_count (int): Number of behavior clusters
        viz_path (str): Output visualization path
    """
    if not hasattr(dataset, 'user_features'):
        dataset.compute_users_features()
    
    participant_ids = list(dataset.user_features.keys())
    behavior_data = np.array(list(dataset.user_features.values()))
    
    # Cluster user behaviors
    clustering = KMeans(n_clusters=cluster_count, random_state=42).fit(behavior_data)
    group_assignments = clustering.labels_
    
    # Reduce dimensionality for visualization
    dim_reducer = PCA(n_components=viz_dims)
    viz_coords = dim_reducer.fit_transform(behavior_data)
    
    # Setup visualization
    plot = plt.figure(figsize=(16, 14))
    color_scheme = plt.get_cmap('tab20')
    
    if viz_dims == 2:
        canvas = plot.add_subplot(111)
        points = canvas.scatter(viz_coords[:, 0], viz_coords[:, 1], 
                              c=group_assignments, cmap=color_scheme, alpha=0.7, s=12)
        canvas.set_xlabel('Primary Behavior Component', fontsize=12)
        canvas.set_ylabel('Secondary Behavior Component', fontsize=12)
    elif viz_dims == 3:
        canvas = plot.add_subplot(111, projection='3d')
        points = canvas.scatter(viz_coords[:, 0], viz_coords[:, 1], viz_coords[:, 2],
                              c=group_assignments, cmap=color_scheme, alpha=0.7, s=12)
        canvas.set_xlabel('Primary Behavior Component', fontsize=12)
        canvas.set_ylabel('Secondary Behavior Component', fontsize=12)
        canvas.set_zlabel('Tertiary Behavior Component', fontsize=12)
    else:
        raise ValueError("Visualization dimensions must be 2 or 3")
    
    plt.title(f'{viz_dims}D Analysis of User Behavior Patterns\n({cluster_count} groups)', fontsize=16)
    
    legend = plt.colorbar(points, ax=canvas, aspect=40)
    legend.set_label('Behavior Group', fontsize=12)
    
    legend.ax.tick_params(labelsize=10)
    canvas.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save results
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Analysis visualization saved as {viz_path}")
    
    analysis_data = {
        'participant_ids': participant_ids, 
        'coordinates': viz_coords,
        'group_assignments': group_assignments
    }
    with open('user_behavior_analysis.pkl', 'wb') as f:
        pickle.dump(analysis_data, f)
    print("Analysis data saved as user_behavior_analysis.pkl")

if __name__ == "__main__":
    dataset = RPlaceDataset()
    analyze_user_patterns(dataset, viz_dims=2, cluster_count=100, viz_path='user_patterns_2d.png')
    analyze_user_patterns(dataset, viz_dims=3, cluster_count=100, viz_path='user_patterns_3d.png')