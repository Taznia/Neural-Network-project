import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
import os

def perform_clustering(features, n_clusters=10, method='kmeans'):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Invalid clustering method")
    
    labels = model.fit_predict(features)
    return labels

def visualize_clusters(features, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(save_path)
    plt.close()

def plot_cluster_distribution(labels, true_labels, title, save_path):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({'Cluster': labels, 'Category': true_labels})
    pivot_df = df.groupby(['Cluster', 'Category']).size().unstack(fill_value=0)
    pivot_df_rel = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 6))
    pivot_df_rel.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Original Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Test with dummy data
    dummy_features = np.random.randn(100, 16)
    labels = perform_clustering(dummy_features, n_clusters=3)
    print(f"Clustering labels: {labels[:10]}")
    # visualize_clusters(dummy_features, labels, "Test Clusters", "results/test_visual.png")
