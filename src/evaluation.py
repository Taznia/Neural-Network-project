from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

def evaluate_clustering(features, labels):
    metrics = {}
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        metrics['silhouette'] = silhouette_score(features, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
    else:
        metrics['silhouette'] = 0
        metrics['calinski_harabasz'] = 0
        metrics['davies_bouldin'] = 0
    return metrics

def evaluate_with_labels(true_labels, pred_labels):
    metrics = {}
    metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
    metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
    return metrics

def compute_purity(true_labels, pred_labels):
    # Purity formula: 1/n * sum_k(max_j |ck intersect tj|)
    # Handle noise labels (-1) by treating them as a separate cluster
    unique_pred = np.unique(pred_labels)
    unique_true = np.unique(true_labels)
    
    # Map labels to 0..K-1
    pred_map = {label: i for i, label in enumerate(unique_pred)}
    true_map = {label: i for i, label in enumerate(unique_true)}
    
    contingency = np.zeros((len(unique_pred), len(unique_true)))
    for i in range(len(true_labels)):
        contingency[pred_map[pred_labels[i]], true_map[true_labels[i]]] += 1
    return np.sum(np.max(contingency, axis=1)) / len(true_labels)

if __name__ == "__main__":
    # Test with dummy data
    features = np.random.randn(100, 10)
    labels = np.random.randint(0, 3, 100)
    print(f"Metrics: {evaluate_clustering(features, labels)}")
