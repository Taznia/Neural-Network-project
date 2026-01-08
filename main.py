import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import load_data, prepare_unified_dataset
from src.vae import VAE, BetaVAE, CVAE, Autoencoder, ConvVAE, train_vae
from src.clustering import perform_clustering, visualize_clusters, plot_cluster_distribution
from src.evaluation import evaluate_clustering, evaluate_with_labels, compute_purity
from sklearn.preprocessing import OneHotEncoder

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_path, "results")
    vis_dir = os.path.join(results_dir, "latent_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    print("--- Phase 2: Loading Unified Multi-modal Data ---")
    audio_df, eng_df, ban_df = load_data(base_path)
    X_a, X_l, y_genre, genre_names, lang_labels = prepare_unified_dataset(audio_df, eng_df, ban_df, n_samples=3000)
    
    # Joint Multi-modal Input
    X_joint = np.hstack([X_a, X_l])
    input_dim = X_joint.shape[1]
    latent_dim = 16
    
    # One-hot encode Genres for Conditioning 
    encoder = OneHotEncoder(sparse_output=False)
    C_genre = encoder.fit_transform(y_genre.reshape(-1, 1))
    cond_dim = C_genre.shape[1]
    
    all_results = []

    experiments = [
        ('Baseline_Direct', 'direct', None, 'kmeans'),
        ('Baseline_PCA', 'pca', None, 'kmeans'),
        ('Autoencoder', 'ae', Autoencoder(input_dim, latent_dim), 'kmeans'),
        ('VAE_Basic', 'vae', VAE(input_dim, latent_dim), 'kmeans'),
        ('BetaVAE_Disentangled', 'vae', BetaVAE(input_dim, latent_dim), 'kmeans'),
        ('ConvVAE_Audio', 'vae', ConvVAE(input_dim, latent_dim), 'kmeans'),
        ('CVAE_Multimodal', 'cvae', CVAE(input_dim, cond_dim, latent_dim), 'kmeans'),
        ('CVAE_Agglomerative', 'cvae', None, 'agglomerative'), # Reuse CVAE latent
        ('CVAE_DBSCAN', 'cvae', None, 'dbscan'), 
    ]

    last_latent = None
    last_recon = None

    for name, mtype, model, clust_method in experiments:
        print(f"\n--- Running Experiment: {name} ({clust_method}) ---")
        
        if mtype == 'direct':
            X_feat = X_joint
        elif mtype == 'pca':
            from sklearn.decomposition import PCA
            X_feat = PCA(n_components=latent_dim).fit_transform(X_joint)
        elif model is not None:
            # Training parameters
            beta = 4.0 if 'BetaVAE' in name else 1.0
            epochs = 30
            
            if mtype == 'cvae':
                trained_model = train_vae(model, X_joint, cond_data=C_genre, epochs=epochs, batch_size=64, beta=beta)
                trained_model.eval()
                with torch.no_grad():
                    recon, mu, _ = trained_model(torch.FloatTensor(X_joint), torch.FloatTensor(C_genre))
                    X_feat = mu.numpy()
                    last_recon = recon
            else:
                trained_model = train_vae(model, X_joint, epochs=epochs, batch_size=64, beta=beta)
                trained_model.eval()
                with torch.no_grad():
                    outputs = trained_model(torch.FloatTensor(X_joint))
                    if len(outputs) == 3: # VAE
                        recon, mu, _ = outputs
                        X_feat = mu.numpy()
                        last_recon = recon
                    else: # AE
                        recon, z = outputs
                        X_feat = z.numpy()
                        last_recon = recon
            last_latent = X_feat
        else:
           
            X_feat = last_latent

        # Clustering
        n_clusters = 10 if clust_method != 'dbscan' else None
        labels = perform_clustering(X_feat, n_clusters=n_clusters, method=clust_method)
        
        # Evaluation
        metrics = evaluate_clustering(X_feat, labels)
        ext_metrics = evaluate_with_labels(y_genre, labels)
        metrics.update(ext_metrics)
        metrics['purity'] = compute_purity(y_genre, labels)
        
        all_results.append({'Experiment': name, 'Clustering': clust_method, **metrics})
        
        # Visualization
        vis_path = os.path.join(vis_dir, f"{name.lower()}_clusters.png")
        visualize_clusters(X_feat, labels, f"Clusters: {name}", vis_path)
        
    # Reconstruction Plot for last VAE/CVAE
    if last_recon is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(X_joint[:20], aspect='auto', cmap='viridis')
        plt.title("Original Features")
        plt.subplot(1, 2, 2)
        plt.imshow(last_recon[:20].numpy(), aspect='auto', cmap='viridis')
        plt.title("Reconstructed Features")
        plt.savefig(os.path.join(results_dir, "last_reconstruction.png"))
        plt.close()

    # Final Results Summary
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, "clustering_metrics.csv"), index=False)
    print("\n--- Final Results Summary ---")
    print(results_df[['Experiment', 'Clustering', 'silhouette', 'purity', 'ari', 'nmi']])
    
    # Distribution for best model 
    best_labels = perform_clustering(last_latent, n_clusters=10, method='kmeans')
    plot_cluster_distribution(best_labels, [genre_names[i] for i in y_genre], 
                              "Genre Distribution in Clusters (CVAE)", 
                              os.path.join(results_dir, "best_model_genre_dist.png"))

if __name__ == "__main__":
    main()
