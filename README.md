# VAE for Hybrid Language Music Clustering

This repository contains the implementation of an unsupervised learning pipeline for clustering hybrid language music tracks (English and Bangla) using Variational Autoencoders (VAEs).

## Features
- **Multi-Modal Data**: Combines audio MFCC features with hybrid language lyrics.
- **Architectures**: Includes Standard VAE, Beta-VAE (for disentanglement), and Conditional VAE (CVAE).
- **Clustering**: Supports K-Means, Agglomerative Clustering, and DBSCAN.
- **Evaluation**: Computes Silhouette Score, CH Index, DB Index, and Purity.
- **Visualization**: Generates t-SNE latent space plots.

## Repository Structure
```
project/
├── data/           # Audio features and lyrics CSVs
├── notebooks/      # Exploratory notebooks
├── src/            # Core source code
│   ├── dataset.py    # Data loading and preprocessing
│   ├── vae.py        # PyTorch model definitions
│   ├── clustering.py # Clustering algorithms
│   └── evaluation.py # Metric computations
├── results/        # Metrics and t-SNE visualizations
├── main.py         # Main execution script
└── report.md       # NeurIPS-like scientific report
```

## How to Run
1. Install dependencies: `pip install torch pandas numpy scikit-learn matplotlib`.
2. Run the full pipeline: `python main.py`.
3. Check `results/` for metrics and visualizations.

## Model Performance
| Method | Silhouette Score |
| :--- | :--- |
| PCA Baseline | 0.0988 |
| Beta-VAE | 0.0912 |
| **CVAE** | **0.1245** |
