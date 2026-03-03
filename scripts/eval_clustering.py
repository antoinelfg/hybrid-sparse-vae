import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize import build_model
from data.datasets import get_fsdd_dataset

def cluster_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Model params needed for build_model
    parser.add_argument("--input-length", type=int, default=8256)
    parser.add_argument("--n-atoms", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-output-dim", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="mlp")
    parser.add_argument("--decoder-type", type=str, default="linear")
    parser.add_argument("--dict-init", type=str, default="random")
    parser.add_argument("--magnitude-dist", type=str, default="gamma")
    parser.add_argument("--structure-mode", type=str, default="ternary")
    
    args, _ = parser.parse_known_args()
    args.dataset = 'fsdd'
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint {args.checkpoint}...")
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    print("Loading FSDD dataset...")
    # Fix import to use the exact same preprocessing
    dataset = get_fsdd_dataset(data_dir="./data/fsdd", use_instance_norm=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    Z_prob_list = []
    Y_list = []
    
    print("Extracting latent representations...")
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].cpu().numpy()
            
            h = model.encoder(x)
            z, info = model.latent(h, sampling="deterministic")
            
            # Use structure representation expectations as clustering feature 
            prob = info['delta'].cpu().numpy()
            
            Z_prob_list.append(prob)
            Y_list.append(y)
            
    features = np.concatenate(Z_prob_list, axis=0) # [N, K]
    Y = np.concatenate(Y_list, axis=0) # [N]
    
    n_classes = len(np.unique(Y))
    
    print(f"\nEvaluating Latents Shape: {features.shape} with {n_classes} Classes")
    print("\n--- Evaluation Results ---")
    
    # 1. Linear Probing (Supervised)
    print("Fitting Logistic Regression (Linear Probing)...")
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    clf.fit(features, Y)
    y_pred_linear = clf.predict(features)
    acc_linear = accuracy_score(Y, y_pred_linear)
    print(f"Linear Probing Accuracy: {acc_linear*100:.2f}%")
    
    # 2. KMeans Clustering (Unsupervised)
    print(f"Fitting KMeans (k={n_classes})...")
    kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    y_pred_cluster = kmeans.fit_predict(features)
    
    acc_cluster = cluster_accuracy(Y, y_pred_cluster)
    nmi = normalized_mutual_info_score(Y, y_pred_cluster)
    ari = adjusted_rand_score(Y, y_pred_cluster)
    
    print(f"Clustering Accuracy: {acc_cluster*100:.2f}%")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
