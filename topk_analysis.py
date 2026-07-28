#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze why top-K patches become outliers after fine-tuning
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sacred import Experiment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import setup, init_environment
from data_kits import datasets
from networks import load_model
from utils_ import misc

ex = setup(Experiment('topk_analysis'))
ex.observers.clear()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_topk_features(model, ds_test, device, opt, logger):
    """Analyze feature distribution of top-K patches."""
    logger.info("=" * 60)
    logger.info("Top-K Patch Feature Analysis")
    logger.info("=" * 60)
    
    model.eval()
    
    # Collect features
    all_features = []
    topk_features = []
    confidences = []
    
    # Hook to capture query features
    captured_features = {}
    
    def purifier_hook(module, input, output):
        captured_features['features'] = output.detach()
    
    purifier_handle = model.purifier.register_forward_hook(purifier_hook)
    
    with torch.no_grad():
        valid_count = 0
        idx = 0
        max_attempts = len(ds_test) * 2
        
        while valid_count < 100 and idx < max_attempts:
            try:
                batch = ds_test[idx]
                
                qry_rgb = batch['qry_rgb'].unsqueeze(0).to(device)
                sup_rgb = batch['sup_rgb'].unsqueeze(0).to(device)
                sup_msk = batch['sup_msk'].unsqueeze(0).to(device)
                
                B = qry_rgb.shape[0]
                S = sup_rgb.shape[1]
                
                # Forward pass
                captured_features.clear()
                output = model(qry_rgb, sup_rgb, sup_msk)
                
                if 'features' not in captured_features:
                    idx += 1
                    continue
                
                features = captured_features['features']
                _, c, h, w = features.shape
                features = features.view(B, S+1, c, h, w)
                sup_fts = features[:, :-1]
                qry_fts = features[:, -1:]
                
                # Compute classifier output
                sup_mask_resized = F.interpolate(sup_msk.view(B*S, 1, sup_msk.shape[-2], sup_msk.shape[-1]),
                                                size=(h, w), mode='nearest')
                pred = model.classifier(sup_fts, qry_fts, sup_mask_resized)
                
                # Get query features
                qry_fts_flat = qry_fts.squeeze(1)  # [B, c, h, w]
                qry_fts_flat = qry_fts_flat.reshape(B, c, -1)  # [B, c, h*w]
                qry_fts_flat = qry_fts_flat.permute(0, 2, 1)  # [B, h*w, c]
                
                # Compute confidence
                C = pred[:, 1] - pred[:, 0]  # [B, h, w]
                C_flat = C.reshape(B, -1)  # [B, h*w]
                
                # Select top-K
                K = getattr(opt, 'lcm_K', 7)
                _, topk_indices = C_flat.topk(K, dim=1)  # [B, K]
                
                # Gather top-K features
                for b in range(B):
                    all_feats = qry_fts_flat[b].cpu().numpy()  # [h*w, c]
                    topk_idx = topk_indices[b].cpu().numpy()  # [K]
                    topk_feats = all_feats[topk_idx]  # [K, c]
                    conf = C_flat[b].cpu().numpy()  # [h*w]
                    
                    all_features.append(all_feats)
                    topk_features.append(topk_feats)
                    confidences.append(conf)
                
                valid_count += 1
            except (IndexError, KeyError):
                idx += 1
                continue
            
            idx += 1
        
        pbar = tqdm(total=100, desc='Analyzing')
        pbar.update(valid_count)
        pbar.close()
    
    purifier_handle.remove()
    
    # Aggregate
    all_features = np.concatenate(all_features, axis=0)  # [N, c]
    topk_features = np.concatenate(topk_features, axis=0)  # [M, c]
    confidences = np.concatenate(confidences, axis=0)  # [N]
    
    logger.info(f"Total patches: {len(all_features)}")
    logger.info(f"Top-K patches: {len(topk_features)}")
    
    # Analysis
    output_dir = Path('output') / str(opt.exp_id) / 'topk_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. t-SNE visualization (sample for speed)
    logger.info("Computing t-SNE (sampling 5000 patches)...")
    n_tsne = min(5000, len(all_features))
    sample_idx = np.random.choice(len(all_features), n_tsne, replace=False)
    all_sampled = all_features[sample_idx]
    conf_sampled = confidences[sample_idx]
    
    # Also sample top-K if too many
    n_topk_sample = min(300, len(topk_features))
    topk_sample_idx = np.random.choice(len(topk_features), n_topk_sample, replace=False)
    topk_sampled = topk_features[topk_sample_idx]
    
    combined = np.concatenate([all_sampled, topk_sampled], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined)
    
    all_embedded = embedded[:len(all_sampled)]
    topk_embedded = embedded[len(all_sampled):]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # t-SNE with confidence coloring
    scatter1 = axes[0].scatter(all_embedded[:, 0], all_embedded[:, 1], 
                               c=conf_sampled, cmap='viridis', s=10, alpha=0.5)
    axes[0].scatter(topk_embedded[:, 0], topk_embedded[:, 1], 
                   c='red', s=50, alpha=0.8, marker='x', label='Top-K')
    axes[0].set_title('t-SNE: All Patches (colored by confidence)')
    axes[0].legend()
    plt.colorbar(scatter1, ax=axes[0], label='Confidence')
    
    # t-SNE with separation
    axes[1].scatter(all_embedded[:, 0], all_embedded[:, 1], 
                   c='blue', s=10, alpha=0.3, label='All patches')
    axes[1].scatter(topk_embedded[:, 0], topk_embedded[:, 1], 
                   c='red', s=50, alpha=0.8, marker='x', label='Top-K patches')
    axes[1].set_title('t-SNE: Top-K vs All')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'tsne_visualization.png'}")
    
    # 2. Statistical analysis
    stats = {}
    
    # Feature norms
    all_norms = np.linalg.norm(all_features, axis=1)
    topk_norms = np.linalg.norm(topk_features, axis=1)
    
    stats['all_norm_mean'] = float(all_norms.mean())
    stats['all_norm_std'] = float(all_norms.std())
    stats['topk_norm_mean'] = float(topk_norms.mean())
    stats['topk_norm_std'] = float(topk_norms.std())
    
    # Distance to mean
    all_mean = all_features.mean(axis=0, keepdims=True)
    all_dists = np.linalg.norm(all_features - all_mean, axis=1)
    topk_dists = np.linalg.norm(topk_features - all_mean, axis=1)
    
    stats['all_dist_to_mean_mean'] = float(all_dists.mean())
    stats['all_dist_to_mean_std'] = float(all_dists.std())
    stats['topk_dist_to_mean_mean'] = float(topk_dists.mean())
    stats['topk_dist_to_mean_std'] = float(topk_dists.std())
    
    # Confidence statistics
    topk_confs = confidences.reshape(-1, confidences.shape[-1])
    topk_conf_values = np.take_along_axis(topk_confs, 
                                          np.argsort(topk_confs, axis=1)[:, -7:], 
                                          axis=1)
    
    stats['confidence_mean'] = float(confidences.mean())
    stats['confidence_std'] = float(confidences.std())
    stats['topk_confidence_mean'] = float(topk_conf_values.mean())
    stats['topk_confidence_std'] = float(topk_conf_values.std())
    
    # Similarity analysis
    # Compute similarity between top-K patches and all patches
    topk_norm = topk_features / (topk_norms[:, None] + 1e-8)
    all_norm = all_features / (all_norms[:, None] + 1e-8)
    
    # Sample for efficiency
    n_samples = min(1000, len(all_norm))
    sample_idx = np.random.choice(len(all_norm), n_samples, replace=False)
    all_sample = all_norm[sample_idx]
    
    sim_matrix = topk_norm @ all_sample.T  # [M, n_samples]
    
    stats['similarity_mean'] = float(sim_matrix.mean())
    stats['similarity_std'] = float(sim_matrix.std())
    stats['similarity_median'] = float(np.median(sim_matrix))
    
    # Log statistics
    logger.info("\n=== Feature Statistics ===")
    logger.info(f"All patches - Norm: {stats['all_norm_mean']:.3f} ± {stats['all_norm_std']:.3f}")
    logger.info(f"Top-K patches - Norm: {stats['topk_norm_mean']:.3f} ± {stats['topk_norm_std']:.3f}")
    logger.info(f"All patches - Dist to mean: {stats['all_dist_to_mean_mean']:.3f} ± {stats['all_dist_to_mean_std']:.3f}")
    logger.info(f"Top-K patches - Dist to mean: {stats['topk_dist_to_mean_mean']:.3f} ± {stats['topk_dist_to_mean_std']:.3f}")
    logger.info(f"Similarity (top-K vs all): {stats['similarity_mean']:.3f} ± {stats['similarity_std']:.3f}")
    
    # Save statistics
    with open(output_dir / 'topk_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved: {output_dir / 'topk_stats.json'}")
    
    # 3. Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Norm distribution
    axes[0, 0].hist(all_norms, bins=50, alpha=0.5, label='All', color='blue')
    axes[0, 0].hist(topk_norms, bins=50, alpha=0.7, label='Top-K', color='red')
    axes[0, 0].set_title('Feature Norm Distribution')
    axes[0, 0].set_xlabel('Norm')
    axes[0, 0].legend()
    
    # Distance to mean
    axes[0, 1].hist(all_dists, bins=50, alpha=0.5, label='All', color='blue')
    axes[0, 1].hist(topk_dists, bins=50, alpha=0.7, label='Top-K', color='red')
    axes[0, 1].set_title('Distance to Global Mean')
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].legend()
    
    # Confidence distribution
    axes[1, 0].hist(confidences, bins=100, alpha=0.7, color='green')
    axes[1, 0].set_title('Confidence Distribution (All Patches)')
    axes[1, 0].set_xlabel('Confidence (fg - bg)')
    axes[1, 0].axvline(confidences.mean(), color='red', linestyle='--', label=f'Mean={confidences.mean():.2f}')
    axes[1, 0].legend()
    
    # Similarity distribution
    axes[1, 1].hist(sim_matrix.flatten(), bins=100, alpha=0.7, color='purple')
    axes[1, 1].set_title('Similarity Distribution (Top-K vs All)')
    axes[1, 1].set_xlabel('Cosine Similarity')
    axes[1, 1].axvline(sim_matrix.mean(), color='red', linestyle='--', label=f'Mean={sim_matrix.mean():.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'feature_distributions.png'}")
    
    return stats


@ex.automain
def main(_run, _config):
    opt, logger, device = init_environment(ex, _run, _config)
    
    # Load data
    ds_test, data_loader, num_classes = datasets.load(opt, logger, "test")
    logger.info(f'Loaded {len(ds_test)} testing samples')
    
    # Load model
    model = load_model(opt, logger)
    model_ckpt = misc.find_snapshot(_run.run_dir.parent, opt.exp_id, None)
    logger.info(f"Loading checkpoint from {model_ckpt}")
    model.load_weights(model_ckpt, logger, strict=True)
    model = model.to(device)
    model.eval()
    
    # Run analysis
    stats = analyze_topk_features(model, ds_test, device, opt, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis Complete")
    logger.info("=" * 60)
    
    # Check if top-K are outliers
    if stats['topk_dist_to_mean_mean'] > stats['all_dist_to_mean_mean'] * 1.5:
        logger.warning("WARNING: Top-K patches are significantly farther from mean!")
        logger.warning("This confirms they are outliers in feature space.")
    
    if stats['similarity_mean'] < 0:
        logger.warning("WARNING: Top-K patches have negative similarity with other patches!")
        logger.warning("This explains why LCM calibration signal is negative.")
