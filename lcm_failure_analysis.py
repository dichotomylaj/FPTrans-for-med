#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LCM Failure Analysis for Fine-tuned Models

This script analyzes why LCM crashes after fine-tuning:
1. Compare feature distributions (with/without fine-tuning)
2. Analyze LCM calibration signals
3. Visualize top-K patch selection
4. Check similarity distributions

Example:
    python lcm_failure_analysis.py with exp_id=6 split=0
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
import seaborn as sns
from tqdm import tqdm

from config import setup, init_environment
from data_kits import datasets
from networks import load_model
from utils_ import misc

ex = setup(Experiment('lcm_failure_analysis'))
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


def analyze_lcm_calibration(model, ds_test, device, opt, logger):
    """Analyze LCM calibration signals for fine-tuned model."""
    logger.info("=" * 60)
    logger.info("LCM Calibration Analysis (Fine-tuned Model)")
    logger.info("=" * 60)
    
    model.eval()
    
    # Collect statistics
    stats = {
        'confidence_maps': [],
        'topk_indices': [],
        'similarities': [],
        'calibrations': [],
        'pred_before': [],
        'pred_after': [],
    }
    
    # Hook to capture query features from purifier output
    captured_features = {}
    
    def purifier_hook(module, input, output):
        # output: [B(S+1), c, h, w]
        captured_features['features'] = output.detach()
    
    # Register hook on purifier
    purifier_handle = model.purifier.register_forward_hook(purifier_hook)
    
    with torch.no_grad():
        valid_count = 0
        idx = 0
        max_attempts = len(ds_test) * 2  # Try up to 2x the reported length
        pbar = tqdm(total=50, desc='Analyzing LCM')
        
        while valid_count < 50 and idx < max_attempts:
            try:
                batch = ds_test[idx]
                
                qry_rgb = batch['qry_rgb'].unsqueeze(0).to(device)
                sup_rgb = batch['sup_rgb'].unsqueeze(0).to(device)
                sup_msk = batch['sup_msk'].unsqueeze(0).to(device)
                qry_msk = batch['qry_msk'].unsqueeze(0).to(device)
                
                valid_count += 1
                pbar.update(1)
            except (IndexError, KeyError) as e:
                idx += 1
                continue
            
            B = qry_rgb.shape[0]
            S = sup_rgb.shape[1]
            
            # Forward pass
            captured_features.clear()
            output = model(qry_rgb, sup_rgb, sup_msk)
            
            # Extract features from captured data
            if 'features' not in captured_features:
                idx += 1
                continue
            
            features = captured_features['features']  # [B(S+1), c, h, w]
            _, c, h, w = features.shape
            features = features.view(B, S+1, c, h, w)
            sup_fts = features[:, :-1]  # [B, S, c, h, w]
            qry_fts = features[:, -1:]  # [B, 1, c, h, w] - last one is query
            
            # Compute classifier output (before upsampling)
            sup_mask_resized = F.interpolate(sup_msk.view(B*S, 1, sup_msk.shape[-2], sup_msk.shape[-1]), 
                                            size=(h, w), mode='nearest')  # [BS, 1, h, w]
            pred_raw = model.classifier(sup_fts, qry_fts, sup_mask_resized)  # [B, 2, h, w]
            
            # Squeeze qry_fts for LCM: [B, 1, c, h, w] -> [B, c, h, w]
            qry_fts_for_lcm = qry_fts.squeeze(1)
            
            # Manually compute LCM calibration
            pred_before_lcm = pred_raw.clone()
            
            # LCM logic (copied from FPTrans.py)
            _, _, h_pred, w_s = pred_raw.shape
            C = pred_raw[:, 1] - pred_raw[:, 0]  # Confidence map
            C_flat = C.reshape(B, -1)
            
            # Top-K selection
            K = getattr(opt, 'lcm_K', 7)
            w = getattr(opt, 'lcm_w', 0.8)
            beta = getattr(opt, 'lcm_beta', 0.9)
            
            _, topk_indices = C_flat.topk(K, dim=1)
            
            # Get top-K features
            feats_flat = qry_fts_for_lcm.reshape(B, qry_fts_for_lcm.shape[1], -1)  # [B, c, h*w]
            topk_feats = torch.gather(feats_flat, 2, topk_indices.unsqueeze(1).expand(-1, feats_flat.shape[1], -1))
            
            # Compute similarities
            feats_norm = F.normalize(feats_flat, dim=1)
            topk_norm = F.normalize(topk_feats, dim=1)
            sim = torch.bmm(topk_norm.permute(0, 2, 1), feats_norm)
            
            # Compute calibration
            calibration = (w * (sim - beta)).sum(dim=1)
            calibration = calibration.reshape(B, h_pred, w_s)
            
            # Apply calibration
            pred_with_lcm = pred_raw.clone()
            pred_with_lcm[:, 1] = pred_with_lcm[:, 1] + calibration
            
            # Collect statistics
            stats['confidence_maps'].append(C.flatten().cpu().numpy())
            stats['topk_indices'].append(topk_indices.cpu().numpy())
            stats['similarities'].append(sim.flatten().cpu().numpy())
            stats['calibrations'].append(calibration.flatten().cpu().numpy())
            stats['pred_before'].append(pred_before_lcm[:, 1].flatten().cpu().numpy())
            stats['pred_after'].append(pred_with_lcm[:, 1].flatten().cpu().numpy())
            
            idx += 1
        
        pbar.close()
    
    purifier_handle.remove()
    
    # Aggregate statistics
    logger.info("\n=== LCM Calibration Statistics ===")
    
    confidences = np.concatenate(stats['confidence_maps'])
    similarities = np.concatenate(stats['similarities'])
    calibrations = np.concatenate(stats['calibrations'])
    pred_before = np.concatenate(stats['pred_before'])
    pred_after = np.concatenate(stats['pred_after'])
    
    logger.info(f"Confidence map: mean={confidences.mean():.4f}, std={confidences.std():.4f}")
    logger.info(f"Similarities: mean={similarities.mean():.4f}, std={similarities.std():.4f}, min={similarities.min():.4f}, max={similarities.max():.4f}")
    logger.info(f"Calibration: mean={calibrations.mean():.4f}, std={calibrations.std():.4f}, min={calibrations.min():.4f}, max={calibrations.max():.4f}")
    logger.info(f"Pred before LCM: mean={pred_before.mean():.4f}, std={pred_before.std():.4f}")
    logger.info(f"Pred after LCM: mean={pred_after.mean():.4f}, std={pred_after.std():.4f}")
    logger.info(f"Pred change: {(pred_after - pred_before).mean():.4f}")
    
    # Plot distributions
    output_dir = Path('output') / str(opt.exp_id) / 'lcm_failure_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Confidence distribution
    axes[0, 0].hist(confidences, bins=100, alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Confidence Map Distribution (fg - bg)')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(confidences.mean(), color='red', linestyle='--', label=f'Mean={confidences.mean():.3f}')
    axes[0, 0].legend()
    
    # 2. Similarity distribution
    axes[0, 1].hist(similarities, bins=100, alpha=0.7, color='coral')
    axes[0, 1].set_title(f'Similarity Distribution (top-{K} patches)')
    axes[0, 1].set_xlabel('Cosine Similarity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(beta, color='red', linestyle='--', label=f'β={beta}')
    axes[0, 1].legend()
    
    # 3. Calibration distribution
    axes[0, 2].hist(calibrations, bins=100, alpha=0.7, color='green')
    axes[0, 2].set_title('Calibration Signal Distribution')
    axes[0, 2].set_xlabel('Calibration Value')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].axvline(0, color='red', linestyle='--', label='Zero')
    axes[0, 2].legend()
    
    # 4. Prediction before vs after
    axes[1, 0].hist(pred_before, bins=100, alpha=0.5, label='Before LCM', color='blue')
    axes[1, 0].hist(pred_after, bins=100, alpha=0.5, label='After LCM', color='orange')
    axes[1, 0].set_title('Prediction Distribution (Foreground Score)')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 5. Prediction change
    pred_change = pred_after - pred_before
    axes[1, 1].hist(pred_change, bins=100, alpha=0.7, color='purple')
    axes[1, 1].set_title('Prediction Change (After - Before)')
    axes[1, 1].set_xlabel('Change')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero')
    axes[1, 1].legend()
    
    # 6. Calibration vs Confidence scatter
    axes[1, 2].scatter(confidences[::100], calibrations[::100], alpha=0.3, s=1)
    axes[1, 2].set_title('Calibration vs Confidence')
    axes[1, 2].set_xlabel('Confidence')
    axes[1, 2].set_ylabel('Calibration')
    axes[1, 2].axhline(0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lcm_calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nSaved: {output_dir / 'lcm_calibration_analysis.png'}")
    
    # Save statistics
    stats_summary = {
        'confidence_mean': float(confidences.mean()),
        'confidence_std': float(confidences.std()),
        'similarity_mean': float(similarities.mean()),
        'similarity_std': float(similarities.std()),
        'calibration_mean': float(calibrations.mean()),
        'calibration_std': float(calibrations.std()),
        'pred_change_mean': float(pred_change.mean()),
        'pred_change_std': float(pred_change.std()),
    }
    
    with open(output_dir / 'lcm_stats.json', 'w') as f:
        json.dump(stats_summary, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved: {output_dir / 'lcm_stats.json'}")
    
    return stats_summary


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
    stats = analyze_lcm_calibration(model, ds_test, device, opt, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis Complete")
    logger.info("=" * 60)
    logger.info(f"Calibration mean: {stats['calibration_mean']:.4f}")
    logger.info(f"Calibration std: {stats['calibration_std']:.4f}")
    logger.info(f"Prediction change: {stats['pred_change_mean']:.4f}")
    
    if abs(stats['calibration_mean']) > 1.0:
        logger.warning("WARNING: Calibration signal is very large! This may cause prediction corruption.")
    
    if stats['calibration_std'] > stats['confidence_std']:
        logger.warning("WARNING: Calibration variance exceeds confidence variance. LCM may dominate predictions.")
