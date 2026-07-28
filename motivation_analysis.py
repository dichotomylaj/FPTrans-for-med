#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motivation Analysis for Cross-Domain Few-Shot Medical Segmentation

This script performs comprehensive analyses to motivate the LCM method:
1. Feature-level: t-SNE, norm distribution, multi-level features, similarity
2. Prediction-level: confidence map quality, error analysis, calibration
3. Loss landscape: 2D visualization, sharpness metrics
4. Domain gap quantification

Outputs to: output/{exp_id}/motivation_analysis/
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set matplotlib to non-interactive backend (no display needed)
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

from config import setup
from constants import on_cloud
from data_kits import datasets
from networks import load_model
from utils_ import misc

ex = setup(Experiment('motivation_analysis'))
ex.observers.clear()


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in json.dump."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# Analysis 1: Feature-level Domain Gap (t-SNE)
# ============================================================================
def analyze_feature_domain_gap(model, data_loader, device, opt, logger):
    """t-SNE visualization of source vs target domain features at different ViT blocks."""
    logger.info("=" * 60)
    logger.info("Analysis 1: Feature Domain Gap (t-SNE)")
    logger.info("=" * 60)

    model.eval()

    # Hook different ViT blocks to extract intermediate features
    block_features = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            B = output.shape[0]
            num_patches = 576
            feats = output[:, 1:1+num_patches, :]  # [B, 576, 768]
            block_features[name] = feats.detach().cpu()
        return hook_fn

    blocks_to_analyze = [0, 3, 6, 9, 11]
    backbone = model.encoder.backbone
    for idx in blocks_to_analyze:
        if idx < len(backbone.blocks):
            h = backbone.blocks[idx].register_forward_hook(make_hook(f'block_{idx}'))
            hooks.append(h)

    target_feats = {f'block_{idx}': [] for idx in blocks_to_analyze}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Collecting target features')):
            qry_rgb = batch['qry_rgb'].to(device)
            sup_rgb = batch['sup_rgb'].to(device)
            sup_msk = batch['sup_msk'].to(device)

            block_features.clear()
            output = model(qry_rgb, sup_rgb, sup_msk)

            for idx in blocks_to_analyze:
                key = f'block_{idx}'
                if key in block_features:
                    feats = block_features[key].mean(dim=1)  # [B, 768]
                    target_feats[key].append(feats)

    for h in hooks:
        h.remove()

    for key in target_feats:
        if target_feats[key]:
            target_feats[key] = torch.cat(target_feats[key], dim=0).numpy()

    output_dir = Path('output') / str(opt.exp_id) / 'motivation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(blocks_to_analyze), figsize=(4*len(blocks_to_analyze), 4))

    for i, idx in enumerate(blocks_to_analyze):
        key = f'block_{idx}'
        if key not in target_feats or len(target_feats[key]) == 0:
            continue

        feats = target_feats[key]
        if not isinstance(feats, np.ndarray):
            feats = np.array(feats)
        n_samples = min(500, len(feats))
        feats_sample = feats[:n_samples]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        feats_2d = tsne.fit_transform(feats_sample)

        ax = axes[i] if len(blocks_to_analyze) > 1 else axes
        scatter = ax.scatter(feats_2d[:, 0], feats_2d[:, 1], c=range(n_samples),
                           cmap='viridis', s=10, alpha=0.7)
        ax.set_title(f'Block {idx}\n(Target Domain)', fontsize=12)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_tsne_target_domain.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_dir / 'fig1_tsne_target_domain.png'}")

    return {'status': 'completed', 'blocks_analyzed': blocks_to_analyze}


# ============================================================================
# Analysis 2: Feature Norm Distribution
# ============================================================================
def analyze_feature_norm_distribution(model, data_loader, device, opt, logger):
    """Analyze feature norm distributions across domains and layers."""
    logger.info("=" * 60)
    logger.info("Analysis 2: Feature Norm Distribution")
    logger.info("=" * 60)
    
    model.eval()
    
    # Collect feature norms at different layers
    block_norms = {f'block_{idx}': [] for idx in [0, 3, 6, 9, 11]}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            B = output.shape[0]
            num_patches = 576
            feats = output[:, 1:1+num_patches, :]  # [B, 576, 768]
            norms = feats.norm(dim=-1)  # [B, 576]
            block_norms[name].append(norms.detach().cpu())
        return hook_fn
    
    backbone = model.encoder.backbone
    for idx in [0, 3, 6, 9, 11]:
        if idx < len(backbone.blocks):
            h = backbone.blocks[idx].register_forward_hook(make_hook(f'block_{idx}'))
            hooks.append(h)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Collecting norms')):
            qry_rgb = batch['qry_rgb'].to(device)
            sup_rgb = batch['sup_rgb'].to(device)
            sup_msk = batch['sup_msk'].to(device)
            
            _ = model(qry_rgb, sup_rgb, sup_msk)
    
    for h in hooks:
        h.remove()
    
    # Plot norm distributions
    output_dir = Path('output') / str(opt.exp_id) / 'motivation_analysis'
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, idx in enumerate([0, 3, 6, 9, 11]):
        key = f'block_{idx}'
        if block_norms[key]:
            norms = torch.cat(block_norms[key], dim=0).numpy().flatten()
            
            ax = axes[i]
            ax.hist(norms, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'Block {idx}\nMean={norms.mean():.2f}, Std={norms.std():.2f}', fontsize=11)
            ax.set_xlabel('Feature Norm')
            ax.set_ylabel('Count')
            ax.axvline(norms.mean(), color='red', linestyle='--', label='Mean')
            ax.legend()
    
    plt.suptitle('Feature Norm Distribution Across ViT Blocks (Target Domain)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_feature_norm_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_dir / 'fig2_feature_norm_distribution.png'}")
    
    return {'status': 'completed'}


# ============================================================================
# Analysis 3: Confidence Map Quality Analysis
# ============================================================================
def analyze_confidence_map_quality(model, data_loader, device, opt, logger):
    """Analyze correlation between prediction confidence and actual errors."""
    logger.info("=" * 60)
    logger.info("Analysis 3: Confidence Map Quality")
    logger.info("=" * 60)
    
    model.eval()
    
    confidence_list = []
    error_list = []
    iou_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Analyzing confidence')):
            qry_rgb = batch['qry_rgb'].to(device)
            sup_rgb = batch['sup_rgb'].to(device)
            sup_msk = batch['sup_msk'].to(device)
            qry_msk = batch['qry_msk'].to(device)
            
            output = model(qry_rgb, sup_rgb, sup_msk)
            pred = output['out']  # [B, 2, H, W]
            
            # Confidence: max probability
            prob = F.softmax(pred, dim=1)
            confidence = prob.max(dim=1)[0]  # [B, H, W]
            
            # Prediction and error
            pred_label = pred.argmax(dim=1)  # [B, H, W]
            gt = qry_msk.to(device)
            
            # Resize to match
            if pred_label.shape != gt.shape:
                pred_label_resized = F.interpolate(pred_label.unsqueeze(1).float(), 
                                                   size=gt.shape[-2:], mode='nearest').squeeze(1).long()
                confidence_resized = F.interpolate(confidence.unsqueeze(1), 
                                                   size=gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            else:
                pred_label_resized = pred_label
                confidence_resized = confidence
            
            # Error map
            error = (pred_label_resized != gt).float()  # [B, H, W]
            
            # Per-sample statistics
            for b in range(pred.shape[0]):
                conf_mean = confidence_resized[b].mean().item()
                error_rate = error[b].mean().item()
                
                # IoU
                intersection = ((pred_label_resized[b] == 1) & (gt[b] == 1)).sum().item()
                union = ((pred_label_resized[b] == 1) | (gt[b] == 1)).sum().item()
                iou = intersection / (union + 1e-6)
                
                confidence_list.append(conf_mean)
                error_list.append(error_rate)
                iou_list.append(iou)
    
    # Plot correlations
    output_dir = Path('output') / str(opt.exp_id) / 'motivation_analysis'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Confidence vs Error
    ax = axes[0]
    ax.scatter(confidence_list, error_list, alpha=0.5, s=20)
    ax.set_xlabel('Mean Confidence', fontsize=11)
    ax.set_ylabel('Error Rate', fontsize=11)
    ax.set_title('Confidence vs Error\n(Lower is better)', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Confidence vs IoU
    ax = axes[1]
    ax.scatter(confidence_list, iou_list, alpha=0.5, s=20, color='green')
    ax.set_xlabel('Mean Confidence', fontsize=11)
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title('Confidence vs IoU\n(Higher is better)', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Error vs IoU
    ax = axes[2]
    ax.scatter(error_list, iou_list, alpha=0.5, s=20, color='red')
    ax.set_xlabel('Error Rate', fontsize=11)
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title('Error vs IoU\n(Negative correlation expected)', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_confidence_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute correlation coefficients
    from scipy.stats import pearsonr, spearmanr
    
    conf_arr = np.array(confidence_list)
    error_arr = np.array(error_list)
    iou_arr = np.array(iou_list)
    
    conf_error_corr, _ = pearsonr(conf_arr, error_arr)
    conf_iou_corr, _ = pearsonr(conf_arr, iou_arr)
    
    logger.info(f"  Confidence-Error correlation: {conf_error_corr:.4f}")
    logger.info(f"  Confidence-IoU correlation: {conf_iou_corr:.4f}")
    logger.info(f"  Saved: {output_dir / 'fig3_confidence_quality.png'}")
    
    return {
        'status': 'completed',
        'confidence_error_correlation': float(conf_error_corr),
        'confidence_iou_correlation': float(conf_iou_corr)
    }


# ============================================================================
# Analysis 4: Multi-Level Feature Similarity
# ============================================================================
def analyze_multilevel_feature_similarity(model, data_loader, device, opt, logger):
    """Analyze intra-class vs inter-class feature similarity at different layers."""
    logger.info("=" * 60)
    logger.info("Analysis 4: Multi-Level Feature Similarity")
    logger.info("=" * 60)
    
    model.eval()
    
    # Collect features and labels
    features_by_block = {f'block_{idx}': [] for idx in [0, 3, 6, 9, 11]}
    labels = []
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            B = output.shape[0]
            num_patches = 576
            feats = output[:, 1:1+num_patches, :].mean(dim=1)  # [B, 768]
            features_by_block[name].append(feats.detach().cpu())
        return hook_fn
    
    backbone = model.encoder.backbone
    for idx in [0, 3, 6, 9, 11]:
        if idx < len(backbone.blocks):
            h = backbone.blocks[idx].register_forward_hook(make_hook(f'block_{idx}'))
            hooks.append(h)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Collecting features')):
            qry_rgb = batch['qry_rgb'].to(device)
            sup_rgb = batch['sup_rgb'].to(device)
            sup_msk = batch['sup_msk'].to(device)

            # Extract case_id from qry_names for organ class label
            case_id = f'batch_{batch_idx}'
            qry_names = batch.get('qry_names', [])
            if qry_names is not None and len(qry_names) > 0:
                raw_name = qry_names[0]
                if isinstance(raw_name, (bytes, np.bytes_)):
                    raw_name = raw_name.decode('utf-8')
                elif isinstance(raw_name, np.str_):
                    raw_name = str(raw_name)
                case_id = str(raw_name.split('_')[0] if '_' in raw_name else raw_name)

            labels.append(case_id)
            _ = model(qry_rgb, sup_rgb, sup_msk)

    for h in hooks:
        h.remove()

    # Map case_ids to integer labels
    unique_cases = sorted(set(labels))
    case_to_int = {cid: i for i, cid in enumerate(unique_cases)}
    labels = np.array([case_to_int[lbl] for lbl in labels])
    
    output_dir = Path('output') / str(opt.exp_id) / 'motivation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    similarity_stats = {}
    
    for i, idx in enumerate([0, 3, 6, 9, 11]):
        key = f'block_{idx}'
        if not features_by_block[key]:
            continue
        
        feats = torch.cat(features_by_block[key], dim=0).numpy()  # [N, 768]
        
        # Normalize
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity matrix
        sim_matrix = feats_norm @ feats_norm.T
        
        # Intra-class vs inter-class
        n = len(labels)
        intra_sims = []
        inter_sims = []
        
        for j in range(n):
            for k in range(j+1, n):
                if labels[j] == labels[k]:
                    intra_sims.append(sim_matrix[j, k])
                else:
                    inter_sims.append(sim_matrix[j, k])
        
        intra_mean = np.mean(intra_sims) if intra_sims else 0
        inter_mean = np.mean(inter_sims) if inter_sims else 0
        
        similarity_stats[key] = {
            'intra_class_mean': float(intra_mean),
            'inter_class_mean': float(inter_mean),
            'gap': float(intra_mean - inter_mean)
        }
        
        ax = axes[i]
        ax.hist([intra_sims, inter_sims], bins=30, label=['Intra-class', 'Inter-class'], 
                alpha=0.7, color=['green', 'red'])
        ax.set_title(f'Block {idx}\nGap={intra_mean-inter_mean:.3f}', fontsize=11)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    
    plt.suptitle('Intra-class vs Inter-class Feature Similarity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_feature_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved: {output_dir / 'fig4_feature_similarity.png'}")
    for key, stats in similarity_stats.items():
        logger.info(f"  {key}: intra={stats['intra_class_mean']:.3f}, "
                   f"inter={stats['inter_class_mean']:.3f}, gap={stats['gap']:.3f}")
    
    return {'status': 'completed', 'similarity_stats': similarity_stats}


# ============================================================================
# Analysis 5: Loss Landscape Visualization (improved)
# ============================================================================
def _filter_normalize_direction(params):
    """Generate filter-normalized random direction (Li et al. 2018).
    
    For conv/linear weights: normalize each filter independently so that
    the direction has unit norm per filter. For other params (bias, LN):
    normalize the whole tensor.
    """
    directions = []
    for p in params:
        d = torch.randn_like(p)
        if d.dim() >= 2:
            # Filter-wise normalization: normalize along all dims except the first (output channels)
            # Reshape to [C_out, -1] and normalize each row
            shape = d.shape
            d_flat = d.reshape(shape[0], -1)
            d_norm = d_flat.norm(dim=1, keepdim=True)
            d_flat = d_flat / (d_norm + 1e-8)
            d = d_flat.reshape(shape)
        else:
            d = d / (d.norm() + 1e-8)
        directions.append(d)
    return directions


def _apply_perturbation(params, directions, original_params, alpha, beta=0.0, directions2=None, scale=0.1):
    """Apply perturbation: params = orig + alpha*scale*d1 + beta*scale*d2."""
    for i, (p, d, orig) in enumerate(zip(params, directions, original_params)):
        perturbation = alpha * scale * d
        if directions2 is not None and beta != 0.0:
            perturbation = perturbation + beta * scale * directions2[i]
        p.data = orig + perturbation


def _restore_params(params, original_params):
    for p, orig in zip(params, original_params):
        p.data = orig


def analyze_loss_landscape(model, data_loader, device, opt, logger, max_samples=20):
    """Visualize loss landscape with filter-normalized directions, 2D heatmap, and LCM comparison."""
    logger.info("=" * 60)
    logger.info("Analysis 5: Loss Landscape (filter-normalized)")
    logger.info("=" * 60)
    
    model.eval()
    output_dir = Path('output') / str(opt.exp_id) / 'motivation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data batches
    batches = []
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_samples:
            break
        batches.append(batch)
    logger.info(f"  Using {len(batches)} batches for loss landscape")
    
    # Loss computation function
    def compute_loss(model, batches, device, lcm_enabled=False):
        total_loss = 0
        # LCM only works in eval mode per FPTrans forward logic
        model.eval()
        # Toggle LCM via opt.lcm (forward checks: not self.training and getattr(self.opt, 'lcm', False))
        old_lcm_val = getattr(opt, 'lcm', False)
        opt.lcm = lcm_enabled
        with torch.no_grad():
            for batch in batches:
                qry_rgb = batch['qry_rgb'].to(device)
                sup_rgb = batch['sup_rgb'].to(device)
                sup_msk = batch['sup_msk'].to(device)
                qry_msk = batch['qry_msk'].to(device)
                
                output = model(qry_rgb, sup_rgb, sup_msk, y=qry_msk)
                pred = output['out']
                gt = qry_msk.view(-1, *qry_msk.shape[-2:])
                loss = F.cross_entropy(pred, gt.long())
                total_loss += loss.item()
        opt.lcm = old_lcm_val
        return total_loss / len(batches)
    
    # Get model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    original_params = [p.clone() for p in params]
    
    # Generate TWO filter-normalized random directions (for 2D landscape)
    torch.manual_seed(42)
    directions1 = []
    directions2 = []
    for p in params:
        d1 = torch.randn_like(p)
        d2 = torch.randn_like(p)
        if d1.dim() >= 2:
            shape = d1.shape
            d1_flat = d1.reshape(shape[0], -1)
            d1_flat = d1_flat / (d1_flat.norm(dim=1, keepdim=True) + 1e-8)
            d1 = d1_flat.reshape(shape)
            d2_flat = d2.reshape(shape[0], -1)
            # Make d2 orthogonal to d1 per filter
            d2_flat = d2_flat.reshape(shape[0], -1)
            proj = (d2_flat * d1_flat).sum(dim=1, keepdim=True) * d1_flat
            d2_flat = d2_flat - proj
            d2_flat = d2_flat / (d2_flat.norm(dim=1, keepdim=True) + 1e-8)
            d2 = d2_flat.reshape(shape)
        else:
            d1 = d1 / (d1.norm() + 1e-8)
            d2 = d2 - (d2 * d1).sum() * d1
            d2 = d2 / (d2.norm() + 1e-8)
        directions1.append(d1)
        directions2.append(d2)
    
    perturbation_scale = 0.1
    
    # =========================================================================
    # Part A: 1D landscape — Baseline vs +LCM comparison
    # =========================================================================
    logger.info("  [A] 1D landscape: Baseline vs +LCM")
    alphas = np.linspace(-1.0, 1.0, 21)
    losses_baseline = []
    losses_lcm = []
    
    for alpha in tqdm(alphas, desc='1D scan (baseline vs LCM)'):
        _apply_perturbation(params, directions1, original_params, alpha, scale=perturbation_scale)
        
        loss_bl = compute_loss(model, batches, device, lcm_enabled=False)
        losses_baseline.append(loss_bl)
        
        loss_lcm = compute_loss(model, batches, device, lcm_enabled=True)
        losses_lcm.append(loss_lcm)
    
    _restore_params(params, original_params)
    
    # Plot 1D comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, losses_baseline, 'b-', linewidth=2, marker='o', markersize=4, label='Baseline')
    ax.plot(alphas, losses_lcm, 'r-', linewidth=2, marker='s', markersize=4, label='+LCM')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Original model')
    ax.set_xlabel('Perturbation Scale (α)', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('1D Loss Landscape: Baseline vs. +LCM (Filter-Normalized)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_loss_landscape_1d_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Part B: 2D loss landscape heatmap (baseline only, to save time)
    # =========================================================================
    logger.info("  [B] 2D loss landscape heatmap")
    grid_size = 15  # 15x15 = 225 evaluations
    alphas_2d = np.linspace(-0.5, 0.5, grid_size)
    betas_2d = np.linspace(-0.5, 0.5, grid_size)
    loss_grid = np.zeros((grid_size, grid_size))
    
    total_2d = grid_size * grid_size
    with tqdm(total=total_2d, desc='2D landscape scan') as pbar:
        for i, alpha in enumerate(alphas_2d):
            for j, beta in enumerate(betas_2d):
                _apply_perturbation(params, directions1, original_params,
                                   alpha, beta=beta, directions2=directions2, scale=perturbation_scale)
                loss_grid[i, j] = compute_loss(model, batches, device, lcm_enabled=False)
                pbar.update(1)
    
    _restore_params(params, original_params)
    
    # Plot 2D heatmap with contours
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    ax = axes[0]
    im = ax.imshow(loss_grid, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower',
                   cmap='viridis', aspect='auto')
    ax.set_xlabel('Direction 1 (α)', fontsize=12)
    ax.set_ylabel('Direction 2 (β)', fontsize=12)
    ax.set_title('2D Loss Landscape (Heatmap)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Loss')
    
    # Contour plot
    ax = axes[1]
    A, B = np.meshgrid(betas_2d, alphas_2d)
    levels = np.linspace(loss_grid.min(), loss_grid.max(), 15)
    cs = ax.contourf(A, B, loss_grid, levels=levels, cmap='viridis')
    ax.contour(A, B, loss_grid, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    ax.plot(0, 0, 'r*', markersize=15, label='Original model')
    ax.set_xlabel('Direction 1 (α)', fontsize=12)
    ax.set_ylabel('Direction 2 (β)', fontsize=12)
    ax.set_title('2D Loss Landscape (Contours)', fontsize=13)
    ax.legend(fontsize=10)
    plt.colorbar(cs, ax=ax, label='Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_loss_landscape_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Compute and log metrics
    # =========================================================================
    losses_bl_arr = np.array(losses_baseline)
    losses_lcm_arr = np.array(losses_lcm)
    
    sharpness_bl = float(losses_bl_arr.max() - losses_bl_arr.min())
    sharpness_lcm = float(losses_lcm_arr.max() - losses_lcm_arr.min())
    current_loss_bl = float(compute_loss(model, batches, device, lcm_enabled=False))
    current_loss_lcm = float(compute_loss(model, batches, device, lcm_enabled=True))
    
    logger.info(f"  Baseline — loss={current_loss_bl:.4f}, sharpness={sharpness_bl:.4f}")
    logger.info(f"  +LCM     — loss={current_loss_lcm:.4f}, sharpness={sharpness_lcm:.4f}")
    logger.info(f"  Saved: {output_dir / 'fig5_loss_landscape_1d_comparison.png'}")
    logger.info(f"  Saved: {output_dir / 'fig5_loss_landscape_2d.png'}")
    
    return {
        'status': 'completed',
        'baseline': {
            'current_loss': current_loss_bl,
            'sharpness': sharpness_bl
        },
        'with_lcm': {
            'current_loss': current_loss_lcm,
            'sharpness': sharpness_lcm
        }
    }


# ============================================================================
# Main Experiment
# ============================================================================
@ex.automain
def main(_run, _config, _log):
    from config import init_environment
    opt, logger, device = init_environment(ex, _run, _config)
    
    logger.info("=" * 60)
    logger.info("MOTIVATION ANALYSIS FOR CROSS-DOMAIN FSS")
    logger.info("=" * 60)
    
    # Load dataset
    ds_eval, data_loader, num_classes = datasets.load(opt, logger, "test")
    logger.info(f'     ==> {len(ds_eval)} test samples')
    
    # Load model
    model = load_model(opt, logger)
    if opt.exp_id >= 0 or opt.ckpt:
        ckpt = misc.find_snapshot(_run.run_dir.parent, opt.exp_id, opt.ckpt, afs=on_cloud)
        model.load_weights(ckpt, logger, strict=opt.strict)
    model = model.to(device)
    
    results = {}
    
    # Run all analyses
    logger.info("\n" + "=" * 60)
    logger.info("Starting comprehensive motivation analysis...")
    logger.info("=" * 60 + "\n")
    
    # Analysis 1: Feature domain gap
    results['feature_domain_gap'] = analyze_feature_domain_gap(
        model, data_loader, device, opt, logger)
    
    # Analysis 2: Feature norm distribution
    results['feature_norm_distribution'] = analyze_feature_norm_distribution(
        model, data_loader, device, opt, logger)
    
    # Analysis 3: Confidence map quality
    results['confidence_map_quality'] = analyze_confidence_map_quality(
        model, data_loader, device, opt, logger)
    
    # Analysis 4: Multi-level feature similarity
    results['multilevel_feature_similarity'] = analyze_multilevel_feature_similarity(
        model, data_loader, device, opt, logger)
    
    # Analysis 5: Loss landscape
    results['loss_landscape'] = analyze_loss_landscape(
        model, data_loader, device, opt, logger)
    
    # Save summary
    output_dir = Path('output') / str(opt.exp_id) / 'motivation_analysis'
    with open(output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("\nGenerated figures:")
    logger.info("  1. fig1_tsne_target_domain.png - t-SNE visualization")
    logger.info("  2. fig2_feature_norm_distribution.png - Feature norm distributions")
    logger.info("  3. fig3_confidence_quality.png - Confidence map quality analysis")
    logger.info("  4. fig4_feature_similarity.png - Intra/inter-class similarity")
    logger.info("  5. fig5_loss_landscape_1d_comparison.png - 1D landscape: Baseline vs +LCM")
    logger.info("  6. fig5_loss_landscape_2d.png - 2D loss landscape heatmap + contours")
    
    return results


if __name__ == '__main__':
    ex.run_commandline()
