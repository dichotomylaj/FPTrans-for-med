import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from torch.hub import download_url_to_file

from constants import pretrained_weights, model_urls
from core.losses import get as get_loss
from networks import vit
from utils_.misc import interpb, interpn


class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = interpb(x, (h * self.up, w * self.up))
        x = x_up + self.layers(x)
        return x


class FPTrans(nn.Module):
    def __init__(self, opt, logger):
        super(FPTrans, self).__init__()
        self.opt = opt
        self.logger = logger
        self.shot = opt.shot
        self.drop_dim = opt.drop_dim
        self.drop_rate = opt.drop_rate
        self.drop2d_kwargs = {'drop_prob': opt.drop_rate, 'block_size': opt.block_size}

        # Check existence.
        pretrained = self.get_or_download_pretrained(opt.backbone, opt.tqdm)

        # Main model
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', vit.vit_model(opt.backbone,
                                       opt.height,
                                       pretrained=pretrained,
                                       num_classes=0,
                                       opt=opt,
                                       logger=logger))
        ]))
        embed_dim = vit.vit_factory[opt.backbone]['embed_dim']
        self.purifier = self.build_upsampler(embed_dim)
        self.use_dfn = getattr(opt, 'DFN', False)
        self.__class__.__name__ = f"FPTrans/{opt.backbone}"

        # Pretrained model
        self.original_encoder = vit.vit_model(opt.backbone,
                                              opt.height,
                                              pretrained=pretrained,
                                              num_classes=0,
                                              opt=opt,
                                              logger=logger,
                                              original=True)
        for var in self.original_encoder.parameters():
            var.requires_grad = False

        # Define pair-wise loss
        self.pairwise_loss = get_loss(opt, logger, loss='pairwise')
        # Background sampler
        self.bg_sampler = np.random.RandomState(1289)

        if self.use_dfn and getattr(opt, 'dfn_freeze_base', False):
            self.freeze_base_for_dfn()

        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} created")
        if self.use_dfn:
            logger.info(' ' * 5 + "==> DFN enabled")

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        ))

    def forward(self, x, s_x, s_y, y=None, out_shape=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            [B, C, H, W], query image
        s_x: torch.Tensor
            [B, S, C, H, W], support image
        s_y: torch.Tensor
            [B, S, H, W], support mask
        y: torch.Tensor
            [B, 1, H, W], query mask, used for calculating the pair-wise loss
        out_shape: list
            The shape of the output predictions. If not provided, it is default
            to the last two dimensions of `y`. If `y` is also not provided, it is
            default to the [opt.height, opt.width].

        Returns
        -------
        output: dict
            'out': torch.Tensor
                logits that predicted by feature proxies
            'out_prompt': torch.Tensor
                logits that predicted by prompt proxies
            'loss_pair': float
                pair-wise loss
        """
        B, S, C, H, W = s_x.size()
        img_cat = torch.cat((s_x, x.view(B, 1, C, H, W)), dim=1).view(B*(S+1), C, H, W)

        # Calculate class-aware prompts
        with torch.no_grad():
            inp = s_x.view(B * S, C, H, W)
            # Forward
            sup_feat = self.original_encoder(inp)['out']
            _, c, h0, w0 = sup_feat.shape
            sup_mask = interpn(s_y.view(B*S, 1, H, W), (h0, w0))                                # [BS, 1, h0, w0]
            sup_mask_fg = (sup_mask == 1).float()                                               # [BS, 1, h0, w0]
            # Calculate fg and bg tokens
            fg_token = (sup_feat * sup_mask_fg).sum((2, 3)) / (sup_mask_fg.sum((2, 3)) + 1e-6)
            fg_token = fg_token.view(B, S, c).mean(1, keepdim=True)  # [B, 1, c]
            bg_token = self.compute_multiple_prototypes(
                self.opt.bg_num,
                sup_feat.view(B, S, c, h0, w0),
                sup_mask == 0,
                self.bg_sampler
            ).transpose(1, 2)    # [B, k, c]

        # Forward
        img_cat = (img_cat, (fg_token, bg_token))
        backbone_out = self.encoder(img_cat)

        features = self.purifier(backbone_out['out'])               # [B(S+1), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S+1, c, h, w)                   # [B, S+1, c, h, w]
        sup_fts, qry_fts = features.split([S, 1], dim=1)            # [B, S, c, h, w] / [B, 1, c, h, w]
        sup_mask = interpn(s_y.view(B * S, 1, H, W), (h, w))        # [BS, 1, h, w]

        pred = self.classifier(sup_fts, qry_fts, sup_mask)          # [B, 2, h, w]

        # Test-time prototype refinement using high-confidence query patches
        if not self.training and getattr(self.opt, 'proto_refine', False):
            pred = self.refine_prototype(
                pred, sup_fts, qry_fts, sup_mask,
                threshold=getattr(self.opt, 'proto_refine_threshold', 0.9),
                alpha=getattr(self.opt, 'proto_refine_alpha', 0.5)
            )

        # Test-time boundary propagation
        if not self.training and getattr(self.opt, 'boundary_prop', False):
            pred = self.boundary_propagation(pred, sup_fts, qry_fts, sup_mask)

        # LCM: test-time low-level calibration (LoEC)
        if not self.training and getattr(self.opt, 'lcm', False):
            qry_fts_flat = qry_fts.reshape(B, -1, h, w)             # [B, c, h, w]
            pred = self.lcm_calibrate(pred, qry_fts_flat,
                                      K=getattr(self.opt, 'lcm_K', 3),
                                      w=getattr(self.opt, 'lcm_w', 0.6),
                                      beta=getattr(self.opt, 'lcm_beta', 0.7))

        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)
        out = interpb(pred, out_shape)    # [BQ, 2, *, *]
        output = dict(out=out)

        if self.training and y is not None:
            # Pairwise loss
            x1 = sup_fts.flatten(3)                 # [B, S, C, N]
            y1 = sup_mask.view(B, S, -1).long()     # [B, S, N]
            x2 = qry_fts.flatten(3)                 # [B, 1, C, N]
            y2 = interpn(y.float(), (h, w)).flatten(2).long()   # [B, 1, N]
            output['loss_pair'] = self.pairwise_loss(x1, y1, x2, y2)

            # Prompt-Proxy prediction
            fg_token = self.purifier(backbone_out['tokens']['fg'])[:, :, 0, 0]        # [B, c]
            bg_token = self.purifier(backbone_out['tokens']['bg'])[:, :, 0, 0]        # [B, c]
            bg_token = bg_token.view(B, self.opt.bg_num, c).transpose(1, 2)     # [B, c, k]
            pred_prompt = self.compute_similarity(fg_token, bg_token, qry_fts.reshape(-1, c, h, w))

            # Up-sampling
            pred_prompt = interpb(pred_prompt, (H, W))
            output['out_prompt'] = pred_prompt

        return output

    def freeze_base_for_dfn(self):
        """Freeze pretrained ViT backbone weights; keep DFN, prompt_tokens, purifier trainable."""
        frozen_count = 0
        trainable_count = 0
        for name, var in self.named_parameters():
            if name.startswith('encoder.backbone.'):
                if 'dfn' in name or 'prompt_tokens' in name:
                    # DFN adapters and prompt_tokens must remain trainable
                    var.requires_grad = True
                    trainable_count += 1
                else:
                    # Freeze pretrained ViT weights (patch_embed, blocks, norm, cls_token, pos_embed)
                    var.requires_grad = False
                    frozen_count += 1
            else:
                # purifier, pairwise_loss etc. remain trainable
                var.requires_grad = True
                trainable_count += 1
        self.logger.info(' ' * 5 + f"==> DFN freeze mode: {frozen_count} params frozen, "
                         f"{trainable_count} param groups trainable (DFN + prompt_tokens + purifier)")

    def classifier(self, sup_fts, qry_fts, sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        # FG proxies
        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
        # Merge multiple shots
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1)    # [B, c]

        # BG proxies
        bg_proto = self.compute_multiple_prototypes(self.opt.bg_num, sup_fts, sup_mask == 0, self.bg_sampler)

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)
        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)   # [B, 2, h, w]
        return pred

    def refine_prototype(self, pred, sup_fts, qry_fts, sup_mask, threshold=0.9, alpha=0.5):
        """
        Test-time prototype refinement using high-confidence query patches.
        
        Refines the foreground prototype by incorporating high-confidence query regions.
        
        Parameters
        ----------
        pred: torch.Tensor
            [B, 2, h, w], initial prediction
        sup_fts: torch.Tensor
            [B, S, c, h, w], support features
        qry_fts: torch.Tensor
            [B, 1, c, h, w], query features
        sup_mask: torch.Tensor
            [BS, 1, h, w], support mask
        threshold: float
            Confidence threshold for selecting query patches
        alpha: float
            Blending weight for query prototype (1-alpha for support prototype)
        
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], refined prediction
        """
        B, S, c, h, w = sup_fts.shape
        
        # 1. Compute original fg_proto from support
        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)
        fg_proto_support = fg_vecs.view(B, S, c).mean(dim=1)  # [B, c]
        
        # 2. Select high-confidence query patches
        confidence = pred[:, 1] - pred[:, 0]  # [B, h, w]
        confident_mask = (confidence > threshold).float()  # [B, h, w]
        
        # 3. Compute fg_proto from confident query patches
        qry_fts_flat = qry_fts.squeeze(1)  # [B, c, h, w]
        qry_fts_reshape = qry_fts_flat.reshape(B, c, -1)  # [B, c, hw]
        confident_mask_flat = confident_mask.reshape(B, -1)  # [B, hw]
        
        # Check if we have enough confident patches
        confident_count = confident_mask_flat.sum(dim=1)  # [B]
        if confident_count.min() < 10:  # Need at least 10 patches
            return pred  # Return original prediction
        
        # Compute query prototype
        fg_proto_query = torch.sum(qry_fts_reshape * confident_mask_flat.unsqueeze(1), dim=-1) / (confident_count.unsqueeze(1) + 1e-5)  # [B, c]
        
        # 4. Blend support and query prototypes
        fg_proto_refined = alpha * fg_proto_query + (1 - alpha) * fg_proto_support
        
        # 5. Recompute prediction with refined prototype
        bg_proto = self.compute_multiple_prototypes(self.opt.bg_num, sup_fts, sup_mask == 0, self.bg_sampler)
        qry_fts_for_sim = qry_fts_flat  # [B, c, h, w]
        pred = self.compute_similarity(fg_proto_refined, bg_proto, qry_fts_for_sim)  # [B, 2, h, w]
        
        return pred

    def boundary_propagation(self, pred, sup_fts, qry_fts, sup_mask):
        """
        Test-time boundary propagation: use high-confidence patches as seeds,
        propagate labels to uncertain boundary regions via feature similarity.

        Parameters
        ----------
        pred: torch.Tensor
            [B, 2, h, w], initial prediction (logits or probabilities)
        sup_fts: torch.Tensor
            [B, S, c, h, w], support features
        qry_fts: torch.Tensor
            [B, 1, c, h, w], query features
        sup_mask: torch.Tensor
            [BS, 1, h, w], support mask

        Returns
        -------
        pred_refined: torch.Tensor
            [B, 2, h, w], refined prediction
        """
        opt = self.opt
        fg_thresh = getattr(opt, 'boundary_fg_thresh', 0.8)
        bg_thresh = getattr(opt, 'boundary_bg_thresh', 0.8)
        sim_temperature = getattr(opt, 'boundary_temperature', 10.0)
        blend_alpha = getattr(opt, 'boundary_blend', 0.5)

        B, S, c, h, w = sup_fts.shape
        qry_fts_flat = qry_fts.reshape(B, c, h, w)  # [B, c, h, w]

        # Convert logits to probabilities
        probs = torch.softmax(pred, dim=1)  # [B, 2, h, w]
        fg_prob = probs[:, 1]  # [B, h, w]
        bg_prob = probs[:, 0]  # [B, h, w]

        # Identify seed patches
        fg_seed_mask = (fg_prob > fg_thresh).float()  # [B, h, w]
        bg_seed_mask = (bg_prob > bg_thresh).float()  # [B, h, w]

        # Check if we have enough seeds
        fg_seed_count = fg_seed_mask.view(B, -1).sum(dim=1)  # [B]
        bg_seed_count = bg_seed_mask.view(B, -1).sum(dim=1)  # [B]
        if (fg_seed_count.min() < 5) or (bg_seed_count.min() < 5):
            return pred  # Not enough seeds, return original

        # Extract seed features
        fg_seed_feats = qry_fts_flat * fg_seed_mask.unsqueeze(1)  # [B, c, h, w]
        bg_seed_feats = qry_fts_flat * bg_seed_mask.unsqueeze(1)  # [B, c, h, w]

        # Reshape for patch-wise similarity computation
        qry_patches = qry_fts_flat.permute(0, 2, 3, 1).reshape(B, h * w, c)  # [B, N, c]
        fg_patches = fg_seed_feats.permute(0, 2, 3, 1).reshape(B, h * w, c)  # [B, N, c]
        bg_patches = bg_seed_feats.permute(0, 2, 3, 1).reshape(B, h * w, c)  # [B, N, c]

        # Normalize for cosine similarity
        qry_norm = F.normalize(qry_patches, dim=-1)  # [B, N, c]
        fg_norm = F.normalize(fg_patches, dim=-1)    # [B, N, c]
        bg_norm = F.normalize(bg_patches, dim=-1)    # [B, N, c]

        # Max similarity to any fg/bg seed
        sim_fg = torch.bmm(qry_norm, fg_norm.transpose(1, 2))  # [B, N, N]
        sim_bg = torch.bmm(qry_norm, bg_norm.transpose(1, 2))  # [B, N, N]
        max_sim_fg = sim_fg.max(dim=2)[0]  # [B, N]
        max_sim_bg = sim_bg.max(dim=2)[0]  # [B, N]

        # Soft vote via softmax with temperature
        votes = torch.stack([max_sim_bg, max_sim_fg], dim=1)  # [B, 2, N]
        votes = votes * sim_temperature
        prop_probs = torch.softmax(votes, dim=1)  # [B, 2, N]
        prop_probs = prop_probs.view(B, 2, h, w)

        # Blend propagated probs with original probs
        pred_refined = blend_alpha * prop_probs + (1 - blend_alpha) * probs
        pred_refined = torch.log(pred_refined.clamp(min=1e-7))  # Convert back to logits

        return pred_refined

    @staticmethod
    def compute_multiple_prototypes(bg_num, sup_fts, sup_bg, sampler):
        """

        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: torch.Tensor
            [B, S, c, h, w], float32
        sup_bg: torch.Tensor
            [BS, 1, h, w], bool
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies

        """
        B, S, c, h, w = sup_fts.shape
        bg_mask = sup_bg.view(B, S, h, w)    # [B, S, h, w]
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []
            for s in range(S):
                bg_mask_i = bg_mask[b, s]     # [h, w]

                # Check if zero
                with torch.no_grad():
                    if bg_mask_i.sum() < bg_num:
                        bg_mask_i = bg_mask[b, s].clone()    # don't change original mask
                        bg_mask_i.view(-1)[:bg_num] = True

                # Iteratively select farthest points as centers of background local regions
                all_centers = []
                first = True
                pts = torch.stack(torch.where(bg_mask_i), dim=1)     # [N, 2]
                for _ in range(bg_num):
                    if first:
                        i = sampler.choice(pts.shape[0])
                        first = False
                    else:
                        dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                        # choose the farthest point
                        i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                    pt = pts[i]   # center y, x
                    all_centers.append(pt)
            
                # Assign bg labels for bg pixels
                dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

                # Compute bg prototypes
                bg_feats = sup_fts[b, s].permute(1, 2, 0)[bg_mask_i]    # [N, c]
                for i in range(bg_num):
                    proto = bg_feats[bg_labels == i].mean(0)    # [c]
                    bg_protos.append(proto)

            bg_protos = torch.stack(bg_protos, dim=1)   # [c, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = torch.stack(batch_bg_protos, dim=0)  # [B, c, k]
        return bg_proto

    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        if len(bg_proto.shape) == 3:    # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar        # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:   # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [B, 2, h, w]

        return pred

    @staticmethod
    def lcm_calibrate(pred, qry_fts, K=3, w=0.6, beta=0.7):
        """Low-level Calibration Module (LCM) from LoEC.

        During testing, calibrates the coarse score map by directly supplementing
        collapsed low-level target-domain information from query features.

        Parameters
        ----------
        pred: torch.Tensor
            [B, 2, h, w], coarse score map (channel 0=bg, channel 1=fg)
        qry_fts: torch.Tensor
            [B, c, h, w], query feature map from the encoder
        K: int
            Number of top-confidence patches to select
        w: float
            Scaling factor for the calibration signal
        beta: float
            Bias term subtracted from similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], calibrated score map
        """
        B, _, h, w_s = pred.shape
        # 1. Confidence map: fg similarity - bg similarity
        C = pred[:, 1] - pred[:, 0]  # [B, h, w]

        # 2. Partition into patches and compute average confidence per patch
        #    For ViT, each spatial position is already a patch token (1x1)
        #    So we just select the top-K positions with highest confidence
        C_flat = C.reshape(B, -1)  # [B, h*w]
        _, topk_indices = C_flat.topk(K, dim=1)  # [B, K]

        # 3. Get features of top-K patches from query feature map
        # qry_fts: [B, c, h, w] -> reshape to [B, c, h*w]
        feats_flat = qry_fts.reshape(B, -1, h * w_s)  # [B, c, h*w]
        # Gather top-K patch features: [B, c, K]
        topk_feats = torch.gather(feats_flat, 2, topk_indices.unsqueeze(1).expand(-1, qry_fts.shape[1], -1))

        # 4. Compute cosine similarity between each top-K patch and all patches
        # feats_flat: [B, c, h*w], topk_feats: [B, c, K]
        # Normalize
        feats_norm = F.normalize(feats_flat, dim=1)   # [B, c, h*w]
        topk_norm = F.normalize(topk_feats, dim=1)     # [B, c, K]
        # Similarity: [B, K, h*w]
        sim = torch.bmm(topk_norm.permute(0, 2, 1), feats_norm)  # [B, K, h*w]

        # 5. Adaptive beta: use mean similarity instead of fixed beta
        #    This ensures calibration signal is centered around zero
        beta_adaptive = sim.mean(dim=[1, 2], keepdim=True)  # [B, 1, 1]

        # 6. Update foreground score with adaptive beta
        calibration = (w * (sim - beta_adaptive)).sum(dim=1)  # [B, h*w]
        calibration = calibration.reshape(B, h, w_s)
        pred = pred.clone()
        pred[:, 1] = pred[:, 1] + calibration

        return pred

    def load_weights(self, ckpt_path, logger, strict=True):
        """

        Parameters
        ----------
        ckpt_path: Path
            path to the checkpoint
        logger
        strict: bool
            strict mode or not

        """
        weights = torch.load(str(ckpt_path), map_location='cpu')
        if "model_state" in weights:
            weights = weights["model_state"]
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        # Update with original_encoder
        weights.update({k: v for k, v in self.state_dict().items() if 'original_encoder' in k})

        model_dict = self.state_dict()

        # Handle pos_embed shape mismatch via 2D interpolation
        if 'encoder.backbone.pos_embed' in weights and 'encoder.backbone.pos_embed' in model_dict:
            ckpt_pos = weights['encoder.backbone.pos_embed']
            model_pos = model_dict['encoder.backbone.pos_embed']
            if ckpt_pos.shape != model_pos.shape:
                # [1, N_ckpt+1, D] -> [1, N_model+1, D]
                cls_token = ckpt_pos[:, :1, :]
                pos_tokens = ckpt_pos[:, 1:, :]
                B, N, D = pos_tokens.shape
                H = W = int(N ** 0.5)
                pos_tokens = pos_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
                H_new = W_new = int((model_pos.shape[1] - 1) ** 0.5)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(H_new, W_new), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(B, -1, D)
                weights['encoder.backbone.pos_embed'] = torch.cat([cls_token, pos_tokens], dim=1)
                logger.info(f'  ==> pos_embed interpolated: {ckpt_pos.shape} -> {weights["encoder.backbone.pos_embed"].shape}')

        # Handle prompt_tokens shape mismatch: tile checkpoint tokens to fill model size
        if 'encoder.backbone.prompt_tokens' in weights and 'encoder.backbone.prompt_tokens' in model_dict:
            ckpt_pt = weights['encoder.backbone.prompt_tokens']
            model_pt = model_dict['encoder.backbone.prompt_tokens']
            if ckpt_pt.shape != model_pt.shape:
                n_ckpt, n_model = ckpt_pt.shape[0], model_pt.shape[0]
                # Tile checkpoint tokens to fill model size
                n_repeats = (n_model + n_ckpt - 1) // n_ckpt  # ceil division
                tiled_pt = ckpt_pt.repeat(n_repeats, 1, 1)[:n_model]
                weights['encoder.backbone.prompt_tokens'] = tiled_pt
                logger.info(f'  ==> prompt_tokens adapted: checkpoint [{n_ckpt},...] -> model [{n_model},...] by tiling')

        self.load_state_dict(weights, strict=False)
        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")

    @staticmethod
    def get_or_download_pretrained(backbone, progress):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')

        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)
        return cached_file

    def get_params_list(self):
        params = []
        for var in self.parameters():
            if var.requires_grad:
                params.append(var)
        return [{'params': params}]
