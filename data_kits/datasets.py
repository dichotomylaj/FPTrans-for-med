import os
import random
import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from constants import data_dir, lists_dir
from data_kits import transformation as tf
from data_kits import voc_coco as pfe
from utils_.misc import load_image

DATA_DIR = {
    "PASCAL": data_dir / "VOCdevkit/VOC2012",
    "COCO": data_dir / "COCO",
}
DATA_LIST = {
    "PASCAL": {
        "train": lists_dir / "pascal/voc_sbd_merge_noduplicate.txt",
        "test": lists_dir / "pascal/val.txt",
        "eval_online": lists_dir / "pascal/val.txt"
    },
    "COCO": {
        "train": lists_dir / "coco/train_data_list.txt",
        "test": lists_dir / "coco/val_data_list.txt",
        "eval_online": lists_dir / "coco/val_data_list.txt"
    },
}
MEAN = [0.485, 0.456, 0.406]    # list, normalization mean in data preprocessing
STD = [0.229, 0.224, 0.225]     # list, normalization std in data preprocessing


def get_train_transforms(opt, height, width):
    supp_transform = tf.Compose([tf.RandomResize(opt.scale_min, opt.scale_max),
                                 tf.RandomRotate(opt.rotate, pad_type=opt.pad_type),
                                 tf.RandomGaussianBlur(),
                                 tf.RandomHorizontallyFlip(),
                                 tf.RandomCrop(height, width, check=True, center=True, pad_type=opt.pad_type),
                                 tf.ToTensor(mask_dtype='float'),   # support mask using float
                                 tf.Normalize(MEAN, STD)], processer=opt.proc)

    query_transform = tf.Compose([tf.RandomResize(opt.scale_min, opt.scale_max),
                                  tf.RandomRotate(opt.rotate, pad_type=opt.pad_type),
                                  tf.RandomGaussianBlur(),
                                  tf.RandomHorizontallyFlip(),
                                  tf.RandomCrop(height, width, check=True, center=True, pad_type=opt.pad_type),
                                  tf.ToTensor(mask_dtype='long'),   # query mask using long
                                  tf.Normalize(MEAN, STD)], processer=opt.proc)

    return supp_transform, query_transform


def get_val_transforms(opt, height, width):
    supp_transform = tf.Compose([tf.Resize(height, width),
                                 tf.ToTensor(mask_dtype='float'),   # support mask using float
                                 tf.Normalize(MEAN, STD)], processer=opt.proc)

    query_transform = tf.Compose([tf.Resize(height, width, do_mask=False),  # keep mask the original size
                                  tf.ToTensor(mask_dtype='long'),   # query mask using long
                                  tf.Normalize(MEAN, STD)], processer=opt.proc)

    return supp_transform, query_transform


# =============================================================================
# ORGANT2 3D Medical Data (NIfTI) - Single Slice Repeated 3 Times
# =============================================================================
class SemData3D(Dataset):
    """3D medical volume dataset for FSS.
    Each case = 1 episode: support = middle foreground slice,
    query = all other foreground slices.
    Labels are binary (0=background, 1=foreground).
    
    NOTE: This version uses single slice repeated 3 times (no 2.5D slab stacking).
    """
    def __init__(self, opt, split, shot, query, data_dir, transform, mode):
        self.opt = opt
        self.shot = shot
        self.query = query
        self.transform = transform
        # Map eval_online to val mode for data splitting
        self.mode = 'val' if mode == 'eval_online' else mode
        self.height = opt.height
        self.width = opt.width
        self.slab_k = getattr(opt, "slab_k", 3)  # kept for compatibility but not used
        self.val_labels = [1]
        self.all_labels = [1]

        # Scan data directory for image/label pairs
        img_dir = os.path.join(data_dir, 'images')
        lab_dir = os.path.join(data_dir, 'labels')
        all_cases = []
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith('.nii.gz'):
                case_id = fname.replace('.nii.gz', '')
                lab_name = fname
                if os.path.exists(os.path.join(lab_dir, lab_name)):
                    all_cases.append((
                        os.path.join(img_dir, fname),
                        os.path.join(lab_dir, lab_name)
                    ))

        # Split cases into train/val/test based on fold
        test_start = getattr(opt, 'test_fold_start', 0)
        test_end = getattr(opt, 'test_fold_end', 4)
        val_start = getattr(opt, 'val_fold_start', 0)
        val_end = getattr(opt, 'val_fold_end', 4)
        
        if mode == 'test':
            self.cases = all_cases[test_start:test_end]
        else:
            # Training cases (excluding test)
            train_cases = all_cases[:test_start] + all_cases[test_end:]
            if mode == 'val':
                # Validation: subset of training cases
                self.cases = train_cases[val_start:val_end]
            else:  # train
                # Training: exclude validation cases
                self.cases = train_cases[:val_start] + train_cases[val_end:]

        # Build episodes
        self.test_episodes = []
        self.train_episodes = []
        self._vol_cache = {}
        if self.mode in ['test', 'val', 'eval_online']:
            self._build_test_episodes()
        elif self.mode == 'train':
            self._build_train_episodes()

    def _build_test_episodes(self):
        for ci in range(len(self.cases)):
            img_p, lab_p = self.cases[ci]
            img_3d = self.load_nii(img_p, is_mask=False).astype(np.float32)
            lab_3d = self.load_nii(lab_p, is_mask=True)
            self._vol_cache[ci] = (img_3d, lab_3d)

            valid_z = np.where(np.any(lab_3d > 0, axis=(1, 2)))[0]
            if len(valid_z) == 0:
                continue
            s_z = int(valid_z[len(valid_z) // 2])
            # Each query slice = 1 episode (support is always the middle fg slice)
            for q_z in valid_z:
                if q_z == s_z:
                    continue
                self.test_episodes.append({
                    'cls': 1, 's_case': ci, 's_z': s_z,
                    'q_case': ci, 'q_z': int(q_z),
                    'case_id': os.path.basename(self.cases[ci][0]).replace('.nii.gz', '')
                })

    def _build_train_episodes(self):
        """Build training episodes: each episode = 1 support + 1 query from same case."""
        for ci in range(len(self.cases)):
            img_p, lab_p = self.cases[ci]
            img_3d = self.load_nii(img_p, is_mask=False).astype(np.float32)
            lab_3d = self.load_nii(lab_p, is_mask=True)
            self._vol_cache[ci] = (img_3d, lab_3d)

            valid_z = np.where(np.any(lab_3d > 0, axis=(1, 2)))[0]
            if len(valid_z) < 2:
                continue
            # Create episodes: each pair of (support, query) slices
            for s_z in valid_z:
                for q_z in valid_z:
                    if s_z == q_z:
                        continue
                    self.train_episodes.append({
                        'cls': 1, 's_case': ci, 's_z': int(s_z),
                        'q_case': ci, 'q_z': int(q_z),
                        'case_id': os.path.basename(self.cases[ci][0]).replace('.nii.gz', '')
                    })

    def load_nii(self, path, is_mask=False):
        data = nib.load(path).get_fdata()
        if is_mask:
            return np.round(data).astype(np.int64)
        # Percentile clipping + min-max normalization
        p_low, p_high = np.percentile(data, [0.5, 99.5])
        data = np.clip(data, p_low, p_high)
        data = (data - p_low) / (p_high - p_low + 1e-8)
        return data

    def _get_data_at_z(self, case_idx, z):
        if case_idx in self._vol_cache:
            img_3d, lab_3d = self._vol_cache[case_idx]
        else:
            img_p, lab_p = self.cases[case_idx]
            img_3d = self.load_nii(img_p, is_mask=False).astype(np.float32)
            lab_3d = self.load_nii(lab_p, is_mask=True)
            self._vol_cache[case_idx] = (img_3d, lab_3d)
        return img_3d, lab_3d, z

    def __getitem__(self, idx):
        if self.mode == 'train':
            ep = self.train_episodes[idx]
        else:
            ep = self.test_episodes[idx]

        # Support: middle foreground slice
        s_i3d, s_l3d, s_z = self._get_data_at_z(ep['s_case'], ep['s_z'])
        s_img = self.make_slab(s_i3d, s_z)  # Now returns single slice repeated 3 times
        s_msk = (s_l3d[s_z] > 0).astype(np.int64)
        s_img = np.stack([self.resize(s, False) for s in s_img], axis=0)
        s_msk = self.resize(s_msk, True)
        sup_rgb = self.zscore(s_img)[None]       # [1, C, H, W]
        sup_msk = s_msk[None].astype(np.float32)  # [1, H, W] float for interpolation

        # Query: single slice
        q_i3d, q_l3d, _ = self._get_data_at_z(ep['q_case'], ep['q_z'])
        q_img = self.make_slab(q_i3d, ep['q_z'])  # Now returns single slice repeated 3 times
        q_msk = (q_l3d[ep['q_z']] > 0).astype(np.int64)
        q_img = np.stack([self.resize(s, False) for s in q_img], axis=0)
        q_msk = self.resize(q_msk, True)
        qry_rgb = self.zscore(q_img)[None]        # [1, C, H, W]
        qry_msk = q_msk[None]                      # [1, H, W]

        ori_h, ori_w = qry_msk.shape[1], qry_msk.shape[2]

        return {
            "sup_rgb": torch.from_numpy(sup_rgb).float(),
            "sup_msk": torch.from_numpy(sup_msk).float(),
            "qry_rgb": torch.from_numpy(qry_rgb).float(),
            "qry_msk": torch.from_numpy(qry_msk).long(),
            "cls": torch.tensor(1, dtype=torch.long),
            "qry_names": [f"{ep['case_id']}_z{ep['q_z']}"],
            "qry_ori_size": torch.tensor([ori_h, ori_w], dtype=torch.long),
        }

    def zscore(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-5)

    def make_slab(self, vol, z):
        """Single slice repeated 3 times (no 2.5D slab stacking)."""
        single_slice = vol[z]
        # Repeat the same slice 3 times to create 3-channel input
        return np.repeat(single_slice[np.newaxis, :, :], 3, axis=0)

    def resize(self, x, is_mask):
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(x, (self.width, self.height), interpolation=interp)

    def reset_sampler(self):
        pass

    def sample_tasks(self):
        pass

    def __len__(self):
        if self.mode in ['test', 'val', 'eval_online']:
            # Apply test_n limit for validation to match training scale
            test_n = getattr(self.opt, 'test_n', 0)
            if test_n > 0 and self.mode in ['val', 'eval_online']:
                return min(test_n, len(self.test_episodes))
            return len(self.test_episodes)
        # Train mode: use train_n to cap episodes per epoch
        train_n = getattr(self.opt, 'train_n', 0)
        if train_n > 0:
            return min(train_n, len(self.train_episodes))
        return len(self.train_episodes)


def load(opt, logger, mode):
    split, shot, query = opt.split, opt.shot, 1
    height, width = opt.height, opt.width

    if mode == "train":
        data_transform = get_train_transforms(opt, height, width)
    elif mode in ["test", "eval_online", "predict"]:
        data_transform = get_val_transforms(opt, height, width)
    else:
        raise ValueError(f'Not supported mode: {mode}. [train|eval_online|test|predict]')

    if opt.dataset == "ORGANT2":
        num_classes = 2
        organt2_dir = getattr(opt, 'organt2_data_dir', '/home/liuyuhan/lianaijia/organt2')
        dataset = SemData3D(opt, split, shot, query,
                            data_dir=organt2_dir,
                            transform=data_transform,
                            mode=mode)
    else:
        if opt.dataset == "PASCAL":
            num_classes = 20
            cache = True
        elif opt.dataset == "COCO":
            num_classes = 80
            cache = False
        else:
            raise ValueError(f'Not supported dataset: {opt.dataset}. [PASCAL|COCO|ORGANT2]')

        dataset = pfe.SemData(opt, split, shot, query,
                              data_root=DATA_DIR[opt.dataset],
                              data_list=DATA_LIST[opt.dataset][mode],
                              transform=data_transform,
                              mode=mode,
                              cache=cache)

    dataloader = DataLoader(dataset,
                            batch_size=opt.bs if mode == 'train' else opt.test_bs,
                            shuffle=True if mode == 'train' else False,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True if mode == 'train' else False)

    logger.info(' ' * 5 + f"==> Data loader {opt.dataset} for {mode}")
    return dataset, dataloader, num_classes


def get_val_labels(opt, mode):
    if opt.dataset == "PASCAL":
        if opt.coco2pascal:
            if opt.split == 0:
                sub_val_list = [1, 4, 9, 11, 12, 15]
            elif opt.split == 1:
                sub_val_list = [2, 6, 13, 18]
            elif opt.split == 2:
                sub_val_list = [3, 7, 16, 17, 19, 20]
            elif opt.split == 3:
                sub_val_list = [5, 8, 10, 14]
            else:
                raise ValueError(f'PASCAL only have 4 splits [0|1|2|3], got {opt.split}')
        else:
            sub_val_list = list(range(opt.split * 5 + 1, opt.split * 5 + 6))
        return sub_val_list
    elif opt.dataset == "COCO":
        if opt.use_split_coco:
            return list(range(opt.split + 1, 81, 4))
        return list(range(opt.split * 20 + 1, opt.split * 20 + 21))
    elif opt.dataset == "ORGANT2":
        return [1]
    else:
        raise ValueError(f'Only support datasets [PASCAL|COCO|ORGANT2], got {opt.dataset}')


def load_p(opt, device):
    supp_t, query_t = get_val_transforms(opt, opt.height, opt.width)
    p = opt.p

    if p.sup and p.qry:
        supp_rgb_path = DATA_DIR[opt.dataset] / "JPEGImages" / f"{p.sup}.jpg"
        supp_lab_path = DATA_DIR[opt.dataset] / "SegmentationClassAug" / f"{p.sup}.png"
        query_rgb_path = DATA_DIR[opt.dataset] / "JPEGImages" / f"{p.qry}.jpg"
        query_lab_path = DATA_DIR[opt.dataset] / "SegmentationClassAug" / f"{p.qry}.png"

        supp_rgb = load_image(supp_rgb_path, 'img', opt.proc)
        _supp_lab = load_image(supp_lab_path, 'lab', opt.proc)
        supp_lab = np.zeros_like(_supp_lab, dtype=_supp_lab.dtype)
        supp_lab[_supp_lab == 255] = 255
        supp_lab[_supp_lab == p.cls] = 1
        query_ori = query_rgb = load_image(query_rgb_path, 'img', opt.proc)
        query_lab = np.zeros(query_rgb.shape[:-1], dtype=_supp_lab.dtype)
        _query_lab = load_image(query_lab_path, 'lab', opt.proc)
        query_lab[_query_lab == 255] = 255
        query_lab[_query_lab == p.cls] = 1

        supp_img, supp_lab, _ = supp_t(supp_rgb, supp_lab)
        query_img, query_lab, _ = query_t(query_rgb, query_lab)

        supp_img = supp_img[None, None].to(device)      # [B, S, 3, H, W]
        supp_lab = supp_lab[None, None].to(device)      # [B, S, H, W]
        query_img = query_img[None].to(device)          # [B, 3, H, W]
        query_lab = query_lab[None].to(device)      # [B, H, W]
    elif p.sup_rgb and p.sup_msk and p.qry_rgb:
        _supp_rgbs = [load_image(x, 'img', opt.proc) for x in p.sup_rgb]
        _supp_labs = [load_image(x, 'lab', opt.proc, mode=cv2.IMREAD_UNCHANGED) for x in p.sup_msk]
        supp_rgbs = []
        supp_labs = []
        for i, _supp_lab in enumerate(_supp_labs):
            if len(_supp_lab.shape) != 2:
                _supp_lab = _supp_lab[:, :, -1]
            supp_lab = np.zeros_like(_supp_lab, dtype=_supp_lab.dtype)
            if p.cls == 255:
                supp_lab[_supp_lab == p.cls] = 1
            else:
                supp_lab[_supp_lab == 255] = 255
                supp_lab[_supp_lab == p.cls] = 1
            supp_img, supp_lab, _ = supp_t(_supp_rgbs[i], supp_lab)
            supp_rgbs.append(supp_img)
            supp_labs.append(supp_lab)
        supp_img = torch.stack(supp_rgbs, dim=0)
        supp_lab = torch.stack(supp_labs, dim=0)

        query_ori = [load_image(x, 'img', opt.proc) for x in p.qry_rgb]
        _query_rgbs = query_ori
        _query_labs = [np.zeros(x.shape[:-1], dtype=_supp_labs[0].dtype) for x in _query_rgbs]
        query_rgbs = []
        for i, _query_lab in enumerate(_query_labs):
            query_img, query_lab, _ = query_t(_query_rgbs[i], _query_lab)
            query_rgbs.append(query_img)
        query_img = torch.stack(query_rgbs, dim=0)

        supp_img = supp_img[None].to(device)            # [B, S, 3, H, W]
        supp_lab = supp_lab[None].to(device)            # [B, S, H, W]
        query_img = query_img.to(device)                # [Q, 3, H, W]
        query_lab = None
    else:
        raise ValueError(f'In the prediction mode, either the [p.sup, p.qry] or the \n'
                         f'[p.sup_rgb, p.sup_msk, p.qry_rgb] should be given. Got \n'
                         f'    p.sup={p.sup}, p.qry={p.qry}\n'
                         f'    p.sup_rgb={p.sup_rgb}, p.sup_msk={p.sup_msk}, p.qry_rgb={p.qry_rgb}')


    return supp_img, supp_lab, query_img, query_lab, query_ori
