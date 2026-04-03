"""
Roof Facet dataset for DeepLSD fine-tuning.

Data layout:
  image_dir (external):  /mnt/harddrive/data/dataset/sam2_data/train/images/
  gt_dir (DATA_PATH):    gt_rooflines/hdf5/   <- .hdf5 GT from convert_gt_lines_to_hdf5

Images and HDF5 are matched by filename stem.
Train/val split is done by val_size (first N sorted = val, rest = train).
"""

from pathlib import Path
import logging
import cv2
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseDataset, worker_init_fn
from .utils.preprocessing import resize_and_crop
from .utils.homographies import sample_homography, warp_points
from .utils.data_augmentation import photometric_augmentation
from ..settings import DATA_PATH


class RoofFacetHA(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'image_dir': '/mnt/harddrive/data/dataset/sam2_data/train/images',
        'gt_dir': 'gt_rooflines/hdf5',
        'val_size': 500,
        'resize': [512, 512],
        'photometric_augmentation': {
            'enable': True,
            'primitives': [
                'random_brightness', 'random_contrast',
                'additive_speckle_noise', 'additive_gaussian_noise',
                'additive_shade', 'motion_blur'],
            'params': {
                'random_brightness': {'brightness': 0.3},
                'random_contrast': {'strength_range': [0.7, 1.3]},
                'additive_gaussian_noise': {'stddev_range': [0, 20]},
                'additive_speckle_noise': {'prob_range': [0, 0.005]},
                'additive_shade': {
                    'transparency_range': [-0.5, 0.5],
                    'kernel_size_range': [100, 150]
                },
                'motion_blur': {'max_kernel_size': 3}
            }
        },
        'warped_pair': False,
        'homographic_augmentation': True,
        'homography': {
            'params': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.15,
                'perspective_amplitude_x': 0.15,
                'perspective_amplitude_y': 0.15,
                'patch_ratio': 0.9,
                'max_angle': 0.5,
                'allow_artifacts': True
            }
        },
        'seed': 0
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)

    def collate_fn(self, batch):
        """ Customized collate_fn for non-batchable data. """
        batch_keys = ['name', 'image', 'ref_valid_mask', 'df', 'line_level',
                      'offset', 'bg_mask', 'H_ref', 'warped_image',
                      'warped_valid_mask', 'warped_df', 'warped_line_level',
                      'warped_offset', 'warped_bg_mask', 'H']
        list_keys = []
        outputs = {}
        for data_key in batch[0].keys():
            batch_match = sum([_ in data_key for _ in batch_keys])
            list_match = sum([_ in data_key for _ in list_keys])
            if batch_match > 0 and list_match == 0:
                outputs[data_key] = torch.utils.data.dataloader.default_collate(
                    [b[data_key] for b in batch])
            elif batch_match == 0 and list_match > 0:
                outputs[data_key] = [b[data_key] for b in batch]
            elif batch_match == 0 and list_match == 0:
                continue
            else:
                raise ValueError(
                    "A key matches batch keys and list keys simultaneously.")
        return outputs

    def get_data_loader(self, split, shuffle=False):
        """Return a data loader for a given split."""
        assert split in ['train', 'val', 'test']
        batch_size = self.conf.get(split + '_batch_size')
        num_workers = self.conf.get('num_workers', batch_size)
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=shuffle or split == 'train',
                          pin_memory=True, num_workers=num_workers,
                          worker_init_fn=worker_init_fn,
                          collate_fn=self.collate_fn)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.conf, self.split = conf, split
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

        # Build matched pairs: image <-> hdf5 by filename stem
        img_dir = Path(conf.image_dir)
        gt_dir = Path(DATA_PATH, conf.gt_dir)

        gt_stems = {p.stem: p for p in gt_dir.iterdir() if p.suffix == '.hdf5'}
        pairs = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            if img_path.stem in gt_stems:
                pairs.append((img_path, gt_stems[img_path.stem]))

        if len(pairs) == 0:
            raise ValueError(
                f'No matched image-GT pairs found.\n'
                f'  image_dir: {img_dir}\n  gt_dir: {gt_dir}')

        logging.info(f'[RoofFacetHA] Found {len(pairs)} matched image-GT pairs')

        # Split: first val_size = val, rest = train
        if split == 'val':
            pairs = pairs[:conf.val_size]
        elif split == 'train':
            pairs = pairs[conf.val_size:]

        self.images = [p[0] for p in pairs]
        self.gt = [p[1] for p in pairs]
        logging.info(f'[RoofFacetHA] {split}: {len(self.images)} samples')

    def get_dataset(self, split):
        return self

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = cv2.imread(str(path), 0)
        img_size = np.array(img.shape)
        h, w = img_size

        with h5py.File(str(self.gt[idx]), 'r') as f:
            gt_df = np.array(f['df']).reshape(img_size)
            gt_angle = np.mod(np.array(f['line_level']).reshape(img_size), np.pi)
            gt_closest = np.array(f['closest']).reshape(h, w, 2)[:, :, [1, 0]]
            bg_mask = np.array(f['bg_mask']).reshape(img_size)

        pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w),
                                       indexing='ij'), axis=-1)
        offset = gt_closest - pix_loc

        if self.conf.resize is not None:
            scale = np.amax(np.array(self.conf.resize) / img_size)
            img_size = self.conf.resize
            h, w = img_size
            img = resize_and_crop(img, img_size)
            gt_df = resize_and_crop(gt_df, img_size) * scale
            gt_angle = resize_and_crop(gt_angle, img_size,
                                       interp_mode=cv2.INTER_NEAREST)
            offset = resize_and_crop(offset, img_size,
                                     interp_mode=cv2.INTER_NEAREST) * scale
            bg_mask = resize_and_crop(bg_mask, img_size,
                                      interp_mode=cv2.INTER_NEAREST)

        if self.split == 'test' or self.conf.warped_pair:
            H = sample_homography(img_size, **self.conf.homography.params)
            (warped_img, warp_valid_mask, warped_df, warped_angle,
             warped_offset, warped_bg_mask) = self.warp_data(
                img, gt_df, gt_angle, offset, bg_mask, H)

        if self.conf.homographic_augmentation:
            H_ref = sample_homography(img_size, **self.conf.homography.params)
            img, ref_valid_mask, gt_df, gt_angle, offset, bg_mask = self.warp_data(
                img, gt_df, gt_angle, offset, bg_mask, H_ref)
        else:
            H_ref = np.eye(3)
            ref_valid_mask = np.ones_like(img)

        config_aug = self.conf.photometric_augmentation
        if config_aug.enable:
            img = photometric_augmentation(img, config_aug)

        img = img.astype(np.float32) / 255.

        data = {
            'name': path.stem,
            'image': torch.tensor(img[None], dtype=torch.float),
            'df': torch.tensor(gt_df, dtype=torch.float),
            'line_level': torch.tensor(gt_angle, dtype=torch.float),
            'offset': torch.tensor(offset, dtype=torch.float),
            'H_ref': torch.tensor(H_ref, dtype=torch.float),
            'ref_valid_mask': torch.tensor(ref_valid_mask, dtype=torch.float),
            'bg_mask': torch.tensor(bg_mask, dtype=torch.float),
        }

        if self.split == 'test' or self.conf.warped_pair:
            if config_aug.enable:
                warped_img = photometric_augmentation(warped_img, config_aug)
            warped_img = warped_img.astype(np.float32) / 255.
            data['warped_image'] = torch.tensor(warped_img[None], dtype=torch.float)
            data['warped_valid_mask'] = torch.tensor(warp_valid_mask, dtype=torch.float)
            data['warped_df'] = torch.tensor(warped_df, dtype=torch.float)
            data['warped_line_level'] = torch.tensor(warped_angle, dtype=torch.float)
            data['warped_offset'] = torch.tensor(warped_offset, dtype=torch.float)
            data['warped_bg_mask'] = torch.tensor(warped_bg_mask, dtype=torch.float)
            data['H'] = torch.tensor(H, dtype=torch.float)

        return data

    def warp_data(self, img, df, angle, offset, mask, H):
        h, w = img.shape[:2]
        warped_img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
        valid_mask = cv2.warpPerspective(np.ones_like(img), H, (w, h),
                                         flags=cv2.INTER_NEAREST).astype(bool)
        pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w),
                                       indexing='ij'), axis=-1)
        closest = pix_loc + offset
        warped_closest = warp_points(closest.reshape(-1, 2), H).reshape(h, w, 2)
        warped_pix_loc = warp_points(pix_loc.reshape(-1, 2), H).reshape(h, w, 2)
        offset_norm = np.linalg.norm(offset, axis=-1)
        zero_offset = offset_norm < 1e-3
        offset_norm[zero_offset] = 1
        scaling = (np.linalg.norm(warped_closest - warped_pix_loc, axis=-1)
                   / offset_norm)
        scaling[zero_offset] = 0
        warped_closest[:, :, 0] = cv2.warpPerspective(
            warped_closest[:, :, 0], H, (w, h), flags=cv2.INTER_NEAREST)
        warped_closest[:, :, 1] = cv2.warpPerspective(
            warped_closest[:, :, 1], H, (w, h), flags=cv2.INTER_NEAREST)
        warped_offset = warped_closest - pix_loc
        warped_df = cv2.warpPerspective(df, H, (w, h), flags=cv2.INTER_LINEAR)
        warped_scaling = cv2.warpPerspective(scaling, H, (w, h), flags=cv2.INTER_LINEAR)
        warped_df *= warped_scaling
        closest = pix_loc + np.stack([np.sin(angle), np.cos(angle)], axis=-1)
        warped_closest = warp_points(closest.reshape(-1, 2), H).reshape(h, w, 2)
        warped_angle = np.mod(np.arctan2(
            warped_closest[:, :, 0] - warped_pix_loc[:, :, 0],
            warped_closest[:, :, 1] - warped_pix_loc[:, :, 1]), np.pi)
        warped_angle = cv2.warpPerspective(warped_angle, H, (w, h),
                                           flags=cv2.INTER_NEAREST)
        warped_mask = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)
        return (warped_img, valid_mask, warped_df, warped_angle,
                warped_offset, warped_mask)

    def offset_to_df_and_angle(self, offset):
        df = np.linalg.norm(offset, axis=-1)
        angle = np.arctan2(offset[:, :, 0], offset[:, :, 1])
        return df, angle
