"""
CPU fallback version of homography_adaptation_df.py.
Works without the afm_op CUDA extension.
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm
from pytlsd import lsd
from joblib import Parallel, delayed

from ..datasets.utils.homographies import sample_homography, warp_lines
from ..datasets.utils.data_augmentation import random_contrast


homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.2,
    'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2,
    'patch_ratio': 0.85,
    'max_angle': 1.57,
    'allow_artifacts': True,
}


def compute_offset_cpu(lines, h, w):
    """CPU fallback for afm_op: compute distance to closest line segment."""
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)  # [h,w,2]
    offset = np.zeros((h, w, 2), dtype=np.float32)
    
    # For each pixel, find closest point on any line segment
    for i in range(h):
        for j in range(w):
            p = np.array([j, i], dtype=np.float32)  # (x, y)
            min_dist = float('inf')
            closest_pt = p.copy()
            
            for line in lines:
                # line is [[x1, y1], [x2, y2]]
                a, b = line[0], line[1]
                v = b - a
                v_len_sq = np.dot(v, v)
                if v_len_sq < 1e-6:
                    # Degenerate line (point)
                    dist = np.linalg.norm(p - a)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pt = a.copy()
                else:
                    # Project point onto line
                    t = np.dot(p - a, v) / v_len_sq
                    t = np.clip(t, 0.0, 1.0)
                    proj = a + t * v
                    dist = np.linalg.norm(p - proj)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pt = proj.copy()
            
            offset[i, j] = closest_pt - p
    
    return offset


def ha_df(img, num=100, border_margin=10):
    """Run homography adaptation to compute distance function."""
    h, w = img.shape
    size = (w + border_margin * 2, h + border_margin * 2)
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    # Pad the image
    img = cv2.copyMakeBorder(img, border_margin, border_margin,
                            border_margin, border_margin,
                            cv2.BORDER_REPLICATE)
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    raster_lines = np.zeros_like(img)

    df_maps = []
    angles = []
    closests = []
    counts = []

    for i in range(num):
        if i == 0:
            H = np.eye(3)
        else:
            H = sample_homography(img.shape, **homography_params)
        H_inv = np.linalg.inv(H)
        
        warped_img = cv2.warpPerspective(img, H, size, borderMode=cv2.BORDER_REPLICATE)
        warped_lines = lsd(warped_img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        lines = warp_lines(warped_lines, H_inv)
        
        # CPU fallback for distance field computation
        offset = compute_offset_cpu(lines, h, w)
        closest = pix_loc + offset
        df = np.linalg.norm(offset, axis=-1)
        angle = np.mod(np.arctan2(offset[:, :, 0], offset[:, :, 1]) + np.pi / 2, np.pi)
        
        df_maps.append(df)
        angles.append(angle)
        closests.append(closest)
        
        count = cv2.warpPerspective(np.ones_like(img), H_inv, size, flags=cv2.INTER_NEAREST)
        count = cv2.erode(count, kernel)
        counts.append(count)
        raster_lines += (df < 1).astype(np.uint8) * count 
        
    df_maps, angles = np.stack(df_maps), np.stack(angles)
    counts, closests = np.stack(counts), np.stack(closests)
    
    df_maps[counts == 0] = np.nan
    avg_df = np.nanmedian(df_maps, axis=0)

    closests[counts == 0] = np.nan
    avg_closest = np.nanmedian(closests, axis=0)

    circ_bound = (np.minimum(np.pi - angles, angles) * counts).sum(0) / counts.sum(0) < 0.3
    angles[:, circ_bound] -= np.where(angles[:, circ_bound] > np.pi / 2,
                                      np.ones_like(angles[:, circ_bound]) * np.pi,
                                      np.zeros_like(angles[:, circ_bound]))
    angles[counts == 0] = np.nan
    avg_angle = np.mod(np.nanmedian(angles, axis=0), np.pi)

    raster_lines = np.where(raster_lines > num * 0.3, np.ones_like(img), np.zeros_like(img))
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    bg_mask = (1 - raster_lines).astype(float)

    return avg_df, avg_angle, avg_closest[:, :, [1, 0]], bg_mask


def process_image(img_path, randomize_contrast, num_H, output_folder):
    img = cv2.imread(img_path, 0)
    if randomize_contrast is not None:
        img = randomize_contrast(img)
    
    df, angle, closest, bg_mask = ha_df(img, num=num_H)

    out_path = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_folder, out_path) + '.hdf5'
    with h5py.File(out_path, "w") as f:
        f.create_dataset("df", data=df.flatten())
        f.create_dataset("line_level", data=angle.flatten())
        f.create_dataset("closest", data=closest.flatten())
        f.create_dataset("bg_mask", data=bg_mask.flatten())


def export_ha(images_list, output_folder, num_H=100, rdm_contrast=False, n_jobs=1):
    with open(images_list, 'r') as f:
        image_files = f.readlines()
    image_files = [path.strip('\n') for path in image_files]
    
    randomize_contrast = random_contrast() if rdm_contrast else None
    
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(process_image)(
        img_path, randomize_contrast, num_H, output_folder) for img_path in tqdm(image_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('images_list', type=str,
                        help='Path to a txt file containing the image paths.')
    parser.add_argument('output_folder', type=str, help='Output folder.')
    parser.add_argument('--num_H', type=int, default=100,
                        help='Number of homographies used during HA.')
    parser.add_argument('--random_contrast', action='store_true',
                        help='Add random contrast to the images (disabled by default).')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs to run in parallel.')
    args = parser.parse_args()

    export_ha(args.images_list, args.output_folder, args.num_H,
              args.random_contrast, args.n_jobs)
