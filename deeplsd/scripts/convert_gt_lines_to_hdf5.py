"""
Convert GT roofline JSON files to HDF5 format required by DeepLSD training.

Computes distance field (df), line angle (line_level), closest point on line,
and background mask directly from known GT line segments — no LSD detection
or homography adaptation needed.

Usage:
    python -m deeplsd.scripts.convert_gt_lines_to_hdf5 \
        --gt_dir /home/kkarthik/ARD/DeepLSD/data/gt_rooflines/gt_lines \
        --output_dir /home/kkarthik/ARD/DeepLSD/data/gt_rooflines/hdf5 \
        --img_size 512 \
        --n_jobs 8

    # Quick test with 10 files
    python -m deeplsd.scripts.convert_gt_lines_to_hdf5 \
        --gt_dir /home/kkarthik/ARD/DeepLSD/data/gt_rooflines/gt_lines \
        --output_dir /home/kkarthik/ARD/DeepLSD/data/gt_rooflines/hdf5 \
        --img_size 512 \
        --max_samples 10
"""

import os
import json
import argparse
import numpy as np
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """
    Vectorized: compute closest point on segment (x1,y1)-(x2,y2) from (px,py).
    All inputs are scalars or arrays of the same shape.
    Returns: closest_x, closest_y, distance
    """
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    # Handle degenerate segments
    len_sq = np.maximum(len_sq, 1e-10)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return proj_x, proj_y, dist


def compute_df_from_lines(lines, h, w):
    """
    Compute distance field, angle, closest point, and bg_mask from GT lines.
    
    Args:
        lines: Nx4 array of [x1, y1, x2, y2] line segments (in target image space)
        h, w: image dimensions
    
    Returns:
        df: [h, w] distance to closest line
        angle: [h, w] angle of closest line
        closest: [h, w, 2] closest point on a line (row, col format)
        bg_mask: [h, w] background mask (1 = far from lines)
    """
    # Create pixel grid
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Initialize with large distance
    min_dist = np.full((h, w), 1e6, dtype=np.float64)
    closest_x = np.zeros((h, w), dtype=np.float64)
    closest_y = np.zeros((h, w), dtype=np.float64)
    
    # For each line segment, compute distance to all pixels
    for seg in lines:
        x1, y1, x2, y2 = seg
        proj_x, proj_y, dist = point_to_segment_dist(
            cols.astype(np.float64), rows.astype(np.float64),
            x1, y1, x2, y2)
        
        mask = dist < min_dist
        min_dist[mask] = dist[mask]
        closest_x[mask] = proj_x[mask]
        closest_y[mask] = proj_y[mask]
    
    # Distance field
    df = min_dist.astype(np.float32)
    
    # Offset from pixel to closest point
    offset_x = closest_x - cols  # dx
    offset_y = closest_y - rows  # dy
    
    # Angle: direction to closest line point
    # DeepLSD convention: angle = arctan2(row_offset, col_offset) mapped to [0, pi)
    angle = np.mod(np.arctan2(offset_y, offset_x) + np.pi / 2, np.pi).astype(np.float32)
    
    # Closest point in (row, col) format for HDF5
    closest = np.stack([closest_y, closest_x], axis=-1).astype(np.float32)
    
    # Background mask: pixels far from any line
    raster_lines = (df < 1.0).astype(np.uint8)
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    bg_mask = (1 - raster_lines).astype(np.float32)
    
    return df, angle, closest, bg_mask


def process_single(json_path, output_dir, img_size):
    """Convert a single GT JSON to HDF5."""
    stem = Path(json_path).stem
    out_path = Path(output_dir) / f"{stem}.hdf5"
    
    if out_path.exists():
        return  # Skip already processed
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lines = np.array(data["lines"], dtype=np.float64)
    if len(lines) == 0:
        return
    
    # Scale lines from 1024x1024 to img_size x img_size
    scale = img_size / 1024.0
    lines *= scale
    
    # Compute distance field from GT lines
    df, angle, closest, bg_mask = compute_df_from_lines(lines, img_size, img_size)
    
    # Save as HDF5
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("df", data=df.flatten())
        f.create_dataset("line_level", data=angle.flatten())
        f.create_dataset("closest", data=closest.flatten())
        f.create_dataset("bg_mask", data=bg_mask.flatten())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True,
                        help='Directory with GT line JSON files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Target image size for DF computation (default 512)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max files to process (for testing)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = sorted(Path(args.gt_dir).glob("*.json"))
    if args.max_samples:
        json_files = json_files[:args.max_samples]
    
    print(f"Converting {len(json_files)} GT JSONs to HDF5 (img_size={args.img_size})...")

    if args.n_jobs > 1:
        Parallel(n_jobs=args.n_jobs, backend='multiprocessing')(
            delayed(process_single)(str(p), args.output_dir, args.img_size)
            for p in tqdm(json_files))
    else:
        for p in tqdm(json_files):
            process_single(str(p), args.output_dir, args.img_size)

    # Count output files
    n_out = len(list(Path(args.output_dir).glob("*.hdf5")))
    print(f"Done. {n_out} HDF5 files in {args.output_dir}")


if __name__ == '__main__':
    main()
