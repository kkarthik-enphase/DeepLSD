"""
convert_rle_to_lines.py

Converts COCO RLE instance segmentation masks to line segment .npy files
that DeepLSD can use for evaluation.

Each output .npy file has shape [N, 2, 2]:
    N line segments, each defined as [[x1, y1], [x2, y2]]

These are derived by:
  1. Decoding each RLE mask → binary mask
  2. Finding contours → simplifying with Douglas-Peucker
  3. Each consecutive polygon vertex pair → one line segment

Output structure:
  <output_dir>/
    <image_stem>.npy   (one per image)

Usage:
    python -m deeplsd.scripts.convert_rle_to_lines \
        --gt_folder  /mnt/harddrive/data/dataset/sam2_format/gt_facetmasks \
        --output_dir /mnt/harddrive/data/dataset/deeplsd_lines \
        --epsilon    2.0 \
        --min_area   200
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


def rle_decode(rle, shape):
    """Decode COCO RLE to binary mask. shape = (H, W)."""
    counts = rle['counts']
    if isinstance(counts, str):
        # compressed RLE string
        from pycocotools import mask as coco_mask
        import numpy as np
        m = coco_mask.decode(rle)
        return m.astype(np.uint8)
    # uncompressed RLE list
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + c] = 1
        pos += c
    return mask.reshape(shape, order='F')


def mask_to_line_segments(binary_mask, epsilon=2.0, min_area=200):
    """
    Extract polygon edge line segments from a binary mask.
    Returns array of shape [N, 2, 2] = N x [[x1,y1],[x2,y2]].
    """
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri / 1000.0, True)
        pts = approx.reshape(-1, 2)  # [M, 2] (x, y)
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            p1 = pts[i].astype(np.float32)
            p2 = pts[(i + 1) % len(pts)].astype(np.float32)
            segments.append([p1, p2])
    if len(segments) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.array(segments, dtype=np.float32)  # [N, 2, 2]


def convert_folder(gt_folder, output_dir, epsilon=2.0, min_area=200):
    gt_folder = Path(gt_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(gt_folder.glob('*.json'))
    if len(json_files) == 0:
        raise ValueError(f'No .json annotation files found in {gt_folder}')

    total_segs = 0
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Support two layouts:
        # Layout A: {"annotations": [...], "images": [...]}  (COCO style)
        # Layout B: flat list of annotation dicts directly
        if isinstance(data, dict) and 'annotations' in data:
            annotations = data['annotations']
            # build image id -> size map if available
            id2size = {}
            for img_info in data.get('images', []):
                id2size[img_info['id']] = (img_info['height'], img_info['width'])
        elif isinstance(data, list):
            annotations = data
            id2size = {}
        else:
            annotations = [data]
            id2size = {}

        # Group annotations by image
        from collections import defaultdict
        by_image = defaultdict(list)
        for ann in annotations:
            img_id = ann.get('image_id', json_path.stem)
            by_image[img_id].append(ann)

        for img_id, anns in by_image.items():
            all_segments = []
            for ann in anns:
                seg = ann.get('segmentation', {})
                if not seg:
                    continue
                # Determine mask shape
                if 'size' in seg:
                    shape = tuple(seg['size'])  # [H, W]
                elif img_id in id2size:
                    shape = id2size[img_id]
                else:
                    continue
                try:
                    binary_mask = rle_decode(seg, shape)
                except Exception:
                    continue
                segs = mask_to_line_segments(binary_mask, epsilon, min_area)
                if len(segs) > 0:
                    all_segments.append(segs)

            if all_segments:
                combined = np.concatenate(all_segments, axis=0)
            else:
                combined = np.zeros((0, 2, 2), dtype=np.float32)

            out_name = str(img_id) if not str(img_id).endswith('.npy') else img_id
            out_path = output_dir / (Path(out_name).stem + '.npy')
            np.save(str(out_path), combined)
            total_segs += len(combined)

    print(f'Converted {len(json_files)} annotation files → {output_dir}')
    print(f'Total line segments extracted: {total_segs}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', required=True,
                        help='Folder containing COCO RLE JSON annotation files')
    parser.add_argument('--output_dir', required=True,
                        help='Output folder for .npy line segment files')
    parser.add_argument('--epsilon', type=float, default=2.0,
                        help='Douglas-Peucker epsilon (default 2.0)')
    parser.add_argument('--min_area', type=int, default=200,
                        help='Minimum mask area in pixels to keep (default 200)')
    args = parser.parse_args()
    convert_folder(args.gt_folder, args.output_dir, args.epsilon, args.min_area)


if __name__ == '__main__':
    main()
