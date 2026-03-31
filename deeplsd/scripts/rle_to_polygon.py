"""
Convert COCO RLE masks to polygons and visualize both.

Usage:
    python -m deeplsd.scripts.rle_to_polygon \
        --rle_json /path/to/rle_annotation.json \
        --image /path/to/image.jpg \
        --output_dir /path/to/output

    # Batch mode: process all RLE JSONs in a folder
    python -m deeplsd.scripts.rle_to_polygon \
        --rle_dir /path/to/rle_folder \
        --image_dir /path/to/image_folder \
        --output_dir /path/to/output
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


def rle_decode(rle):
    """Decode COCO RLE to binary mask (H x W)."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        try:
            from pycocotools import mask as coco_mask
            m = coco_mask.decode(rle)
            return m.astype(np.uint8)
        except ImportError:
            counts = _decompress_rle(counts, h * w)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, cnt in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + cnt] = 1
        pos += cnt
    return flat.reshape((h, w), order="F").astype(np.uint8)


def _decompress_rle(s, n):
    """Decompress COCO compressed RLE string to list of counts."""
    counts = []
    m = 0
    p = 0
    k = 0
    for i, c in enumerate(s):
        v = ord(c) - 48
        more = v & 0x20
        v &= 0x1F
        m |= v << (5 * p)
        p += 1
        if not more:
            if k & 1:
                m = -m if m & 1 else m
                m = (m >> 1) + (-(m & 1))
            else:
                m = m >> 1
            counts.append(m)
            m = 0
            p = 0
            k += 1
    # Convert delta-encoded to absolute counts
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]
    return counts


def mask_to_polygon(binary_mask, epsilon_factor=0.002):
    """
    Convert binary mask to polygon using contour detection + Douglas-Peucker.
    Returns list of polygons, each is Nx2 array of (x, y) points.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
        pts = approx.reshape(-1, 2)
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def process_single(rle_json_path, image_path, output_dir, epsilon_factor=0.002):
    """Process a single RLE JSON and its corresponding image."""
    with open(rle_json_path, 'r') as f:
        data = json.load(f)

    stem = Path(rle_json_path).stem

    # Load image
    if image_path and os.path.exists(image_path):
        img = cv2.imread(str(image_path))
    else:
        h = data["image"]["height"]
        w = data["image"]["width"]
        img = np.zeros((h, w, 3), dtype=np.uint8)

    h, w = img.shape[:2]
    annotations = data.get("annotations", [])

    # Generate colors
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(max(len(annotations), 1), 3)).tolist()

    # --- Mask visualization ---
    mask_vis = img.copy()
    for idx, ann in enumerate(annotations):
        binary_mask = rle_decode(ann["segmentation"])
        color = colors[idx % len(colors)]
        mask_overlay = np.zeros_like(img)
        mask_overlay[binary_mask == 1] = color
        mask_vis = cv2.addWeighted(mask_vis, 1.0, mask_overlay, 0.5, 0)
        # Draw mask boundary
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_vis, contours, -1, color, 2)

    # --- Polygon visualization ---
    poly_vis = img.copy()
    all_polygons = {}
    for idx, ann in enumerate(annotations):
        binary_mask = rle_decode(ann["segmentation"])
        polygons = mask_to_polygon(binary_mask, epsilon_factor)
        color = colors[idx % len(colors)]
        all_polygons[ann["id"]] = []
        for poly in polygons:
            pts = poly.astype(np.int32)
            cv2.polylines(poly_vis, [pts], isClosed=True, color=color, thickness=2)
            # Draw vertices
            for pt in pts:
                cv2.circle(poly_vis, tuple(pt), 3, color, -1)
            all_polygons[ann["id"]].append(pts.tolist())

    # --- Side by side ---
    combined = np.hstack([mask_vis, poly_vis])

    # Add labels
    cv2.putText(combined, "Masks", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Polygons", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_masks.jpg"), mask_vis)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_polygons.jpg"), poly_vis)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_combined.jpg"), combined)

    # Save polygon JSON
    poly_json = {
        "image": data["image"],
        "polygons": [
            {"id": ann_id, "vertices": verts}
            for ann_id, verts in all_polygons.items()
        ]
    }
    with open(Path(output_dir) / f"{stem}_polygons.json", 'w') as f:
        json.dump(poly_json, f, indent=2)

    print(f"  {stem}: {len(annotations)} masks -> "
          f"{sum(len(v) for v in all_polygons.values())} polygons")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_json', type=str, help='Single RLE JSON file')
    parser.add_argument('--image', type=str, help='Corresponding image file')
    parser.add_argument('--rle_dir', type=str, help='Directory of RLE JSONs (batch mode)')
    parser.add_argument('--image_dir', type=str, help='Directory of images (batch mode)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--epsilon', type=float, default=0.002,
                        help='Douglas-Peucker epsilon factor (default 0.002)')
    args = parser.parse_args()

    if args.rle_json:
        process_single(args.rle_json, args.image, args.output_dir, args.epsilon)
    elif args.rle_dir:
        rle_dir = Path(args.rle_dir)
        image_dir = Path(args.image_dir) if args.image_dir else None
        json_files = sorted(rle_dir.glob('*.json'))
        print(f"Processing {len(json_files)} RLE files...")
        for json_path in json_files:
            img_path = None
            if image_dir:
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = image_dir / (json_path.stem + ext)
                    if candidate.exists():
                        img_path = str(candidate)
                        break
            process_single(str(json_path), img_path, args.output_dir, args.epsilon)
    else:
        parser.error("Provide either --rle_json or --rle_dir")


if __name__ == '__main__':
    main()
