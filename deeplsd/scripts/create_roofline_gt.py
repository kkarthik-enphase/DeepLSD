"""
Create roofline ground truth for DeepLSD training from payload JSONs.

Steps:
1. Map image files to payload JSONs via project_id
2. Extract roof outline edges (points + edges) from payload
3. Scale coordinates from original image size to 1024x1024
4. Render line segment GT and save as .npy + visualization

Usage:
    # Step 1: Create mapping
    python -m deeplsd.scripts.create_roofline_gt --mode map \
        --image_dir /mnt/harddrive/data/dataset/sam2_data/train/images \
        --payload_dir /mnt/harddrive/data/dataset/all_data_till_dec23_wo_state/payloads \
        --output_dir /mnt/harddrive/data/dataset/sam2_data/train/gt_rooflines

    # Step 2: Generate GT from mapping
    python -m deeplsd.scripts.create_roofline_gt --mode generate \
        --image_dir /mnt/harddrive/data/dataset/sam2_data/train/images \
        --payload_dir /mnt/harddrive/data/dataset/all_data_till_dec23_wo_state/payloads \
        --output_dir /mnt/harddrive/data/dataset/sam2_data/train/gt_rooflines \
        --visualize

    # Step 3: Visualize a few samples
    python -m deeplsd.scripts.create_roofline_gt --mode visualize \
        --image_dir /mnt/harddrive/data/dataset/sam2_data/train/images \
        --output_dir /mnt/harddrive/data/dataset/sam2_data/train/gt_rooflines \
        --num_samples 20
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def extract_project_id_from_image(image_name):
    """
    Image filename: {project_id}_{lat-int}_{lat-dec}_{lon-int}_{lon-dec}.jpg
    Extract project_id (first token before underscore).
    """
    stem = Path(image_name).stem
    parts = stem.split("_")
    if len(parts) >= 1:
        return parts[0]
    return None


def extract_project_id_from_payload(payload_name):
    """
    Payload filename: panel_shading_{project_id}_{project_id}.json
    Extract project_id.
    """
    stem = Path(payload_name).stem
    # Remove 'panel_shading_' prefix
    if stem.startswith("panel_shading_"):
        rest = stem[len("panel_shading_"):]
        # Format is {id}_{id}, take the first one
        parts = rest.split("_")
        if len(parts) >= 1:
            return parts[0]
    return None


def build_mapping(image_dir, payload_dir):
    """Build a mapping from image files to payload JSON files via project_id."""
    print("Building image -> payload mapping...")

    # Index payloads by project_id
    payload_index = {}
    payload_files = list(Path(payload_dir).glob("*.json"))
    print(f"  Found {len(payload_files)} payload files")
    for pf in tqdm(payload_files, desc="  Indexing payloads"):
        pid = extract_project_id_from_payload(pf.name)
        if pid:
            payload_index[pid] = str(pf)

    # Match images to payloads
    image_files = sorted(Path(image_dir).glob("*.jpg"))
    print(f"  Found {len(image_files)} image files")

    mapping = {}
    matched = 0
    unmatched = 0
    for img_path in tqdm(image_files, desc="  Matching images"):
        pid = extract_project_id_from_image(img_path.name)
        if pid and pid in payload_index:
            mapping[img_path.name] = payload_index[pid]
            matched += 1
        else:
            unmatched += 1

    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    return mapping


def extract_rooflines_from_payload(payload_path):
    """
    Extract all roof outline edges as line segments [(x1,y1,x2,y2), ...].
    Also returns edge types and the original image size.
    """
    with open(payload_path, 'r') as f:
        data = json.load(f)

    original_size = data.get("original_img_size", [800, 800])  # [w, h] or [h, w]
    # The imageSize field is the rendered size
    image_size = data.get("imageSize", [800, 800])

    roof_outlines = data.get("roofOutlines", {})
    all_segments = []
    all_types = []

    for outline_id, outline in roof_outlines.items():
        points = outline.get("points", {})
        edges = outline.get("edges", {})

        for edge_id, edge_info in edges.items():
            start_id = edge_info.get("start")
            end_id = edge_info.get("end")
            edge_type = edge_info.get("type", "unknown")
            is_wall = edge_info.get("isWall", False)

            if start_id in points and end_id in points:
                x1 = points[start_id]["x"]
                y1 = points[start_id]["y"]
                x2 = points[end_id]["x"]
                y2 = points[end_id]["y"]
                all_segments.append([x1, y1, x2, y2])
                all_types.append(edge_type)

    return np.array(all_segments) if all_segments else np.zeros((0, 4)), all_types, original_size, image_size


def scale_segments_to_target(segments, original_img_size, target_size=1024):
    """
    Scale line segments from original coordinate space to target_size x target_size.
    original_img_size: [w, h] from the payload JSON
    """
    if len(segments) == 0:
        return segments

    # The point coordinates in the JSON are in the original_img_size space
    orig_w, orig_h = original_img_size[0], original_img_size[1]
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    scaled = segments.copy().astype(np.float64)
    scaled[:, 0] *= scale_x  # x1
    scaled[:, 1] *= scale_y  # y1
    scaled[:, 2] *= scale_x  # x2
    scaled[:, 3] *= scale_y  # y2

    return scaled


def render_line_map(segments, height=1024, width=1024, thickness=1):
    """Render binary line map from segments."""
    line_map = np.zeros((height, width), dtype=np.uint8)
    for seg in segments:
        x1, y1, x2, y2 = seg.astype(int)
        x1 = np.clip(x1, 0, width - 1)
        x2 = np.clip(x2, 0, width - 1)
        y1 = np.clip(y1, 0, height - 1)
        y2 = np.clip(y2, 0, height - 1)
        cv2.line(line_map, (x1, y1), (x2, y2), 255, thickness)
    return line_map


def render_visualization(image, segments, edge_types, thickness=2):
    """Render colored line overlay on image."""
    vis = image.copy()
    type_colors = {
        "eave": (0, 255, 0),      # green
        "ridge": (255, 0, 0),     # blue (BGR)
        "rake": (0, 0, 255),      # red
        "hip": (255, 255, 0),     # cyan
        "valley": (0, 255, 255),  # yellow
        "unknown": (200, 200, 200),
    }

    for seg, etype in zip(segments, edge_types):
        x1, y1, x2, y2 = seg.astype(int)
        x1 = np.clip(x1, 0, vis.shape[1] - 1)
        x2 = np.clip(x2, 0, vis.shape[1] - 1)
        y1 = np.clip(y1, 0, vis.shape[0] - 1)
        y2 = np.clip(y2, 0, vis.shape[0] - 1)
        color = type_colors.get(etype, type_colors["unknown"])
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
        # Draw endpoints
        cv2.circle(vis, (x1, y1), 3, color, -1)
        cv2.circle(vis, (x2, y2), 3, color, -1)

    # Add legend
    y_offset = 30
    for etype, color in type_colors.items():
        if etype == "unknown":
            continue
        cv2.putText(vis, etype, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    return vis


def process_generate(image_dir, payload_dir, output_dir, visualize=False, max_samples=None):
    """Generate GT roofline data for all matched images."""
    mapping = build_mapping(image_dir, payload_dir)

    # Save mapping
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = Path(output_dir) / "image_payload_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {mapping_path}")

    # Create output subdirs
    lines_dir = Path(output_dir) / "line_segments"
    os.makedirs(lines_dir, exist_ok=True)

    if visualize:
        vis_dir = Path(output_dir) / "visualizations"
        os.makedirs(vis_dir, exist_ok=True)

    stats = {"total": 0, "with_lines": 0, "empty": 0, "errors": 0}

    items = list(mapping.items())
    if max_samples:
        items = items[:max_samples]
        print(f"Processing only first {max_samples} samples")

    for img_name, payload_path in tqdm(items, desc="Generating GT"):
        stem = Path(img_name).stem
        stats["total"] += 1

        try:
            # Extract rooflines
            segments, edge_types, orig_size, img_size = \
                extract_rooflines_from_payload(payload_path)

            if len(segments) == 0:
                stats["empty"] += 1
                continue

            # Scale to 1024x1024
            scaled_segments = scale_segments_to_target(segments, orig_size, 1024)

            # Save line segments as .npy (Nx4 array: x1, y1, x2, y2)
            np.save(str(lines_dir / f"{stem}.npy"), scaled_segments.astype(np.float32))

            # Save edge types
            types_path = lines_dir / f"{stem}_types.json"
            with open(types_path, 'w') as f:
                json.dump(edge_types, f)

            # Render binary line map
            line_map = render_line_map(scaled_segments)
            cv2.imwrite(str(lines_dir / f"{stem}_linemap.png"), line_map)

            stats["with_lines"] += 1

            # Optional visualization
            if visualize:
                img_path = Path(image_dir) / img_name
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    vis = render_visualization(img, scaled_segments, edge_types)
                    cv2.imwrite(str(vis_dir / f"{stem}_vis.jpg"), vis)

        except Exception as e:
            stats["errors"] += 1
            print(f"  Error processing {img_name}: {e}")

    print(f"\nStats: {json.dumps(stats, indent=2)}")
    return stats


def process_visualize(image_dir, output_dir, num_samples=20):
    """Visualize a few samples from already generated GT."""
    lines_dir = Path(output_dir) / "line_segments"
    vis_dir = Path(output_dir) / "visualizations_sample"
    os.makedirs(vis_dir, exist_ok=True)

    npy_files = sorted(lines_dir.glob("*.npy"))[:num_samples]
    print(f"Visualizing {len(npy_files)} samples...")

    for npy_path in npy_files:
        stem = npy_path.stem
        segments = np.load(str(npy_path))

        # Load edge types if available
        types_path = lines_dir / f"{stem}_types.json"
        if types_path.exists():
            with open(types_path) as f:
                edge_types = json.load(f)
        else:
            edge_types = ["unknown"] * len(segments)

        # Load image
        img_path = Path(image_dir) / f"{stem}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
        else:
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)

        vis = render_visualization(img, segments, edge_types)

        # Also create side-by-side with line map
        line_map = render_line_map(segments)
        line_map_color = cv2.cvtColor(line_map, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, line_map_color])

        cv2.imwrite(str(vis_dir / f"{stem}_combined.jpg"), combined)
        print(f"  {stem}: {len(segments)} line segments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['map', 'generate', 'visualize'],
                        required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--payload_dir', type=str, default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--visualize', action='store_true',
                        help='Also save visualizations during generate')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples for visualize mode')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max images to process in generate mode (for quick testing)')
    args = parser.parse_args()

    if args.mode == 'map':
        mapping = build_mapping(args.image_dir, args.payload_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = Path(args.output_dir) / "image_payload_mapping.json"
        with open(out_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved mapping ({len(mapping)} entries) to {out_path}")

    elif args.mode == 'generate':
        assert args.payload_dir, "--payload_dir required for generate mode"
        process_generate(args.image_dir, args.payload_dir,
                        args.output_dir, args.visualize, args.max_samples)

    elif args.mode == 'visualize':
        process_visualize(args.image_dir, args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
