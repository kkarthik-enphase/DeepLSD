"""
prepare_roof_facet.py

Prepares the roof facet dataset for DeepLSD training:
  1. Reads all images from img_folder
  2. Splits into train/val and copies into DATA_PATH/roof_facet/train|val/
  3. Creates GT output dirs: DATA_PATH/export_datasets/roof_facet_ha/train|val/
  4. Writes image_list_train.txt and image_list_val.txt for homography_adaptation_df

After running this script:

  Step 1 - Generate GT distance/angle fields (on GPU server):
    python -m deeplsd.scripts.homography_adaptation_df \
        /path/to/deeplsd_data/image_list_train.txt \
        /path/to/deeplsd_data/export_datasets/roof_facet_ha/train \
        --num_H 50 --n_jobs 4

    python -m deeplsd.scripts.homography_adaptation_df \
        /path/to/deeplsd_data/image_list_val.txt \
        /path/to/deeplsd_data/export_datasets/roof_facet_ha/val \
        --num_H 50 --n_jobs 4

  Step 2 - Fine-tune DeepLSD from pretrained weights:
    python -m deeplsd.scripts.train deeplsd_roof_facet \
        --conf deeplsd/configs/train_roof_facet.yaml \
        --restore weights/deeplsd_md/checkpoint_best.tar

Usage:
    python -m deeplsd.scripts.prepare_roof_facet \
        --img_folder /mnt/harddrive/data/dataset/sam2_format/rgb_resized \
        --data_path  /mnt/harddrive/data/dataset/deeplsd_data \
        --val_split  0.05
"""

import argparse
import random
import shutil
from pathlib import Path


def prepare(img_folder, data_path, val_split=0.05, seed=42):
    random.seed(seed)

    img_folder = Path(img_folder)
    data_path = Path(data_path)

    exts = {'.jpg', '.jpeg', '.png'}
    all_images = sorted([p for p in img_folder.iterdir()
                         if p.suffix.lower() in exts])
    if len(all_images) == 0:
        raise ValueError(f'No images found in {img_folder}')
    print(f'Found {len(all_images)} images')

    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_split))
    val_images = all_images[:n_val]
    train_images = all_images[n_val:]
    print(f'Split: {len(train_images)} train / {len(val_images)} val')

    for split, imgs in [('train', train_images), ('val', val_images)]:
        out_dir = data_path / 'roof_facet' / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            dst = out_dir / img_path.name
            if not dst.exists():
                shutil.copy2(str(img_path), str(dst))
        print(f'  Copied {len(imgs)} images to {out_dir}')

    for split in ['train', 'val']:
        gt_dir = data_path / 'export_datasets' / 'roof_facet_ha' / split
        gt_dir.mkdir(parents=True, exist_ok=True)

    for split, imgs in [('train', train_images), ('val', val_images)]:
        list_path = data_path / f'image_list_{split}.txt'
        with open(list_path, 'w') as f:
            for img_path in imgs:
                dst = data_path / 'roof_facet' / split / img_path.name
                f.write(str(dst.resolve()) + '\n')
        print(f'  Wrote {list_path}')

    print()
    print('=== Next steps ===')
    print(f'1. Set DATA_PATH = "{data_path}" in deeplsd/settings.py')
    print()
    print('2. Generate GT (distance + angle fields) — run on GPU server:')
    print(f'   python -m deeplsd.scripts.homography_adaptation_df \\')
    print(f'       {data_path}/image_list_train.txt \\')
    print(f'       {data_path}/export_datasets/roof_facet_ha/train \\')
    print(f'       --num_H 50 --n_jobs 4')
    print()
    print(f'   python -m deeplsd.scripts.homography_adaptation_df \\')
    print(f'       {data_path}/image_list_val.txt \\')
    print(f'       {data_path}/export_datasets/roof_facet_ha/val \\')
    print(f'       --num_H 50 --n_jobs 4')
    print()
    print('3. Fine-tune DeepLSD:')
    print('   python -m deeplsd.scripts.train deeplsd_roof_facet \\')
    print('       --conf deeplsd/configs/train_roof_facet.yaml \\')
    print('       --restore weights/deeplsd_md/checkpoint_best.tar')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True,
                        help='Folder with all roof images')
    parser.add_argument('--data_path', required=True,
                        help='Root data path (will be set as DATA_PATH)')
    parser.add_argument('--val_split', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.img_folder, args.data_path, args.val_split, args.seed)


if __name__ == '__main__':
    main()
