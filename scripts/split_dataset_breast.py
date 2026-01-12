#!/usr/bin/env python3
"""
Split `dataset_breast` into training (70%) and test (30%).
- Test set will contain only image files (no masks).
- Training set will contain images + their corresponding masks (if present).

Usage:
  python scripts/split_dataset_breast.py           # dry-run, shows planned moves
  python scripts/split_dataset_breast.py --apply   # perform moves (default: move)
  python scripts/split_dataset_breast.py --copy --apply  # copy instead of move
  python scripts/split_dataset_breast.py --path mydir --test-frac 0.25 --seed 42 --recursive

The script identifies mask files by the presence of the words 'mask' or 'annotation' in the filename (case-insensitive).
"""
import os
import argparse
import random
import shutil
from typing import List

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_files(folder: str, recursive: bool = False) -> List[str]:
    files = []
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for f in filenames:
                files.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(folder):
            files.append(os.path.join(folder, f))
    return files


def is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def is_mask(fname: str) -> bool:
    n = os.path.basename(fname).lower()
    return ('mask' in n) or ('annotation' in n)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_dataset(root: str, test_frac: float, apply: bool, copy: bool, recursive: bool, seed: int):
    if not os.path.isdir(root):
        # Try resolving relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(script_dir)
        
        # Try multiple common paths
        candidates = [
            os.path.abspath(root),  # as-is
            os.path.join(script_dir, '..', root),  # parent of scripts/
            os.path.join(workspace_root, root),  # workspace root
            os.path.join(workspace_root, 'notebooks', root),  # notebooks/dataset_breast
        ]
        
        resolved = None
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if os.path.isdir(candidate):
                resolved = candidate
                break
        
        if resolved:
            root = resolved
        else:
            # List available directories to help user
            parent_dir = workspace_root
            available = []
            if os.path.isdir(parent_dir):
                available = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
            raise ValueError(
                f"Path does not exist or is not a directory.\n"
                f"Tried:\n"
                + "\n".join(f"  {c}" for c in candidates) + 
                f"\nAvailable folders in {parent_dir}:\n"
                f"  {', '.join(available) if available else '(none)'}\n"
                f"Please check that dataset_breast folder exists or provide an absolute path."
            )

    all_files = list_files(root, recursive=recursive)
    all_files = [f for f in all_files if is_image(f)]

    # Separate masks and images
    mask_files = [f for f in all_files if is_mask(f)]
    image_files = [f for f in all_files if f not in mask_files]

    if not image_files:
        # Provide diagnostic info
        all_files_in_root = []
        subdirs = {}
        if os.path.isdir(root):
            for item in os.listdir(root):
                item_path = os.path.join(root, item)
                if os.path.isdir(item_path):
                    subdirs[item] = os.listdir(item_path)[:10]  # first 10 items
                else:
                    all_files_in_root.append(item)
        print(f"No image files found in the dataset path: {root}")
        print(f"Files in root:")
        for item in sorted(all_files_in_root)[:20]:  # show first 20
            print(f"  {item}")
        print(f"\nSubdirectories:")
        for subdir, items in sorted(subdirs.items()):
            print(f"  {subdir}/")
            for item in items:
                print(f"    {item}")
        print(f"\nSupported image extensions: {IMAGE_EXTS}")
        print(f"Total image files scanned: {len(all_files)}")
        print(f"\nNote: Run with --recursive flag to search in subdirectories")
        return

    random.seed(seed)
    random.shuffle(image_files)

    n_test = int(len(image_files) * test_frac)
    test_images = set(image_files[:n_test])
    train_images = set(image_files[n_test:])

    # Prepare directories
    train_dir = os.path.join(root, 'train')
    test_img_dir = os.path.join(root, 'test', 'images')

    for d in (train_dir, test_img_dir):
        ensure_dir(d)

    # Helper to move/copy
    def move_or_copy(src, dst):
        if not apply:
            print(f"DRY-RUN: {src} -> {dst}")
            return
        ensure_dir(os.path.dirname(dst))
        if copy:
            shutil.copy2(src, dst)
            print(f"COPIED: {src} -> {dst}")
        else:
            shutil.move(src, dst)
            print(f"MOVED: {src} -> {dst}")

    # Index masks by basename (without extension) for matching
    mask_basenames = {}
    for m in mask_files:
        bn = os.path.splitext(os.path.basename(m))[0]
        mask_basenames.setdefault(bn, []).append(m)

    # Also maintain list of mask files to search by containing image basename
    remaining_masks = set(mask_files)

    # Process train images (along with their masks in the same folder)
    for img in train_images:
        img_base = os.path.splitext(os.path.basename(img))[0]
        dst_img = os.path.join(train_dir, os.path.basename(img))
        move_or_copy(img, dst_img)

        # Find masks that correspond to this image
        candidates = []
        # exact match with appended suffixes
        for suffix in ('_mask', '_annotation', '_annot'):
            key = img_base + suffix
            if key in mask_basenames:
                candidates.extend(mask_basenames[key])
        # fallback: any mask that contains the image basename
        if not candidates:
            for m in list(remaining_masks):
                m_base = os.path.splitext(os.path.basename(m))[0]
                if img_base in m_base:
                    candidates.append(m)
        for m in candidates:
            dst_mask = os.path.join(train_dir, os.path.basename(m))
            move_or_copy(m, dst_mask)
            if not copy and m in remaining_masks:
                remaining_masks.discard(m)

    # Process test images (images only)
    for img in test_images:
        dst_img = os.path.join(test_img_dir, os.path.basename(img))
        move_or_copy(img, dst_img)

    # Report any masks left behind (if move was performed)
    if apply and not copy:
        leftover = list(remaining_masks)
        if leftover:
            print('\nMasks remaining in the root directory:')
            for m in leftover:
                print('  ', m)
    print('\nSummary:')
    print(f'  total images: {len(image_files)}')
    print(f'  train images: {len(train_images)}')
    print(f'  test images:  {len(test_images)}')
    print('  masks moved/copied into train/masks (if found)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Split dataset_breast into train/test and keep masks only in train')
    p.add_argument('--path', '-p', default='dataset_breast', help='Path to dataset folder (default: dataset_breast)')
    p.add_argument('--test-frac', type=float, default=0.3, help='Fraction of images to use for test (default: 0.3)')
    p.add_argument('--apply', action='store_true', help='Actually perform moves/copies (default: dry-run)')
    p.add_argument('--copy', action='store_true', help='Copy files instead of moving')
    p.add_argument('--recursive', action='store_true', help='Recurse into subdirectories')
    p.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = p.parse_args()

    split_dataset(args.path, test_frac=args.test_frac, apply=args.apply, copy=args.copy, recursive=args.recursive, seed=args.seed)
