#!/usr/bin/env python3
"""
Simple utility to remove spaces from filenames inside the `dataset_breast` folder.
Usage examples:
  python scripts/rename_dataset_breast.py                # dry-run on default folder
  python scripts/rename_dataset_breast.py --apply       # actually rename
  python scripts/rename_dataset_breast.py --path mydir --apply --recursive
"""
import os
import argparse


def rename_files(root: str, apply: bool = False, recursive: bool = False) -> None:
    if not os.path.isdir(root):
        print(f"Error: path does not exist or is not a directory: {root}")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if " " not in fname:
                continue
            new_fname = fname.replace(" ", "")
            src = os.path.join(dirpath, fname)
            dst = os.path.join(dirpath, new_fname)

            if os.path.exists(dst):
                print(f"Skipping (target exists): {src} -> {dst}")
                continue

            if apply:
                try:
                    os.rename(src, dst)
                    print(f"Renamed: {src} -> {dst}")
                except Exception as e:
                    print(f"Failed to rename {src} -> {dst}: {e}")
            else:
                print(f"DRY-RUN: {src} -> {dst}")

        if not recursive:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Remove spaces from filenames in a dataset folder")
    p.add_argument("--path", "-p", default="dataset_breast", help="Path to dataset folder (default: dataset_breast)")
    p.add_argument("--apply", "-a", action="store_true", help="Actually perform renames (default: dry-run)")
    p.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
    args = p.parse_args()

    rename_files(args.path, apply=args.apply, recursive=args.recursive)
