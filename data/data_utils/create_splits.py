"""
Dataset Split Generator and Organizer for Train/Validation/Test
---------------------------------------------------------------

This module handles the creation and organization of dataset splits
for a corpus exported from Transkribus.

Features:
    ✓ Lists all JPG images in ground_truth_data/jpg/
    ✓ Shuffles files with a fixed seed for reproducibility
    ✓ Creates train/validation/test splits (default 80/10/10)
    ✓ Saves the splits in JSON format (splits.json)
    ✓ Organizes all files (images, txt, xml, box files, YOLO labels)
      into the corresponding split folders

USAGE:
    - This script should be run AFTER data preparation and XML parsing scripts.
    - Make sure images exist in:
          data/<CORPUS_NAME>/ground_truth_data/jpg/

USER:
    Set your corpus name in the main script before calling functions.
"""

import os
import random
import json
import shutil


def make_splits(corpus_name, seed=42, train_ratio=0.8, val_ratio=0.1):
    """
    Generate train/validation/test splits for the dataset.

    Parameters:
        corpus_name (str): Name of the corpus folder.
        seed (int): Random seed for reproducibility.
        train_ratio (float): Fraction of images for the training set.
        val_ratio (float): Fraction of images for the validation set.

    Output:
        Creates a JSON file (splits.json) in the corpus folder containing:
        - train: list of training filenames (without extensions)
        - val: list of validation filenames
        - test: list of testing filenames
    """
    jpg_folder = os.path.join(corpus_name, 'ground_truth_data', "jpg")
    output_split_file = os.path.join(corpus_name, "splits.json")

    # List all JPG images in the folder
    all_files = [os.path.splitext(f)[0] for f in os.listdir(jpg_folder) if f.lower().endswith(".jpg")]

    # Shuffle files consistently
    random.seed(seed)
    random.shuffle(all_files)

    # Determine split sizes
    n_total = len(all_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Assign files to splits
    splits = {
        "seed": seed,
        "train": all_files[:n_train],
        "val": all_files[n_train:n_train + n_val],
        "test": all_files[n_train + n_val:]
    }

    # Save splits to JSON
    with open(output_split_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4, ensure_ascii=False)

    print(f"\tSplits saved to {output_split_file}")
    print(f"\tTrain: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    return splits

def organize_splits(corpus_name, splits=None):
    """
    Copy all dataset files into folders according to the splits.

    Parameters:
        corpus_name (str): Name of the corpus folder.
        splits (dict, optional): Preloaded splits dictionary. If None, it loads from splits.json.

    Folder structure created:
        <CORPUS_NAME>/
            train/
                images/, txt/, xml/, box_lines/, yolo_labels/, tiff/
            val/
                ...
            test/
                ...
    """
    ground_truth = os.path.join(corpus_name, "ground_truth_data")
    split_file = os.path.join(corpus_name, "splits.json")

    # Load splits if not provided
    if splits is None:
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)

    # Source folders
    src_folders = {
        "images": os.path.join(ground_truth, "jpg"),
        "tiff": os.path.join(ground_truth, "tiff"),
        "txt": os.path.join(ground_truth, "txt"),
        "xml": os.path.join(ground_truth, "xml"),
        "box_lines": os.path.join(ground_truth, "box_lines"),
        "yolo_labels": os.path.join(ground_truth, "yolo_labels")
    }

    # Create destination folders for each split
    for split in ["train", "val", "test"]:
        for folder in src_folders.keys():
            os.makedirs(os.path.join(corpus_name, split, folder), exist_ok=True)

    # Helper function to copy files
    def copy_files(file_list, split_name):
        for base_name in file_list:
            for folder, src_path in src_folders.items():
                ext = ".jpg" if folder == "images" else ".tiff" if folder == "tiff" else ".txt" if folder in ["txt", "yolo_labels"] else ".xml" if folder == "xml" else ".box"
                src_file = os.path.join(src_path, base_name + ext)
                dst_file = os.path.join(corpus_name, split_name, folder, base_name + ext)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)

    # Copy files to each split
    copy_files(splits["train"], "train")
    copy_files(splits["val"], "val")
    copy_files(splits["test"], "test")

    print("\tFiles organized according to splits.json")