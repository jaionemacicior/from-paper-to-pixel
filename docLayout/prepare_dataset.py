"""
prepare_dataset.py

This script prepares a dataset for the docLayout model from a given corpus.
It performs the following steps:

1. Copies images and YOLO labels into a standardized dataset folder.
2. Converts YOLO labels to XYXY format and saves ground truth JSON files per split.
3. Generates a data.yaml file for YOLO-based training.

Usage:
    python docLayout/prepare_dataset.py --corpus <CORPUS_NAME>

Arguments:
    --corpus: Name of the corpus folder inside the 'data/' directory.
"""

import os
import shutil
import random
import json
from PIL import Image
import argparse

random.seed(42)  # Ensure reproducibility

# ------------------------
# Functions
# ------------------------
def copy_files(img_list, images_folder, labels_folder, split, output_dir, id_to_class):
    """
    Copy images and corresponding YOLO label files to the standardized dataset folder.

    Args:
        img_list (list): List of image filenames.
        images_folder (str): Path to source images.
        labels_folder (str): Path to source YOLO label files.
        split (str): Dataset split ('train', 'val', 'test').
        output_dir (str): Destination dataset folder.
    """
    for img_name in img_list:
        # Copy image
        shutil.copy(os.path.join(images_folder, img_name), os.path.join(output_dir, 'images', split, img_name))

        # Copy label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_folder, label_name)
        dst_label = os.path.join(output_dir, 'labels', split, label_name)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: Label for {img_name} not found!")


def yolo_to_xyxy(line, img_w, img_h):
    """
    Convert a YOLO-format label line to XYXY absolute coordinates.

    Args:
        line (str): Line from YOLO label file (cls x_center y_center width height).
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.

    Returns:
        tuple: class_id (int), [x1, y1, x2, y2] (list of float)
    """
    cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())
    cls_id = int(cls_id)
    x_c, y_c, bw, bh = x_c * img_w, y_c * img_h, bw * img_w, bh * img_h
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2
    return cls_id, [x1, y1, x2, y2]


def prepare_dataset(corpus_name):
    """
    Prepare the dataset for the given corpus.
    """
    BASE_DIR = 'docLayout'
    OUTPUT_DIR = os.path.join(BASE_DIR, corpus_name)
    CORPUS_PATH = os.path.join('data', corpus_name)

    # Load class mappings
    with open(os.path.join(CORPUS_PATH, 'classes.json'), 'r') as f:
        class_to_id = json.load(f)
    id_to_class = {v: k for k, v in class_to_id.items()}

    splits = ['train', 'val', 'test']

    for split in splits:
        images_folder = os.path.join(CORPUS_PATH, split, 'images')
        labels_folder = os.path.join(CORPUS_PATH, split, 'yolo_labels')

        # Create output folders
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

        # List images
        imgs = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Copy images and labels
        copy_files(imgs, images_folder, labels_folder, split, OUTPUT_DIR, id_to_class)

        # Convert YOLO labels to ground truth boxes
        gt_boxes_dict = {}
        for img_name in imgs:
            label_file = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(labels_folder, label_file)
            img_path = os.path.join(images_folder, img_name)

            if not os.path.exists(label_path):
                continue

            with Image.open(img_path) as im:
                w, h = im.size

            boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls_id, xyxy = yolo_to_xyxy(line, w, h)
                    boxes.append({
                        "xyxy": xyxy,
                        "class_name": id_to_class.get(cls_id, str(cls_id))
                    })

            gt_boxes_dict[os.path.splitext(img_name)[0]] = boxes

        # Save GT boxes for this split
        gt_json_path = os.path.join(OUTPUT_DIR, 'labels', split, "gt_boxes.json")
        with open(gt_json_path, "w", encoding="utf-8") as f:
            json.dump(gt_boxes_dict, f, indent=4, ensure_ascii=False)

        print(f"Dataset split '{split}' completed!")
        print(f"Ground truth saved at: {gt_json_path}")

    # Generate data.yaml
    num_classes = max(class_to_id.values()) + 1
    class_names = [None] * num_classes
    for cls, idx in class_to_id.items():
        class_names[idx] = cls

    output_yaml = os.path.join(OUTPUT_DIR, 'data.yaml')
    yaml_content = f"""
train: {os.path.join('images', 'train')}
val: {os.path.join('images', 'val')}
nc: {num_classes}
names: {class_names}
"""

    with open(output_yaml, 'w') as f:
        f.write(yaml_content.strip())

    print(f"YOLO data.yaml created at {output_yaml}")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for docLayout model")
    parser.add_argument("--corpus", type=str, required=True, help="Name of the corpus folder inside data/")
    args = parser.parse_args()

    prepare_dataset(args.corpus)
