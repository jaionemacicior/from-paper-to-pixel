"""
prepare_dataset.py — Granite OCR Dataset Preparation
----------------------------------------------------

This script prepares the dataset for Granite OCR fine-tuning.

It performs:
1. Image resizing to the fixed input size required by Granite Vision models.
2. Padding to preserve aspect ratio.
3. Creation of train/val/test folders inside: granite/<CORPUS_NAME>/data/

Input folder structure (created in Stage 1):
    data/<CORPUS_NAME>/<split>/images/*.jpg
    data/<CORPUS_NAME>/<split>/txt/*.txt

Output folder structure:
    granite/<CORPUS_NAME>/data/<split>/*.jpg

Usage:
    python prepare_dataset.py --corpus <CORPUS_NAME>
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------
# Image helpers
# ---------------------------------------------------------
def resize_image_with_padding(image, target_width, target_height):
    """
    Resize an image maintaining aspect ratio and add white padding.

    Args:
        image (PIL.Image): Input image.
        target_width (int): Target width for the output image.
        target_height (int): Target height for the output image.

    Returns:
        PIL.Image: The resized and padded image.
    """
    # Resize while keeping aspect ratio
    image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)

    # Create white background
    new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))

    # Center the resized image
    left = (target_width - image.width) // 2
    top = (target_height - image.height) // 2

    new_img.paste(image, (left, top))
    return new_img


# ---------------------------------------------------------
# Main processing function
# ---------------------------------------------------------
def resize_and_save_images(input_dir, output_dir, target_width, target_height):
    """
    Resize all images inside input_dir and save them into output_dir.

    Args:
        input_dir (str): Directory containing original images.
        output_dir (str): Directory where resized images will be saved.
        target_width (int): Target width for output images.
        target_height (int): Target height for output images.
    """
    print(f"\nProcessing images in: {input_dir}")

    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    if len(image_files) == 0:
        print("No images found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(image_files)} images.")

    for fname in tqdm(image_files, desc="Resizing", unit="image"):
        img_path = os.path.join(input_dir, fname)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue

        resized = resize_image_with_padding(img, target_width, target_height)
        resized.save(os.path.join(output_dir, fname))


# ---------------------------------------------------------
# CLI and execution
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Granite OCR training.")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus inside data/")

    args = parser.parse_args()
    corpus = args.corpus

    INPUT_BASE = os.path.join("data", corpus)
    OUTPUT_BASE = os.path.join("granite", corpus)

    TARGET_WIDTH = 414
    TARGET_HEIGHT = 585

    splits = ["train", "val", "test"]

    print(f"\nPreparing Granite dataset for corpus: {corpus}")
    print(f"Output directory: {OUTPUT_BASE}")

    for split in splits:
        input_dir = os.path.join(INPUT_BASE, split, "images")
        output_dir = os.path.join(OUTPUT_BASE, split)

        if not os.path.exists(input_dir):
            print(f"⚠Split '{split}' does not exist in the corpus. Skipping.")
            continue

        resize_and_save_images(
            input_dir=input_dir,
            output_dir=output_dir,
            target_width=TARGET_WIDTH,
            target_height=TARGET_HEIGHT
        )

    print("\nGranite dataset preparation completed!")


if __name__ == "__main__":
    main()
