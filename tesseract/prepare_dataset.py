import os
import shutil
import argparse

def copy_tiff_and_txt(source_dir, dest_dir):
    """
    Copy TIFF images and TXT ground truth files from source_dir to dest_dir.

    TIFF images are copied as-is.
    TXT files are renamed to *.gt.txt for Tesseract training.

    Args:
        source_dir (str): Path to the source split directory (train/val/test).
        dest_dir (str): Path to the destination split directory.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Copy TIFF images
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".tiff"):
                src = os.path.join(root, file)
                dst = os.path.join(dest_dir, file)
                shutil.copy2(src, dst)

    # Copy TXT files and rename to *.gt.txt
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                base_name = os.path.splitext(file)[0]
                new_name = base_name + ".gt.txt"
                src = os.path.join(root, file)
                dst = os.path.join(dest_dir, new_name)
                shutil.copy2(src, dst)

def prepare_tesseract_dataset(corpus_name, base_path="data", output_base="tesseract"):
    """
    Prepare a Tesseract-compatible dataset from a corpus folder.

    Copies TIFF images and TXT ground truth files for train, val, and test splits.

    Args:
        corpus_name (str): Name of the corpus folder inside `base_path`.
        base_path (str): Root path containing the corpus folder (default: "data").
        output_base (str): Root path where the Tesseract dataset will be stored (default: "tesseract").
    """
    splits = ['train', 'val', 'test']

    for split in splits:
        # Source and destination directories
        split_in_dir = os.path.join(base_path, corpus_name, split)
        split_out_dir = os.path.join(output_base, corpus_name, split)

        copy_tiff_and_txt(split_in_dir, split_out_dir)

        print(f"Dataset '{split}' ready at: {split_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Tesseract OCR training.")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus folder inside 'data/'")

    args = parser.parse_args()
    prepare_tesseract_dataset(corpus_name=args.corpus)
