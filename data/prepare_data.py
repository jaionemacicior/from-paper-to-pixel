"""
From Paper To Pixel: Experimental framework for OCR and document layout analysis
Author: Jaione Macicior-Mitxelena
License: MIT
Repository: https://github.com/jaionemacicior/from-paper-to-pixel
---------------------------------------------------

This script runs the complete pipeline:

1. Organize raw Transkribus exports
2. Parse PAGE-XML files to generate .box files, YOLO labels, and JSON metadata
3. Create train/validation/test splits and organize files
"""

from data_utils.organize_transkribus_data import organize_raw_data
from data_utils.parse_pagexml import parse_xml_files
from data_utils.create_splits import make_splits, organize_splits
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, required=True,
                    help="Name of the corpus inside data/")

args = parser.parse_args()
CORPUS_NAME = args.corpus

# Run pipeline
print("\n\033[1m[STEP 1] Preparing raw data...\033[0m")
organize_raw_data(CORPUS_NAME)

print("\n\033[1m[STEP 2] Parsing PAGE-XML files...\033[0m")
parse_xml_files(CORPUS_NAME)

print("\n\033[1m[STEP 3] Creating dataset splits...\033[0m")
splits = make_splits(CORPUS_NAME)
organize_splits(CORPUS_NAME, splits)

print("\n\033[1m[DONE] Data preparation pipeline completed successfully!\033[0m")