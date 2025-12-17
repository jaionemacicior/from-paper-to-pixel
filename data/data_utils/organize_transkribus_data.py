"""
Prepare and extract PAGE XML, images, and transcriptions downloaded from Transkribus.
Save it in data/CORPUS_NAME/raw_data

This script:
    - Normalizes the folder structure inside data/<CORPUS_NAME>/raw_data
    - Moves and renames PAGE XML files, JPG files, and TXT transcriptions
    - Converts JPG scans to TIFF format
    - Extracts bounding boxes and text from PAGE XML files
    - Produces word-level annotations from line-level transcriptions

"""
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image


# ======================================================================
# UTILITIES
# ======================================================================

import os
import shutil

def move_subfolders_and_delete_parent(parent_folder: str) -> None:
    """
    Move all subfolders inside a folder up one level and delete the parent.
    Also deletes any .txt or .pdf files inside the parent folder.

    Parameters:
        parent_folder (str): Path to folder whose contents should be moved.

    Example:
        raw_data/0001/0001_p001/  →  raw_data/0001_p001
    """
    parent_path = os.path.abspath(parent_folder)
    parent_dir = os.path.dirname(parent_path)

    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)

        # Delete  PDF files
        if os.path.isfile(item_path) and item_path.endswith(".pdf"):
            os.remove(item_path)
            continue

        # Move folders or other files up one level
        shutil.move(item_path, parent_dir)

    # Delete the now-empty parent folder
    shutil.rmtree(parent_folder)


def extract_bbox(coords: str):
    """
    Parse PAGE 'points' string and return bounding box (x_min, y_min, x_max, y_max).

    Example points string:
        "100,200 300,200 300,400 100,400"
    """
    points = [tuple(map(int, p.split(","))) for p in coords.split()]
    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)

    return x_min, y_min, x_max, y_max


def parse_page_xml(xml_file: str, output_file: str, region: bool = False) -> None:
    """
    Extract text and bounding boxes from a PAGE XML file.

    Parameters:
        xml_file (str): Path to input PAGE XML file.
        output_file (str): Path where bounding box file (.box) will be stored.
        region (bool): If True, read TextRegions; otherwise read TextLines.

    Output format per line:
        x1 y1 x2 y2 text
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'page': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

    with open(output_file, "w", encoding="utf-8") as box_file:
        tag = 'TextRegion' if region else 'TextLine'

        for region in root.findall(f".//page:{tag}", ns):
            coords = region.find("page:Coords", ns).attrib["points"]
            x_min, y_min, x_max, y_max = extract_bbox(coords)

            text_elements = region.findall(".//page:TextEquiv/page:Unicode", ns)
            if not text_elements:
                box_file.write(f"{x_min} {y_min} {x_max} {y_max}\n")
            else:
                full_text = " ".join(t.text for t in text_elements if t.text)
                full_text = full_text.replace("\r", "").replace("\n", "")
                box_file.write(f"{x_min} {y_min} {x_max} {y_max} {full_text}\n")


def split_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split each line of text into individual words.

    Input DataFrame must contain:
        ['filename', 'x1', 'y1', 'x2', 'y2', 'line']

    Returns:
        A DataFrame where each row is a word.
    """
    rows = []
    for _, row in df.iterrows():
        words = row["line"].split()
        for word in words:
            rows.append({
                "filename": row["filename"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "word": word
            })
    return pd.DataFrame(rows)


def normalize_box(box, width=1000, height=1000):
    """
    Normalize bounding box coordinates to the LayoutLM expected range [0, 1000].

    Parameters:
        box (list): [x1, y1, x2, y2]
    """
    return [
        int(1000 * (int(box[0]) / width)),
        int(1000 * (int(box[1]) / height)),
        int(1000 * (int(box[2]) / width)),
        int(1000 * (int(box[3]) / height)),
    ]


def read_bbox_and_words(path: str) -> pd.DataFrame:
    """
    Read a .box file and convert it to word-level annotations.

    Returns:
        DataFrame with columns:
            ['filename', 'x1', 'y1', 'x2', 'y2', 'word']
    """
    bbox_and_words = []

    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        x1, y1, x2, y2 = normalize_box(parts[0:4])
        text = ' '.join(parts[4:])
        filename = os.path.basename(path).split('.')[0]

        bbox_and_words.append([filename, x1, y1, x2, y2, text])

    df = pd.DataFrame(bbox_and_words,
                      columns=['filename', 'x1', 'y1', 'x2', 'y2', 'line'])
    return split_line(df)


# ======================================================================
# PROCESS RAW DATA
# ======================================================================

def organize_raw_data(CORPUS_NAME):
    # PATHS
    data_path = os.path.join(CORPUS_NAME, 'raw_data')
    tiff_folder = os.path.join(CORPUS_NAME, 'ground_truth_data', 'tiff')
    jpg_folder = os.path.join(CORPUS_NAME, 'ground_truth_data', 'jpg')
    transcriptions_folder = os.path.join(CORPUS_NAME, 'ground_truth_data', 'txt')
    xml_folder = os.path.join(CORPUS_NAME, 'ground_truth_data', 'xml')

    # Make sure required folders exist
    os.makedirs(tiff_folder, exist_ok=True)
    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(transcriptions_folder, exist_ok=True)
    os.makedirs(xml_folder, exist_ok=True)

    # Flatten nested folders from Transkribus
    print("\tFlattening nested folders...")
    for item in tqdm(os.listdir(data_path), desc="Flattening raw data"):
        full_path = os.path.join(data_path, item)
        if item == 'log.txt':
            os.remove(full_path)
        else:
            move_subfolders_and_delete_parent(full_path)

    # Remove "-_Copy" duplicates created by Windows
    print("\tCleaning duplicate folder names...")
    for item in tqdm(os.listdir(data_path), desc="Cleaning duplicates"):
        old_path = os.path.join(data_path, item)
        new_path = old_path.replace('_-_Copy', '')
        if old_path != new_path:
            os.rename(old_path, new_path)

    # Move TXT, JPG, and XML files to their corresponding folders
    print("\tMoving TXT, JPG, and XML files...")
    for item in tqdm(os.listdir(data_path), desc="Organizing files"):
        item_path = os.path.join(data_path, item)
        if item.endswith('.txt'):
            shutil.copy2(item_path, os.path.join(transcriptions_folder, item))
        else:  # XML folder
            jpg_src = os.path.join(item_path, '0001_p001.jpg')
            xml_src = os.path.join(item_path, 'page', '0001_p001.xml')
            shutil.copy2(jpg_src, os.path.join(jpg_folder, f"{item}.jpg"))
            shutil.copy2(xml_src, os.path.join(xml_folder, f"{item}.xml"))

    # Convert JPG to TIFF
    print("\tConverting JPG to TIFF...")
    for filename in tqdm(os.listdir(jpg_folder), desc="Converting images"):
        if filename.lower().endswith('.jpg'):
            jpg_path = os.path.join(jpg_folder, filename)
            img = Image.open(jpg_path)
            base_name = os.path.splitext(filename)[0]
            tiff_path = os.path.join(tiff_folder, f"{base_name}.tiff")
            img.save(tiff_path, format="TIFF")