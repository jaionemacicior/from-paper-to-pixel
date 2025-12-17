"""
PAGE-XML Parser for Region, Line, and Relation Extraction
---------------------------------------------------------

This script reads PAGE-XML files exported from Transkribus and extracts:

    ✓ Text regions (with coordinates and normalized YOLO boxes)
    ✓ Text lines (with coordinates, text, and YOLO boxes)
    ✓ Relations between regions (if present)
    ✓ JSON structured data
    ✓ YOLO labels for regions
    ✓ .box files (Tesseract format) for text lines

USAGE:
    - This script should be run AFTER the raw data preparation script.
    - XML files must be located in:
          data/<CORPUS_NAME>/ground_truth_data/xml/

DEPENDENCIES:
    - class_mapper.py  (must provide: load_class_map(), get_class_id() )
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from .class_mapper import load_class_map, get_class_id


# =====================================================================
# PAGE-XML PARSER
# =====================================================================

def extract_info_from_pagexml(xml_path: str) -> dict:
    """
    Extract regions, lines, and relations from a PAGE XML file.

    Parameters:
        xml_path (str): Path to the PAGE XML file.

    Returns:
        dict containing:
            - image (jpg filename)
            - width, height
            - regions: list of region dicts
            - relations: list of relation dicts
    """

    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page = root.find('.//pc:Page', ns)
    image_width = int(page.attrib['imageWidth'])
    image_height = int(page.attrib['imageHeight'])

    # Assume JPG has same name as XML
    image_filename = xml_path.replace("xml", "jpg")

    regions = []

    # --------------------------------------------------------------
    # Extract TEXT REGIONS
    # --------------------------------------------------------------
    for region in page.findall('.//pc:TextRegion', ns):

        region_id = region.attrib.get('id')
        region_type = "paragraph"

        # Extract type from custom attribute if available
        custom_attr = region.attrib.get("custom", "")
        if "structure" in custom_attr:
            try:
                region_type = custom_attr.split("structure {type:")[1].split(";")[0].replace("}", "")
            except Exception:
                pass

        # Parse polygon coordinates
        coords_el = region.find('./pc:Coords', ns)
        points = coords_el.attrib['points']
        poly = [(int(x), int(y)) for x, y in (p.split(',') for p in points.split())]

        min_x = min(p[0] for p in poly)
        max_x = max(p[0] for p in poly)
        min_y = min(p[1] for p in poly)
        max_y = max(p[1] for p in poly)

        # YOLO normalized bounding box
        cx = ((min_x + max_x) / 2) / image_width
        cy = ((min_y + max_y) / 2) / image_height
        w = (max_x - min_x) / image_width
        h = (max_y - min_y) / image_height

        region_dict = {
            "id": region_id,
            "type": region_type,
            "bbox": [min_x, min_y, max_x, max_y],
            "yolo": [cx, cy, w, h],
            "lines": []
        }
        regions.append(region_dict)

        # ----------------------------------------------------------
        # Extract TEXT LINES within the region
        # ----------------------------------------------------------
        for line in region.findall('.//pc:TextLine', ns):
            line_id = line.attrib.get('id')

            # Extract text
            unicode_el = line.find('.//pc:Unicode', ns)
            text = unicode_el.text if unicode_el is not None else ""

            # Line coordinates
            line_coords = line.find('./pc:Coords', ns)
            if line_coords is None:
                continue

            line_points = line_coords.attrib['points']
            line_poly = [(float(x), float(y)) for x, y in (p.split(',') for p in line_points.split())]

            line_min_x = min(p[0] for p in line_poly)
            line_max_x = max(p[0] for p in line_poly)
            line_min_y = min(p[1] for p in line_poly)
            line_max_y = max(p[1] for p in line_poly)

            # Normalized YOLO for lines
            line_cx = ((line_min_x + line_max_x) / 2) / image_width
            line_cy = ((line_min_y + line_max_y) / 2) / image_height
            line_w = (line_max_x - line_min_x) / image_width
            line_h = (line_max_y - line_min_y) / image_height

            region_dict["lines"].append({
                "id": line_id,
                "text": text,
                "coords": line_poly,
                "bbox": [line_min_x, line_min_y, line_max_x, line_max_y],
                "yolo": [line_cx, line_cy, line_w, line_h]
            })

    # --------------------------------------------------------------
    # Extract RELATIONS (if present)
    # --------------------------------------------------------------
    relations = []

    for rel in root.findall(".//pc:Relation", ns):
        custom_attr = rel.get("custom", "") or ""

        # Optional relationName or relationType
        m_name = re.search(r'relationName\s*\{\s*value\s*:\s*([^;}\s]+)', custom_attr)
        m_rtype = re.search(r'relationType\s*\{\s*value\s*:\s*([^;}\s]+)', custom_attr)

        relation_name = m_name.group(1) if m_name else None
        relation_type_val = m_rtype.group(1) if m_rtype else None

        # Fallback: PAGE attribute "type"
        name = relation_name or relation_type_val or rel.get("type")

        # Related regions
        related = []
        for rref in rel.findall("pc:RegionRef", ns):
            region_ref = rref.get("regionRef")
            if region_ref:
                related.append(region_ref)

        if related:
            relations.append({
                "relationName": name,
                "regions": related
            })

    return {
        "image": image_filename,
        "width": image_width,
        "height": image_height,
        "regions": regions,
        "relations": relations
    }


# =====================================================================
# OUTPUT GENERATION
# =====================================================================

def save_outputs(class_map:dict, data: dict, xml_name: str, output_dir: str, corpus_name:str) -> None:
    """
    Saves YOLO labels, .box files, and JSON output for a parsed XML file.

    Parameters:
        class_map (dict): Mapping from class names to IDs.
        data (dict): Parsed XML data.
        xml_name (str): Filename of XML file.
        output_dir (str): Base folder.
    """

    base_name = os.path.splitext(xml_name)[0]

    # --------------------------------------------------------------
    # 1. YOLO labels for REGIONS
    # --------------------------------------------------------------
    yolo_path = os.path.join(output_dir, 'yolo_labels', f'{base_name}.txt')
    with open(yolo_path, 'w', encoding='utf-8') as f:
        for reg in data.get("regions", []):
            class_name = reg["type"]
            class_id = get_class_id(class_map, class_name, corpus_name)
            cx, cy, w, h = reg["yolo"]
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # --------------------------------------------------------------
    # 2. BOX files for LINES (Tesseract)
    # --------------------------------------------------------------
    box_path = os.path.join(output_dir, 'box_lines', f'{base_name}.box')
    with open(box_path, 'w', encoding='utf-8') as f:
        for reg in data.get("regions", []):
            for line in reg["lines"]:
                x_min, y_min, x_max, y_max = line['bbox']
                text = line['text'] if line['text'] else " "
                f.write(f"{text} {x_min} {y_min} {x_max} {y_max} 0\n")

    # --------------------------------------------------------------
    # 3. JSON full structured output
    # --------------------------------------------------------------
    json_path = os.path.join(output_dir, 'json', f'{base_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_xml_files(CORPUS_NAME:str):
    """
    Main execution:
        - Loads XML files
        - Runs PAGE-XML parser
        - Saves YOLO, BOX, and JSON outputs
    Parameters:
        CORPUS_NAME (str): Name of the corpus folder (e.g., 'los101')
    """
    # Load class mapping for YOLO labels
    class_map = load_class_map(CORPUS_NAME)

    input_folder = os.path.join(CORPUS_NAME, 'ground_truth_data', 'xml')
    output_folder = os.path.join(CORPUS_NAME, 'ground_truth_data')

    # Create output subfolders
    os.makedirs(os.path.join(output_folder, 'yolo_labels'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'box_lines'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'json'), exist_ok=True)

    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]

    for file in tqdm(xml_files, desc="Parsing PAGE-XML files"):
        xml_path = os.path.join(input_folder, file)
        data = extract_info_from_pagexml(xml_path)
        save_outputs(class_map, data, file, output_folder, CORPUS_NAME)