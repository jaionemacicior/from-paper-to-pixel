"""
Class Map Manager
-----------------

This module handles class-to-ID mapping for document layout regions.
It stores the mapping in a JSON file (classes.json) and provides
helper functions to load, save, and query class IDs.

Features:
    ✓ Load existing class mapping from classes.json
    ✓ Save updated mapping to classes.json
    ✓ Retrieve or create class IDs automatically
"""

import json
import os


def load_class_map(CORPUS_NAME: str) -> dict:
    """
    Load the class mapping from classes.json.

    Parameters:
        CORPUS_NAME (str): Corpus folder name

    Returns:
        dict: {class_name: class_id}

    If the file does not exist, it will be created as an empty dictionary.
    """
    CLASS_FILE = f"{CORPUS_NAME}/classes.json"  # JSON file to store class → ID mapping

    if not os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
        print(f"\t\t{CLASS_FILE} created as an empty dictionary.")
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_class_map(class_map: dict, CORPUS_NAME: str) -> None:
    """
    Save the class mapping to classes.json.

    Parameters:
        class_map (dict): Dictionary of {class_name: class_id}
        CORPUS_NAME (str): Corpus folder name
    """

    CLASS_FILE = f"{CORPUS_NAME}/classes.json"
    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)


def get_class_id(class_map: dict, class_name: str, CORPUS_NAME: str) -> int:
    """
    Retrieve the class ID for a given class name.
    If the class does not exist, it is added to the map with a new ID.

    Parameters:
        class_map (dict): Existing class map
        class_name (str): Name of the class
        CORPUS_NAME (str): Corpus folder name

    Returns:
        int: ID assigned to the class
    """
    if class_name not in class_map:
        new_id = len(class_map)
        class_map[class_name] = new_id
        print(f"\t\tNew class added: '{class_name}' → ID {new_id}")
        save_class_map(class_map, CORPUS_NAME)
    return class_map[class_name]
