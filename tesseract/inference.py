import os
import subprocess
from tqdm import tqdm

# Paths for models and predictions
BASE_DIR = 'tesseract'
MODEL_DIR = os.path.join(BASE_DIR, "models")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)


def predict(image_path: str, model_name: str, corpus: str, split: str) -> str:
    """
    Apply OCR with Tesseract using a specific traineddata model.

    Args:
        image_path (str): Path to the image (TIFF)
        model_name (str): Name of the Tesseract language/traineddata model
        corpus (str): Corpus name
        split (str): Dataset split (train/val/test)

    Returns:
        str: Recognized text
    """
    if not image_path.endswith(".tiff"):
        raise ValueError(f"The image {image_path} is not a TIFF file.")

    out_dir = os.path.join(PRED_DIR, model_name, corpus, split)
    os.makedirs(out_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    output_path = os.path.join(out_dir, os.path.splitext(image_name)[0])

    command = [
        "tesseract",
        str(image_path),
        str(output_path),
        "-l",
        model_name
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {image_path}: {e}")


def predict_on_dataset(folder_path: str, model_name: str, corpus: str):
    """
    Run Tesseract OCR on all images in a folder (uniform interface).

    Args:
        folder_path (str): Path to the split folder containing TIFF images
        model_name (str): Name of the Tesseract model
        corpus (str): Corpus name
    """
    split = folder_path.split("/")[-2]
    for file in tqdm(os.listdir(folder_path), desc=f"Tesseract predictions ({model_name})"):
        image_path = os.path.join(folder_path, file)
        predict(image_path, model_name, corpus, split)
