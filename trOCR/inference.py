import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

# Paths for models and predictions
BASE_DIR = 'trOCR'
MODEL_DIR = os.path.join(BASE_DIR, "models")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# Cache for loaded models (optional, not used in this snippet)
_loaded_models = {}


def predict(image_path: str, model_name: str, model, processor) -> str:
    """
    Perform OCR using TrOCR (pretrained or fine-tuned model).

    Args:
        image_path (str): Path to the image (JPG)
        model_name (str): Name of the fine-tuned or HuggingFace model
        model: TrOCR model
        processor: TrOCR processor

    Returns:
        str: Recognized text
    """
    split = image_path.split("/")[2]  # assumes path: data/<corpus>/<split>/images/xxx.jpg
    out_dir = os.path.join(PRED_DIR, model_name, split)
    os.makedirs(out_dir, exist_ok=True)

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate text
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Save prediction
    output_path = os.path.join(out_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def predict_on_dataset(folder_path: str, model_name: str, corpus: str):
    """
    Run TrOCR OCR on all images in a folder (uniform interface).

    Args:
        folder_path (str): Path to the split folder containing images
        model_name (str): Name of the TrOCR model
        corpus (str): Corpus name
    """
    if model_name.startswith('baseline'):
        model_path = 'microsoft/trocr-base-handwritten'
    else:
        model_path = os.path.join(MODEL_DIR, model_name)

    print(f"Loading TrOCR model from {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.eval()

    for file in tqdm(os.listdir(folder_path), desc=f"TrOCR predictions ({model_name})"):
        image_path = os.path.join(folder_path, file)
        predict(image_path, model_name, model, processor)
