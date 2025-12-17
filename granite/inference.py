import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from tqdm import tqdm

SYSTEM_PROMPT = """You are an OCR expert specialized in Spanish documents.
You are analyzing an old book scan with potentially low quality.

INSTRUCTIONS:

    Extract ALL text exactly as it appears.

    Do not correct, interpret or modify the text in any way.

    Return ONLY the raw text, without any additional comments or formatting.

    Do not invent content not present in the image.

The output must be EXACTLY the recognized text, without adding anything else.
"""

USER_PROMPT = "Please perform OCR on this Spanish document."

# Paths for models and predictions
BASE_DIR = 'granite'
MODEL_DIR = os.path.join(BASE_DIR, "models")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)


def predict(image_path, model_name, model, processor, corpus, split):
    """
    Perform OCR on a single image using Granite fine-tuned model.

    Args:
        image_path (str): Path to the image (JPG)
        model_name (str): Name of the fine-tuned model
        model: Loaded model
        processor: Processor for the model
        corpus (str): Corpus name
        split (str): Dataset split (train/val/test)
    """
    image_name = os.path.basename(image_path)
    new_image_path = os.path.join(BASE_DIR, 'data', corpus, split, 'images', image_name)
    image = Image.open(new_image_path).convert("RGB")

    # Prepare input prompt
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]

    chat_text = processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=chat_text, images=[[image]], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        text = processor.tokenizer.decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip().replace('<doc> ', '')

    # Save prediction
    out_dir = os.path.join(PRED_DIR, model_name, corpus, split)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, image_name.replace('.jpg', '.txt'))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def predict_on_dataset(folder_path, model_name, corpus):
    """
    Run Granite OCR on all images in a folder (uniform interface).

    Args:
        folder_path (str): Path to images
        model_name (str): Model name
        corpus (str): Corpus name
    """
    if not model_name:
        raise ValueError("model_name cannot be None or empty")

    if model_name.startswith('baseline'):
        model_path = 'ibm-granite/granite-vision-3.2-2b'
    else:
        model_path = os.path.join(MODEL_DIR, model_name)

    print(f"Loading model from {model_path}")

    # Load processor and base model
    processor = AutoProcessor.from_pretrained('ibm-granite/granite-vision-3.2-2b')
    model = AutoModelForImageTextToText.from_pretrained(
        'ibm-granite/granite-vision-3.2-2b', device_map="auto", dtype=torch.bfloat16
    )

    # Load LoRA fine-tuning if available
    try:
        model = PeftModel.from_pretrained(model, model_path)
    except Exception:
        pass

    model.eval()

    # The folder_path now points to the split folder ('train', 'val', 'test')
    for file in tqdm(os.listdir(folder_path), desc=f"Granite OCR predictions ({model_name})"):
        image_path = os.path.join(folder_path, file)
        predict(image_path, model_name, model, processor, corpus, os.path.basename(folder_path))