import sys
import os
sys.path.append(os.path.dirname(__file__))  # Adds trOCR folder to path
import pandas as pd
from torch.utils.data import DataLoader
from utils import IAMDataset  # Custom Dataset class for trOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torch.optim import AdamW
from tqdm import tqdm
import json
import argparse
import logging
logging.getLogger("codecarbon").disabled = True
from codecarbon import EmissionsTracker

def finetune_model(corpus: str, input_model: str, output_model: str):
    """
    Fine-tune a trOCR model on a custom corpus.

    Args:
        corpus (str): Name of the corpus folder inside 'trOCR/'.
        input_model (str): Pretrained HuggingFace model to fine-tune.
        output_model (str): Directory to save the fine-tuned model.
    """
    base_folder = 'trOCR'
    models_dir = os.path.join(base_folder, 'models')
    output_folder = os.path.join(models_dir, output_model)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # === Start emissions tracker ===
    tracker = EmissionsTracker(output_file=os.path.join(output_folder, "emissions.json"))
    tracker.start()

    # === Load datasets ===
    train_df = pd.read_csv(os.path.join(base_folder, corpus, 'train.csv'))
    val_df = pd.read_csv(os.path.join(base_folder, corpus, 'val.csv'))

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # === Load processor and model ===
    processor = TrOCRProcessor.from_pretrained(input_model)

    train_dataset = IAMDataset(root_dir='./', df=train_df, processor=processor)
    val_dataset = IAMDataset(root_dir='./', df=val_df, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(input_model)
    model.to(device)

    # === Configure model decoder and beam search parameters ===
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # === Training loop ===
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # Move batch tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Average loss after epoch {epoch+1}: {train_loss / len(train_dataloader):.4f}")

    # === Save trained model and processor ===
    model.save_pretrained(output_folder)
    processor.save_pretrained(output_folder)

    # === Stop emissions tracker ===
    tracker.stop()

    # === Convert emissions CSV to JSON ===
    df = pd.read_csv(os.path.join(output_folder, "emissions.json"))
    data = df.iloc[-1]  # Take the last row

    emissions_data = {col: str(data[col]) for col in [
        "duration", "emissions", "cpu_energy", "gpu_energy", "ram_energy",
        "energy_consumed", "cpu_count", "gpu_count", "cpu_model", "gpu_model",
        "ram_total_size", "country_iso_code"
    ]}

    with open(os.path.join(output_folder, "emissions.json"), "w") as f:
        json.dump(emissions_data, f, indent=4)

    print(f"OCR fine-tuning completed. Model and emissions saved in: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a trOCR model on a custom corpus")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus folder inside 'trOCR/'")
    parser.add_argument("--input_model", type=str, default='microsoft/trocr-base-handwritten',
                        help="Pretrained HuggingFace model to fine-tune")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Directory to save fine-tuned model")
    args = parser.parse_args()

    finetune_model(args.corpus, args.input_model, args.output_model)
