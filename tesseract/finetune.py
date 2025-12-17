import os
import subprocess
import json
import pandas as pd
from tqdm import tqdm
import argparse
import logging
logging.getLogger("codecarbon").setLevel(logging.ERROR)
from codecarbon import EmissionsTracker


def generate_lstm_training_files(data_folder, lang):
    """
    Generate .box and .lstmf files required for LSTM training.
    Args:
        data_folder (str): Path to the folder containing TIFF images and .gt.txt files.
        lang (str): Language code, e.g., 'spa'
    """
    file_bases = [os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith('.tiff')]

    for base in tqdm(file_bases, desc="Generating .box and .lstmf files"):
        tiff_path = os.path.join(data_folder, base + ".tiff")
        output_base = os.path.join(data_folder, base)

        if not os.path.exists(tiff_path):
            print(f"Missing TIFF file for: {base}")
            continue

        try:
            # Generate .box file
            subprocess.run([
                "tesseract", tiff_path, output_base,
                "-l", lang,         # <-- use the input model language
                "--psm", "6", "makebox"
            ], check=True)

            # Generate .lstmf file
            subprocess.run([
                "tesseract", tiff_path, output_base,
                "-l", lang,         # <-- use the input model language
                "--psm", "6", "lstm.train"
            ], check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error processing {base}: {e}")

def create_list_file(data_folder):
    """
    Create a 'list.txt' containing absolute paths to all .lstmf files.
    Args:
        data_folder (str): Path to the folder containing .lstmf files.
    Returns:
        str: Path to the created list.txt file.
    """
    list_file_path = os.path.join(data_folder, "list.txt")
    if not os.path.exists(list_file_path):
        with open(list_file_path, "w", encoding="utf-8", newline='\n') as f:
            for filename in os.listdir(data_folder):
                if filename.endswith(".lstmf"):
                    f.write(os.path.abspath(os.path.join(data_folder, filename)) + "\n")
        print(f"list.txt created at {list_file_path}")
    else:
        print("list.txt already exists")
    return list_file_path

def train_tesseract_model(data_folder, input_model_path, output_model_path, models_dir):
    """
    Train the LSTM model using Tesseract's lstmtraining tool.
    Args:
        data_folder (str): Folder containing training data.
        input_model_path (str): Path to the pretrained .traineddata model.
        output_model_path (str): Folder to save the fine-tuned model.
        models_dir (str): Base directory for Tesseract models.
    """
    checkpoint_dir = os.path.join(output_model_path, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    lstm_model_path = input_model_path.replace('.traineddata', '.lstm')
    final_model_path = os.path.join(output_model_path, os.path.basename(output_model_path) + '.traineddata')

    # Extract LSTM from pretrained model
    subprocess.run(['combine_tessdata', '-e', input_model_path, lstm_model_path], check=True)

    list_txt_path = create_list_file(data_folder)

    # Step 1: Train LSTM
    print("🔁 Starting LSTM training...")
    train_cmd = [
        "lstmtraining",
        "--model_output", os.path.join(checkpoint_dir, os.path.basename(output_model_path)),
        "--continue_from", lstm_model_path,
        "--traineddata", input_model_path,
        "--train_listfile", list_txt_path,
        "--max_iterations", "4000"
    ]
    subprocess.run(train_cmd, check=True)
    print("✅ Training complete!")

    # Step 2: Stop training & generate final model
    print("📦 Generating final .traineddata model...")
    checkpoint_model = os.path.join(checkpoint_dir, os.path.basename(output_model_path) + "_checkpoint")
    final_cmd = [
        "lstmtraining",
        "--stop_training",
        "--continue_from", checkpoint_model,
        "--traineddata", input_model_path,
        "--model_output", os.path.join(checkpoint_dir, os.path.basename(output_model_path)),
    ]
    subprocess.run(final_cmd, check=True)
    print(f"✅ Final model created at: {final_model_path}")

def finetune_model(input_model, output_model, corpus):
    # Paths
    base_dir = 'tesseract'
    data_folder = os.path.join(base_dir, corpus, 'train')
    models_dir = os.path.join(base_dir, 'models')
    input_model_path = os.path.join(models_dir, input_model)
    output_model_path = os.path.join(models_dir, output_model)
    os.makedirs(output_model_path, exist_ok=True)

    # Start emissions tracker
    tracker = EmissionsTracker(output_file=os.path.join(output_model_path, "emissions.json"))
    tracker.start()

    # Generate training files
    generate_lstm_training_files(data_folder, lang = input_model)

    # Train model
    train_tesseract_model(data_folder, input_model_path, output_model_path, models_dir)

    # Stop emissions tracker
    tracker.stop()

    # Save emissions as JSON
    emissions_csv_path = os.path.join(output_model_path, "emissions.csv")
    if os.path.exists(emissions_csv_path):
        df = pd.read_csv(emissions_csv_path)
        data = df.iloc[-1]
        emissions_data = {col: str(data[col]) for col in [
            "duration", "emissions", "cpu_energy", "gpu_energy", "ram_energy",
            "energy_consumed", "cpu_count", "gpu_count", "cpu_model", "gpu_model",
            "ram_total_size", "country_iso_code"
        ]}
        with open(os.path.join(output_model_path, "emissions.json"), "w") as f:
            json.dump(emissions_data, f, indent=4)
        print("✔️ OCR training completed and emissions saved as JSON.")

if __name__ == "__main__":
    """
    Parse command-line arguments.
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a Tesseract OCR model on a custom corpus")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus folder inside 'tesseract/'")
    parser.add_argument("--input_model", type=str, required=True,
                        help="Pretrained model to fine-tune (traineddata file)")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Directory to save fine-tuned model")
    args = parser.parse_args()
    finetune_model(input_model=args.input_model, output_model=args.output_model, corpus=args.corpus,)