"""
finetune_model.py

Fine-tune a YOLOv10 docLayout model on a prepared dataset, while tracking
energy consumption using CodeCarbon.

Usage:
    python docLayout/finetune_model.py --corpus <CORPUS_NAME> --input_model <INPUT_MODEL> --output_model <OUTPUT_MODEL>

Arguments:
    --corpus       : Name of the corpus folder inside 'docLayout/'.
    --input_model  : Pretrained model path/name to fine-tune.
    --output_model : Name/path where the fine-tuned model will be saved.
"""

import os
import json
import argparse
import pandas as pd
import logging
logging.getLogger("codecarbon").disabled = True
from codecarbon import EmissionsTracker

# Set dataset root to project root
os.environ["ULTRALYTICS_DATASETS_DIR"] = "./"
from doclayout_yolo import YOLOv10

def start_emissions_tracker(output_path):
    """
    Start the CodeCarbon emissions tracker.

    Args:
        output_path (str): Path to save emissions JSON.

    Returns:
        EmissionsTracker: Active tracker object.
    """
    tracker = EmissionsTracker(output_file=output_path)
    tracker.start()
    return tracker


def stop_and_save_emissions(tracker, output_path):
    """
    Stop the emissions tracker, read the CSV, convert to JSON, and save.

    Args:
        tracker (EmissionsTracker): Active tracker object.
        output_path (str): Path to save emissions JSON.
    """
    tracker.stop()

    # Read emissions CSV
    df = pd.read_csv(output_path)
    last_row = df.iloc[-1]  # take last row if multiple entries

    emissions_data = {
        "duration": str(last_row["duration"]),
        "emissions": str(last_row["emissions"]),
        "cpu_energy": str(last_row["cpu_energy"]),
        "gpu_energy": str(last_row["gpu_energy"]),
        "ram_energy": str(last_row["ram_energy"]),
        "energy_consumed": str(last_row["energy_consumed"]),
        "cpu_count": str(last_row["cpu_count"]),
        "gpu_count": str(last_row["gpu_count"]),
        "cpu_model": str(last_row["cpu_model"]),
        "gpu_model": str(last_row["gpu_model"]),
        "ram_total_size": str(last_row["ram_total_size"]),
    }

    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(emissions_data, f, indent=4)

    print(f"DocLayout fine-tuning emissions saved at {output_path}")


def finetune_model(corpus_name, input_model_name, output_model_name, epochs=30, batch_size=2, device='cuda:0'):
    """
    Fine-tune a YOLOv10 model for the given corpus.

    Args:
        corpus_name (str): Name of the dataset folder inside 'docLayout/'.
        input_model_name (str): Pretrained model path/name to fine-tune.
        output_model_name (str): Directory to save fine-tuned model and logs.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): Device to run training on ('cuda:0' or 'cpu').
    """
    base_dir = 'docLayout'
    models_dir = os.path.join(base_dir, 'models')
    corpus_dir = os.path.join(base_dir, corpus_name)

    data_yaml = os.path.join(corpus_dir, 'data.yaml')
    input_model_path = os.path.join(models_dir, input_model_name)
    output_model_path = os.path.join(models_dir, output_model_name)
    os.makedirs(output_model_path, exist_ok=True)

    emissions_file = os.path.join(output_model_path, "emissions.json")

    # Start emissions tracking
    tracker = start_emissions_tracker(emissions_file)

    # Load YOLOv10 model
    model = YOLOv10(input_model_path)

    # Train model
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        project=output_model_path,
        exist_ok=True,
        device=device,
        amp=False,
        val=True
    )

    # Stop emissions tracker and save results
    stop_and_save_emissions(tracker, emissions_file)

    print(f"DocLayout fine-tuning completed. Model saved at {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a YOLOv10 docLayout model")
    parser.add_argument("--corpus", type=str, required=True, help="Name of the corpus folder inside 'docLayout/'")
    parser.add_argument("--input_model", type=str, default='doclayout_yolo_docstructbench_imgsz1024.pt', help="Pretrained model to fine-tune")
    parser.add_argument("--output_model", type=str, required=True, help="Directory to save fine-tuned model")
    args = parser.parse_args()

    finetune_model(args.corpus, args.input_model, args.output_model)