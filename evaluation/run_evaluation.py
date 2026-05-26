'''
From Paper To Pixel: Experimental framework for OCR and document layout analysis
Author: Jaione Macicior-Mitxelena
License: MIT
Repository: https://github.com/jaionemacicior/from-paper-to-pixel
'''


import argparse
from evaluation.evaluator import predict_on_dataset
import os
import pandas as pd
import json
import logging
logging.getLogger("codecarbon").disabled = True
from codecarbon import EmissionsTracker


def main():
    """
    Usage:
    python -m evaluation.run_evaluation \
        --model_type <MODEL_TYPE> \
        --model_name <MODEL_NAME> \
        --corpus <CORPUS> \
        --split val \
        --eval True
    """
    parser = argparse.ArgumentParser(description="Evaluate OCR or Layout models on a dataset")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['tesseract', 'trOCR', 'granite', 'docLayout'],
                        help="Type of model (tesseract, trOCR, granite, docLayout)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model (folder containing the trained model)")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus to evaluate")
    parser.add_argument("--split", type=str, required=True,
                        choices=['train', 'val', 'test'], help="Dataset split to evaluate")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for layout models")
    parser.add_argument("--eval", type=str, default=None,
                        help="Whether to compute evaluation metrics")

    args = parser.parse_args()

    # Setup output folder for predictions
    output_folder = os.path.join(args.model_type, 'predictions', args.model_name, args.corpus, args.split)
    os.makedirs(output_folder, exist_ok=True)

    # Start emissions tracker
    tracker = EmissionsTracker(output_file=os.path.join(output_folder, "emissions.csv"))
    tracker.start()

    # Run predictions on the dataset
    predict_on_dataset(
        model_type=args.model_type,
        model_name=args.model_name,
        split=args.split,
        iou_threshold=args.iou_threshold,
        eval=args.eval,
        corpus=args.corpus
    )

    # Stop emissions tracker
    tracker.stop()

    # Convert emissions CSV to JSON
    df = pd.read_csv(os.path.join(output_folder, "emissions.csv"))
    data = df.iloc[-1]  # take last row in case of multiple entries

    emissions_data = {col: str(data[col]) for col in [
        "duration", "emissions", "cpu_energy", "gpu_energy", "ram_energy",
        "energy_consumed", "cpu_count", "gpu_count", "cpu_model", "gpu_model",
        "ram_total_size", "country_iso_code"
    ]}

    with open(os.path.join(output_folder, "emissions.json"), "w") as f:
        json.dump(emissions_data, f, indent=4)

    print("Evaluation completed. Emissions saved in ", os.path.join(output_folder, "emissions.json"))

if __name__ == "__main__":
    main()
