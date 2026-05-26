"""
From Paper To Pixel: Experimental framework for OCR and document layout analysis
Author: Jaione Macicior-Mitxelena
License: MIT
Repository: https://github.com/jaionemacicior/from-paper-to-pixel
---------------------------------------------------
"""


import os
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def show_best_worst(df_sorted, metric, model_type, model_name, out_folder, top_n=3):
    """
    Generates two figures:
      1. Best instances
      2. Worst instances
    """
    best = df_sorted.head(top_n)
    worst = df_sorted.tail(top_n)

    os.makedirs(out_folder, exist_ok=True)

    # ---- Best Figure ----
    fig_best, axes = plt.subplots(1, top_n, figsize=(20, 12))
    for idx, (_, row) in enumerate(best.iterrows()):
        ax = axes[idx] if top_n > 1 else axes
        img_path = row.get("path", "")
        page_id = row.get("page_id", "N/A")
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Best {idx+1}\nID: {page_id}\n{metric.upper()}: {row[metric]:.3f}",
                     fontsize=10, pad=10)

    plt.tight_layout()
    best_path = os.path.join(out_folder, f"{model_type}_{model_name}_best.png")
    plt.savefig(best_path, dpi=200)
    plt.close()
    print(f"Best figure saved at {best_path}")

    # ---- Worst Figure ----
    fig_worst, axes = plt.subplots(1, top_n, figsize=(20, 12))
    for idx, (_, row) in enumerate(worst.iterrows()):
        ax = axes[idx] if top_n > 1 else axes
        img_path = row.get("path", "")
        page_id = row.get("page_id", "N/A")
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Worst {idx+1}\nID: {page_id}\n{metric.upper()}: {row[metric]:.3f}",
                     fontsize=10, pad=10)

    plt.tight_layout()
    worst_path = os.path.join(out_folder, f"{model_type}_{model_name}_worst.png")
    plt.savefig(worst_path, dpi=200)
    plt.close()
    print(f"Worst figure saved at {worst_path}")


def process_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_page_metrics", [])


def main(base_dir, corpus,  split="test", top_n=3):
    """
    Analyze best and worst instances per model.

    Args:
        base_dir (str): Base folder containing model evaluations
        corpus (str): Name of the corpus folder
        split (str): Dataset split ('train', 'val', 'test')
        top_n (int): Number of top/bottom images to display
    """
    out_folder = os.path.join("images", "worst_best")
    os.makedirs(out_folder, exist_ok=True)

    model_types = ["tesseract", "granite", "trOCR", "docLayout"]
    target_metrics = {
        "tesseract": "cer",
        "granite": "cer",
        "trOCR": "cer",
        "docLayout": "avg_iou"
    }

    results = {}

    for model_type in model_types:
        print('Processing ', model_type)
        metric = target_metrics[model_type]
        json_dir = os.path.join(base_dir, model_type, "evaluation", corpus, split)
        if not os.path.exists(json_dir):
            print(f"No evaluation json for {model_type}, {json_dir} not found")
            continue

        results[model_type] = {}

        for json_file in os.listdir(json_dir):
            if not json_file.endswith(".json"):
                continue
            match = re.match(r"metrics-(.+)\.json", json_file)
            if not match:
                continue
            model_name = match.group(1)
            print(f"Processing {model_type}/{model_name}")
            json_path = os.path.join(json_dir, json_file)

            pages = process_json(json_path)
            pages_metrics = []
            for page in pages:
                page_info = {
                    "page_id": page.get("page_id", None),
                    metric: page.get(metric, None),
                    "path": os.path.join("data", corpus, split, "images", f"{page.get('page_id')}.jpg")
                }
                pages_metrics.append(page_info)

            df = pd.DataFrame(pages_metrics)
            if df.empty:
                continue

            # Sort according to metric
            ascending = True if metric.lower() == "cer" else False
            df_sorted = df.sort_values(metric, ascending=ascending)

            results[model_type][model_name] = {
                "best": df_sorted.head(top_n).to_dict(orient="records"),
                "worst": df_sorted.tail(top_n).to_dict(orient="records")
            }

            # Generate plots
            show_best_worst(df_sorted, metric, model_type, model_name, out_folder, top_n=top_n)

    # Save results
    with open(os.path.join(out_folder, "worst_best.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Analysis completed. JSON and figures saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze best and worst instances per model.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base folder for model evaluations")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus folder name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top/bottom images to show")
    args = parser.parse_args()
    main(args.base_dir, args.corpus, args.split, args.top_n)
