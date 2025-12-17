#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def load_metrics(base_path, exclude=None):
    """
    Load evaluation JSONs from a folder structure.

    Args:
        base_path (str): path to folder containing model evaluation subfolders
                         e.g., granite/evaluation/los101/val
        exclude (list): list of model names to skip

    Returns:
        pd.DataFrame: DataFrame with model metrics
    """
    if exclude is None:
        exclude = []

    results = {}
    for filename in tqdm(os.listdir(base_path), desc="Loading models"): ## filename: metrics-<model_name>.json
        model_name = filename[len("metrics-"):-len(".json")]
        if model_name in exclude:
            continue
        print(model_name)
        # Assume metrics JSON is metrics-<model_name>.json
        json_file = os.path.join(base_path, filename)
        if os.path.isfile(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                results[model_name] = metrics
        else:
            print(f"Warning: JSON not found for {model_name} at {json_file}")

    df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    df = df.sort_values("Model")
    print(df)
    return df

def plot_metrics(df, title="Model Evaluation Metrics"):
    """
    Plot evaluation metrics in a grid.

    Args:
        df (pd.DataFrame): DataFrame containing metrics (columns: Model + metrics)
    """
    metrics = [col for col in df.columns if col != "Model"]
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Convert values to float and replace NaN/masked with 0
        y_values = pd.to_numeric(df[metric], errors='coerce').fillna(0).values

        x_labels = df["Model"].astype(str).values

        ax.scatter(x_labels, y_values, color='blue')
        ax.plot(x_labels, y_values, color='blue', linestyle='-')
        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, fontsize=16)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation metrics for OCR/Layout models")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to evaluation folder containing model metrics")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="List of model names to exclude from the plot")
    parser.add_argument("--title", type=str, default="Model Evaluation Metrics",
                        help="Title for the plot")
    args = parser.parse_args()

    df_metrics = load_metrics(args.base_path, exclude=args.exclude)
    if df_metrics.empty:
        print("No metrics found! Check the base_path and model names.")
    else:
        plot_metrics(df_metrics, title=args.title)
