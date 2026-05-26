'''
From Paper To Pixel: Experimental framework for OCR and document layout analysis
Author: Jaione Macicior-Mitxelena
License: MIT
Repository: https://github.com/jaionemacicior/from-paper-to-pixel
'''

from evaluation.metrics import *
import importlib
import numpy as np
from collections import defaultdict
import os
import json
import math

def evaluate_ocr(pred_dir, gt_dir):
    """
    Compares OCR predictions with ground truth text files.

    Args:
        pred_dir (str): Path to the folder with predictions (TXT files)
        gt_dir (str): Path to the folder with ground truth TXT files

    Returns:
        dict: Average metrics across all samples, including CER, WER, BLEU, ROUGE, etc.
    """
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    for gt_file in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, gt_file)

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_text = f.read().strip()

        cer_val = cer(pred_text, gt_text)
        if math.isinf(cer_val) or math.isnan(cer_val):
            cer_val = 1.0
        wer_val = wer(pred_text, gt_text)
        if math.isinf(wer_val) or math.isnan(wer_val):
            wer_val = 1.0

        rouge_scores = rouge(pred_text, gt_text)
        results.append({
            "pred": pred_text,
            "gt": gt_text,
            "cer": cer_val,
            "wer": wer_val,
            "lev": lev(pred_text, gt_text),
            "ned": ned(pred_text, gt_text),
            "bleu": bleu(pred_text, gt_text),
            "rouge1": rouge_scores["rouge1"],
            "rougeL": rouge_scores["rougeL"]
        })

    # Compute average metrics
    avg_metrics = {key: (sum(r[key] for r in results) / len(results) if results else None)
                   for key in ["cer", "wer", "lev", "ned", "bleu", "rouge1", "rougeL"]}

    return {"n_samples": len(results), **avg_metrics}


def iou(box1, box2):
    """
    Computes Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1, box2: Lists of [x1, y1, x2, y2]

    Returns:
        float: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def evaluate_layout_json(gt_json_path, pred_json_path, iou_threshold=0.5, alpha=0.5):
    """
    Evaluates layout predictions against ground truth JSON files.

    Args:
        gt_json_path (str): Path to ground truth JSON
        pred_json_path (str): Path to predictions JSON
        iou_threshold (float): Minimum IoU to consider a match
        alpha (float): Weight for composite score (IoU vs F1)

    Returns:
        dict: Global metrics and per-page metrics
    """
    os.makedirs(os.path.dirname(gt_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_json_path), exist_ok=True)

    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    with open(pred_json_path, "r") as f:
        pred_data = json.load(f)

    per_page_metrics = []
    avg_ious = []

    for page_id, gt_boxes in gt_data.items():
        pred_boxes = pred_data.get(page_id, [])
        matches = []

        # Match each GT box to the prediction with the highest IoU
        for gt in gt_boxes:
            best_iou = 0
            best_pred = None
            for pred in pred_boxes:
                iou_val = iou(gt["xyxy"], pred["xyxy"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred = pred
            matches.append((gt, best_pred, best_iou))

        avg_iou = np.mean([m[2] for m in matches]) if matches else 0
        avg_ious.append(avg_iou)

        # Box type statistics
        type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for gt, pred, iou_val in matches:
            cls = gt["class_name"]
            if pred and iou_val >= iou_threshold:
                if pred["class_name"] == cls:
                    type_stats[cls]["tp"] += 1
                else:
                    type_stats[cls]["fp"] += 1
                    type_stats[cls]["fn"] += 1
            else:
                type_stats[cls]["fn"] += 1

        # Count FP for unused predictions
        used_preds = {id(pred) for _, pred, iou_val in matches if pred and iou_val >= iou_threshold}
        for pred in pred_boxes:
            if id(pred) not in used_preds:
                type_stats[pred["class_name"]]["fp"] += 1

        # Compute precision, recall, and F1 by box type
        metrics_by_type = {}
        for cls, s in type_stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics_by_type[cls] = {"precision": precision, "recall": recall, "f1": f1}

        # Exact Match Rate (EMR) per page
        emr = sum(1 for gt, pred, iou_val in matches
                  if pred and iou_val >= iou_threshold and pred["class_name"] == gt["class_name"]) / len(
            gt_boxes) if gt_boxes else 0

        per_page_metrics.append({
            "page_id": page_id,
            "avg_iou": avg_iou,
            "emr": emr,
            "metrics_by_type": metrics_by_type
        })

    # Global metrics
    mean_f1_global = np.mean(
        [f1["f1"] for page in per_page_metrics for f1 in page["metrics_by_type"].values()]) if per_page_metrics else 0
    composite_score_global = alpha * np.mean(avg_ious) + (1 - alpha) * mean_f1_global

    global_metrics = {
        "avg_iou": np.mean(avg_ious) if avg_ious else 0,
        "composite_score": composite_score_global,
        "mean_f1": mean_f1_global
    }

    return {"global_metrics": global_metrics, "per_page_metrics": per_page_metrics}


def evaluate_model(model_type: str, model_name: str, corpus: str, split: str):
    """
    Evaluates a model on a dataset split and saves metrics.

    Args:
        model_type: 'tesseract', 'trOCR', 'granite', or 'docLayout'
        model_name: Name of the fine-tuned model
        corpus: Corpus name
        split: Dataset split ('train', 'val', 'test')
    """
    if model_type.lower() == 'doclayout':
        gt_path = os.path.join('docLayout', corpus, 'labels', split, 'gt_boxes.json')
        pred_path = os.path.join('docLayout', 'predictions', model_name, corpus, split, 'pred_boxes.json')
        metrics = evaluate_layout_json(gt_json_path=gt_path, pred_json_path=pred_path)
    else:
        gt_path = os.path.join('data', corpus, split, 'txt')
        pred_path = os.path.join(model_type, 'predictions', model_name, corpus, split)
        metrics = evaluate_ocr(pred_dir=pred_path, gt_dir=gt_path)

    # Save metrics
    out_dir = os.path.join(model_type, 'evaluation', corpus, split)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'metrics-{model_name}.json')
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def predict_on_dataset(model_type: str,
                       model_name: str,
                       split: str,
                       eval: bool,
                       corpus: str,
                       iou_threshold=0.5):
    """
    Runs the model's prediction function on a dataset split and optionally evaluates it.

    Args:
        model_type: 'tesseract', 'trOCR', 'granite', or 'docLayout'
        model_name: Name of the model
        split: Dataset split
        eval: Whether to compute evaluation metrics
        corpus: Corpus name
        iou_threshold: IoU threshold for layout models
    """
    dataset_path = os.path.join('data', corpus, split)

    # Dynamically import the model's predict function
    predict_module = importlib.import_module(f"{model_type}.inference")
    predict_fn = getattr(predict_module, "predict_on_dataset")

    folder = 'tiff' if model_type == 'tesseract' else 'images'
    images_path = os.path.join(dataset_path, folder)

    predict_fn(images_path, model_name, corpus=corpus)

    if eval:
        evaluate_model(model_type=model_type, model_name=model_name, split=split, corpus=corpus)