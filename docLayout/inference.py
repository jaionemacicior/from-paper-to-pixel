import os
import json
from tqdm import tqdm
from doclayout_yolo import YOLOv10
from docLayout.utils import save_image_with_boxes

# Paths for models and predictions
BASE_DIR = 'docLayout'
MODEL_DIR = os.path.join(BASE_DIR, "models")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)


def predict(image_path, model, model_name, class_names=None, conf=0.2):
    """
    Detects document blocks using DocLayout YOLOv10.

    Args:
        image_path (str): Path to the image (JPG): data/<corpus>/<split>/images/xxx.jpg
        model: YOLOv10 model
        model_name (str): Model name
        class_names (list): List of class names indexed by class id
        conf (float): Confidence threshold

    Returns:
        list: List of dictionaries for each detected box:
              [{'class_name': str, 'xyxy': [x1,y1,x2,y2], 'confidence': float}, ...]
    """
    split = image_path.split("/")[2]
    corpus = image_path.split("/")[1]
    out_dir = os.path.join(PRED_DIR, model_name, corpus, split)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "pred_boxes.json")
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # Run prediction
    det_res = model.predict(image_path, imgsz=1024, conf=conf, device="cuda:0")
    results = det_res[0]
    boxes = results.boxes

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()

    detections = []
    for box, conf_score, cls_id in zip(boxes_xyxy, confidences, class_ids):
        cls_name = class_names[int(cls_id)] if class_names else str(int(cls_id))
        detections.append({
            "class_name": cls_name,
            "xyxy": box.tolist(),
            "confidence": float(conf_score)
        })

    # Load existing JSON or create a new one
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_predictions = json.load(f)
    else:
        all_predictions = {}

    # Update predictions for this image
    all_predictions[image_id] = detections

    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    # Save image with boxes
    save_image_with_boxes(image_path, detections, save_dir=out_dir)


def predict_on_dataset(folder_path, model_name, corpus):
    """
    Runs YOLOv10 predictions on all images in a folder.

    Args:
        folder_path (str): Path to the folder containing images
        model_name (str): Model name
        corpus (str): Corpus name
    """
    # Load class names
    classes_path = os.path.join('data', corpus, 'classes.json')
    with open(classes_path, 'r') as f:
        class_to_id = json.load(f)

    num_classes = max(class_to_id.values()) + 1
    class_names = [None] * num_classes
    for cls_name, cls_id in class_to_id.items():
        class_names[cls_id] = cls_name

    # Load model
    if model_name == 'baseline':
        model_path = os.path.join(MODEL_DIR, 'doclayout_yolo_docstructbench_imgsz1024.pt')
    else:
        model_path = os.path.join(MODEL_DIR, model_name, 'train', 'weights', 'best.pt')

    model = YOLOv10(model_path)

    # Run prediction for all images
    for file in tqdm(os.listdir(folder_path), desc=f"DocLayout YOLO predictions ({model_name})"):
        image_path = os.path.join(folder_path, file)
        predict(image_path=image_path, model=model, model_name=model_name, class_names=class_names)
