from PIL import Image, ImageDraw, ImageFont
import os

def save_image_with_boxes(image_path: str, detections: list, save_dir: str):
    """
    Dibuja las cajas de detección sobre la imagen y guarda el resultado.

    Args:
        image_path (str): ruta de la imagen original
        detections (list): lista de detecciones [{'class_name', 'xyxy', 'confidence'}, ...]
        save_dir (str): carpeta donde guardar la imagen con cajas
    """
    os.makedirs(save_dir, exist_ok=True)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{image_id}_boxed.jpg")

    # abrir imagen
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # intentar cargar fuente, sino usar default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # dibujar cajas y etiquetas
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        # Asegurarse de que x1<x2 y y1<y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cls_name = det["class_name"]
        conf = det["confidence"]

        # dibujar rectángulo
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # dibujar etiqueta
        label = f"{cls_name} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)  # corregido
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    # guardar imagen con cajas
    image.save(out_path)
    print(f"Imagen con cajas guardada en: {out_path}")