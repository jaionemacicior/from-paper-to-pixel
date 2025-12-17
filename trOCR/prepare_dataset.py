import os
import pandas as pd
import argparse

def create_dataset(txt_folder: str, image_folder: str) -> pd.DataFrame:
    """
    Prepare a dataset for trOCR fine-tuning by matching text and image files.

    Args:
        txt_folder (str): Folder containing .txt files with OCR text.
        image_folder (str): Folder containing corresponding images (.jpg or .tiff).

    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'text', 'image_path'].
    """
    # List all text and image files
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.tiff', '.png'))]

    # Map filenames (without extension) to text content
    txt_dict = {}
    for txt_file in txt_files:
        base_name, ext = os.path.splitext(txt_file)
        # Remove possible ".gt" suffix in filename
        if base_name.endswith(".gt"):
            base_name = base_name[:-3]
        with open(os.path.join(txt_folder, txt_file), 'r', encoding='utf-8') as f:
            content = f.read().strip()
        txt_dict[base_name] = content

    # Map filenames (without extension) to image paths
    image_dict = {os.path.splitext(f)[0]: os.path.join(image_folder, f) for f in image_files}

    # Keep only files that exist in both text and image folders
    common_keys = set(txt_dict.keys()) & set(image_dict.keys())

    # Build dataset as a list of dictionaries
    data = []
    for key in sorted(common_keys):
        text = txt_dict.get(key, "")
        data.append({
            'filename': key,
            'text': text,
            'image_path': image_dict[key]
        })

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    return df

def prepare_trocr_dataset(corpus_name):
    base_dir = 'trOCR'
    corpus_dir = os.path.join('data', corpus_name)

    # Ensure output directory exists
    os.makedirs(os.path.join(base_dir, corpus_name), exist_ok=True)

    # Process train/val/test splits
    for split in ['train', 'val', 'test']:
        txt_folder = os.path.join(corpus_dir, split, 'txt')
        image_folder = os.path.join(corpus_dir, split, 'images')

        df = create_dataset(txt_folder, image_folder)

        # Filter out empty or null text rows
        df = df[df['text'].notna() & (df['text'].str.strip() != "")]
        df.reset_index(drop=True, inplace=True)

        # Save to CSV
        output_csv = os.path.join(base_dir, corpus_name, f"{split}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Dataset for {split} saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for trOCR fine-tuning")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus folder inside 'data/'")
    args = parser.parse_args()
    prepare_trocr_dataset(args.corpus)
