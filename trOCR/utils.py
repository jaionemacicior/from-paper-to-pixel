from torch.utils.data import Dataset
from PIL import Image
import os

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_path'][idx]
        text = self.df['text'][idx]

        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        pixel_values = encoding.pixel_values.squeeze()  # (3, H, W)
        labels = encoding.labels.squeeze()  # (max_target_length, )

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }