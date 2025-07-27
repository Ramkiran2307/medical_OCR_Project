import os
import json
from PIL import Image
from torch.utils.data import Dataset

class MedicalPrescriptionDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.image_dir = os.path.join(root_dir, split, "images")
        self.annotation_dir = os.path.join(root_dir, split, "annotations")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.annotations = self._load_annotations()
        
        self.image_files = [img_file for img_file in self.image_files if img_file in self.annotations]

    def _load_annotations(self):
        annotations = {}
        all_image_basenames = {os.path.splitext(f)[0] for f in self.image_files}
        
        for ann_file in os.listdir(self.annotation_dir):
            if ann_file.endswith(".json"):
                basename = os.path.splitext(ann_file)[0]
                if basename in all_image_basenames:
                    img_filename = next((f for f in self.image_files if os.path.splitext(f)[0] == basename), None)
                    if img_filename:
                        filepath = os.path.join(self.annotation_dir, ann_file)
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            annotations[img_filename] = data.get("text", "")
        return annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        image = Image.open(image_path).convert("RGB")
        text = self.annotations[image_file]

        return {"image": image, "text": text}

def load_data(split="train"):
    ## CHANGE: Updated with your specific dataset path.
    root_dir = r"C:\Users\ramki\OneDrive\Desktop\medical_OCR_Project\medical-prescription-dataset"
    return MedicalPrescriptionDataset(root_dir=root_dir, split=split)