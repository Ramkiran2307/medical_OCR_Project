import os
import json

image_folder = "medical-prescription-dataset/train/images"
output_json = "train.json"  # You’ve placed it in root folder

data = []

# Loop through image files
for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        data.append({
            "image": f"medical-prescription-dataset/train/images/{filename}",
            "text": "dummy text"
        })

# Write to JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ train.json created with {len(data)} entries.")

