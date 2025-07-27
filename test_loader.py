from dataset_loader import load_data

dataset = load_data(split="train")
print(f"Total samples: {len(dataset)}")
image, text = dataset[0]
print(f"Text: {text}")
print(f"Image shape: {image.shape}")  # If transformed
