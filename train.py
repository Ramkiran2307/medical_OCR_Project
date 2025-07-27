import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
from dataset_loader import load_data

def train_model():
    # Define model and processor
    MODEL_NAME = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configure model for training
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Load dataset
    train_dataset = load_data(split="train")
    eval_dataset = load_data(split="val")

    ## FIX: Corrected collate function to handle a list of dictionaries from the dataset
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels = processor(text=texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100 # Ignore padding in loss
        
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):  # Adjust number of epochs
        # --- Training Phase ---
        model.train()
        train_loop = tqdm(train_dataloader, leave=True, desc=f"Epoch {epoch + 1}/{3} [Training]")
        for batch in train_loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loop.set_postfix(loss=loss.item())

        # --- Validation Phase ---
        model.eval()
        eval_loss = 0.0
        eval_loop = tqdm(eval_dataloader, leave=True, desc=f"Epoch {epoch + 1}/{3} [Validation]")
        with torch.no_grad():
            for batch in eval_loop:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()
        print(f"Epoch {epoch + 1} - Validation Loss: {eval_loss / len(eval_dataloader):.4f}")

    print("✅ Training complete.")

    ## CHANGE: Save the trained model and processor for later use
    output_dir = "./medical_ocr_model_final"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}")

    return model, processor