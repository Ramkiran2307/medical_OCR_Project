from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

def infer(image_path: str, model_or_path, processor=None):
    if isinstance(model_or_path, str):
        processor = TrOCRProcessor.from_pretrained(model_or_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_or_path)
    else:
        model = model_or_path
    
    if processor is None:
        raise ValueError("Processor must be provided if model is passed as an object.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find image at: {image_path}")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()