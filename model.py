from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def get_model():
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    return model, processor
