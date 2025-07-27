import torch
from train import train_model
from infer import infer

def main():
    # --- Option 1: Train the model and then immediately infer ---
    # This trains the model from scratch every time you run main.py
    print("Starting model training...")
    trained_model, trained_processor = train_model()
    
    print("\nPerforming inference with the newly trained model...")
    result = infer("sample_image.png", trained_model, trained_processor)
    print(f"\n✅ Predicted text for 'sample_image.png': {result}")


    # --- Option 2: Load a pre-trained model and infer (faster) ---
    # After training once, you can comment out Option 1 and uncomment this block.
    """
    print("\nSkipping training. Loading model from disk...")
    model_path = "./medical_ocr_model_final"
    
    print(f"Performing inference with model from: {model_path}")
    result = infer("sample_image.png", model_path)
    print(f"\n✅ Predicted text for 'sample_image.png': {result}")
    """

if __name__ == "__main__":
    main()