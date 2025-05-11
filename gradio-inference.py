#!/usr/bin/env python
# Eye Disease Detection - Gradio Inference App
# Date: May 11, 2025

import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)

# Import custom modules
sys.path.append("./utils")
from ModelCreator import EyeDetectionModels

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define class names (make sure these match your model's classes)
CLASSES = [
    "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]


def get_transform():
    """Get the standard transformation pipeline for inference."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path, model_type="efficientvit"):
    """
    Load a pretrained model for inference.

    Args:
        model_path: Path to the saved model state dict
        model_type: Type of model to load (mobilenetv4, levit, efficientvit, gernet, regnetx)

    Returns:
        Loaded model ready for inference
    """
    # Initialize model creator
    logging.info("Initializing model creator...")
    model_creator = EyeDetectionModels(
        num_classes=len(CLASSES), freeze_layers=False  # Not relevant for inference
    )

    # Check if model type exists
    if model_type not in model_creator.models:
        raise ValueError(
            f"Model type '{model_type}' not found. Available models: {list(model_creator.models.keys())}"
        )

    # Create model of specified type
    logging.info(f"Creating model of type: {model_type}")
    model = model_creator.models[model_type]()

    # Load state dict if provided
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
    elif model_path is None:
        #  Use default model path if it exists
        if os.path.exists(f"./weights/{model_type}.pth"):
            model_path = f"./weights/{model_type}.pth"
        else:
            model_path = None
            logging.warning(
                f"Default model path '{model_path}' not found. Using untrained model."
            )
    # Set model to evaluation mode
    model.eval()
    return model


def predict_image(image, model_path, model_type):
    """
    Predict eye disease from an uploaded image.

    Args:
        image: Input image from Gradio
        model_path: Path to the model state dict
        model_type: Type of model architecture

    Returns:
        Dictionary of class probabilities
    """
    try:

        logging.info("Starting prediction...")
        # Load model
        model = load_model(model_path, model_type)

        # Preprocess image
        logging.info("Preprocessing image...")
        if image is None:
            logging.warning("No image provided.")
            return {cls: 0.0 for cls in CLASSES}
        transform = get_transform()
        if image is None:
            return {cls: 0.0 for cls in CLASSES}

        # Convert numpy array to PIL Image
        img = Image.fromarray(image).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        logging.info("Image preprocessed successfully.")

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        # Return probabilities for each class
        return {cls: float(prob) for cls, prob in zip(CLASSES, probabilities)}

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {cls: 0.0 for cls in CLASSES}


def main():
    """Main function to run the Gradio interface."""
    # Define available models
    model_types = ["mobilenetv4", "levit", "efficientvit", "gernet", "regnetx"]

    # Create the Gradio interface
    with gr.Blocks(title="Eye Disease Detection") as demo:
        gr.Markdown("# Eye Disease Detection System")
        gr.Markdown(
            """This application uses deep learning to detect eye diseases from fundus images.
                    Currently supports detection of: 
                    - Central Serous Chorioretinopathy
                    - Diabetic Retinopathy
                    - Disc Edema
                    - Glaucoma
                    - Healthy (normal eye)
                    - Macular Scar
                    - Myopia
                    - Retinal Detachment
                    - Retinitis Pigmentosa
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Fundus Image", type="numpy")
                model_path = gr.Textbox(
                    label="Model Path (leave empty to use default)",
                    placeholder="Path to model .pth file",
                    value="",
                )
                model_type = gr.Dropdown(
                    label="Model Architecture", choices=model_types, value="mobilenetv4"
                )
                submit_btn = gr.Button("Analyze Image", variant="primary")

            with gr.Column():
                output_chart = gr.Label(label="Prediction")

        # Process the image when the button is clicked
        submit_btn.click(
            fn=predict_image,
            inputs=[input_image, model_path, model_type],
            outputs=output_chart,
        )

        # Examples section
        gr.Markdown("### Examples (Please add your own example images)")
        gr.Examples(
            examples=[],  # Add example paths here
            inputs=input_image,
            outputs=[output_chart],
            fn=predict_image,
            cache_examples=True,
        )

        # Usage instructions
        with gr.Accordion("Usage Instructions", open=False):
            gr.Markdown(
                """
            ## How to use this application:
            
            1. **Upload an image**: Click the upload button to select a fundus image from your computer
            2. **Specify model** (Optional): 
                - Enter the path to your trained model file (.pth)
                - Select the model architecture that was used for training
            3. **Analyze**: Click the "Analyze Image" button to get results
            4. **Interpret results**: The system will show the detected condition and probability distribution
            
            ## Model Information:
            
            This system supports multiple model architectures:
            - **MobileNetV4**: Lightweight and efficient model
            - **LeViT**: Vision Transformer designed for efficiency
            - **EfficientViT**: Hybrid CNN-Transformer architecture
            - **GENet**: General and Efficient Network
            - **RegNetX**: Systematically designed CNN architecture
            
            For best results, ensure you're using a high-quality fundus image and the correct model type.
            """
            )

    # Launch the app
    demo.launch(
        share=True,
        pwa=True,
    )


if __name__ == "__main__":
    main()
