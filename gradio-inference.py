#!/usr/bin/env python
# Eye Disease Detection - Gradio Inference App
# Date: May 11, 2025

import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import gradio as gr
from PIL import Image
import logging

from main import get_transform

logging.basicConfig(level=logging.INFO)

# Import custom modules
from utils.ModelCreator import EyeDetectionModels

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


def load_model(model_type: str = "efficientvit") -> nn.Module:
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
    if os.path.exists(f"./weights/{model_type}.pth"):
        model_path = f"./weights/{model_type}.pth"
        logging.info(f"Loading model from default path: ./weights/{model_type}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Model loaded successfully.")
    else:
        model_path = None
        logging.warning(
            f"Default model path '{model_path}' not found. Using untrained model."
        )
    # Set model to evaluation mode
    model.eval()
    return model


class GradCAM:
    """
    GradCAM (Gradient-weighted Class Activation Mapping) implementation
    for generating attention map heatmaps.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        
        Args:
            model: The model to generate CAM for
            target_layer: The layer to compute gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            cam: Class activation map
            prediction: Model prediction
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, output


def get_target_layer(model, model_type):
    """
    Get the target layer for GradCAM based on model type.
    
    Args:
        model: The model
        model_type: Type of model
        
    Returns:
        target_layer: The layer to use for GradCAM
    """
    try:
        if model_type == "mobilenetv4":
            # For MobileNetV4, use the last convolutional layer in features
            return model.features[-1]
        elif model_type == "levit":
            # For LeViT (transformer), use the last block
            return model.blocks[-1]
        elif model_type == "efficientvit":
            # For EfficientViT, use the last stage
            return model.stages[-1]
        elif model_type == "gernet":
            # For GENet, use the last stage
            return model.stages[-1]
        elif model_type == "regnetx":
            # For RegNetX, use the last trunk layer
            return model.trunk[-1]
        else:
            # Default: try to get the last feature layer
            if hasattr(model, 'features'):
                return model.features[-1]
            elif hasattr(model, 'stages'):
                return model.stages[-1]
            elif hasattr(model, 'blocks'):
                return model.blocks[-1]
            else:
                raise ValueError(f"Cannot determine target layer for model type: {model_type}")
    except Exception as e:
        logging.warning(f"Error getting target layer: {e}. Using fallback.")
        # Fallback: try to get any reasonable last conv layer
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("Could not find suitable target layer for GradCAM")


def apply_heatmap_on_image(img, cam, alpha=0.5):
    """
    Apply CAM heatmap overlay on the original image.
    
    Args:
        img: Original image (PIL Image or numpy array)
        cam: Class activation map
        alpha: Overlay transparency
        
    Returns:
        Heatmap overlay image as numpy array
    """
    # Convert PIL to numpy if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Resize CAM to match image size
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def predict_image(image: np.ndarray, model_type: str) -> tuple[dict, np.ndarray]:
    """
    Predict eye disease from an uploaded image and generate attention heatmap.

    Args:
        image: Input image from Gradio
        model_type: Type of model architecture

    Returns:
        Tuple of (Dictionary of class probabilities, Heatmap overlay image)
    """
    try:
        logging.info("Starting prediction...")
        
        # Handle None image
        if image is None:
            logging.warning("No image provided.")
            return {cls: 0.0 for cls in CLASSES}, None
        
        # Load model
        model = load_model(model_type)
        model.to(device)

        # Preprocess image
        logging.info("Preprocessing image...")
        transform = get_transform()

        # Convert numpy array to PIL Image and keep original for heatmap
        img_pil = Image.fromarray(image).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        logging.info("Image preprocessed successfully.")

        # Get target layer for GradCAM
        try:
            target_layer = get_target_layer(model, model_type)
            logging.info(f"Using target layer: {target_layer}")
            
            # Initialize GradCAM
            grad_cam = GradCAM(model, target_layer)
            
            # Generate CAM and prediction
            cam, outputs = grad_cam.generate_cam(img_tensor)
            
            # Generate heatmap overlay
            heatmap_overlay = apply_heatmap_on_image(img_pil, cam, alpha=0.4)
            
        except Exception as e:
            logging.error(f"Error generating heatmap: {e}")
            # Fallback: just do prediction without heatmap
            with torch.no_grad():
                outputs = model(img_tensor)
            heatmap_overlay = np.array(img_pil)  # Return original image
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)[0].cpu().detach().numpy()

        # Return probabilities and heatmap
        result_dict = {cls: float(prob) for cls, prob in zip(CLASSES, probabilities)}
        
        logging.info("Prediction completed successfully.")
        return result_dict, heatmap_overlay

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {cls: 0.0 for cls in CLASSES}, None


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
                model_type = gr.Dropdown(
                    label="Model Architecture", choices=model_types, value="mobilenetv4"
                )
                submit_btn = gr.Button("Analyze Image", variant="primary")

            with gr.Column():
                output_chart = gr.Label(label="Prediction")
                output_heatmap = gr.Image(label="Attention Heatmap")

        # Process the image when the button is clicked
        submit_btn.click(
            fn=predict_image,
            inputs=[input_image, model_type],
            outputs=[output_chart, output_heatmap],
        )

        # Examples section
        gr.Markdown("### Examples (Please add your own example images)")
        gr.Examples(
            examples=[],  # Add example paths here
            inputs=input_image,
            outputs=[output_chart, output_heatmap],
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
            4. **Interpret results**: The system will show the detected condition, probability distribution, and an attention heatmap
            
            ## Attention Heatmap:
            
            The attention heatmap visualizes which regions of the fundus image the model is focusing on when making its prediction.
            - **Red/Yellow areas**: Regions the model considers most important for the diagnosis
            - **Blue/Green areas**: Regions with less influence on the prediction
            
            This helps in understanding and validating the model's decision-making process.
            
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
