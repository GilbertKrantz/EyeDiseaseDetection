---
title: Eye Disease Detection Models
emoji: 🛕🛕
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: gradio-inference.py
pinned: true
short_description: Eye disease detection using deep learning models
license: apache-2.0
---


# Eye Disease Detection 

This repository contains a Gradio web application for eye disease detection using deep learning models. The application allows users to upload fundus photographs and get predictions for common eye conditions.

## Features

- **Easy-to-use web interface** for eye disease detection
- Support for **multiple model architectures** (MobileNetV4, LeViT, EfficientViT, GENet, RegNetX)
- **Custom model loading** from saved model checkpoints
- **Visualization** of prediction probabilities
- **Dockerized deployment** option

## Supported Eye Conditions

The system can detect the following eye conditions:
- Central Serous Chorioretinopathy
- Diabetic Retinopathy
- Disc Edema
- Glaucoma
- Healthy (normal eye)
- Macular Scar
- Myopia
- Retinal Detachment
- Retinitis Pigmentosa

## Installation

### Prerequisites

- Python 3.12+
- PyTorch 2.7.0+
- CUDA-compatible GPU (optional, but recommended for faster inference)

### Option 1: Local Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/GilbertKrantz/eye-disease-detection.git
   cd eye-disease-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python gradio_inference.py
   ```

4. Open your browser and go to http://localhost:7860

### Option 2: Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t eye-disease-detection .
   ```

2. Run the container:
   ```bash
   docker run -p 7860:7860 eye-disease-detection
   ```

3. Open your browser and go to http://localhost:7860

## Usage

1. Upload a fundus image of the eye
2. (Optional) Specify the path to your trained model file (.pth)
3. Select the model architecture (MobileNetV4, LeViT, EfficientViT, GENet, RegNetX)
4. Click "Analyze Image" to get the prediction
5. View the results and probability distribution

## Model Training

This repository focuses on inference. For training your own models, refer to the main training script and follow these steps:

1. Prepare your dataset in the required directory structure
2. Train a model using the main.py script:
   ```bash
   python main.py --train-dir "/path/to/training/data" --eval-dir "/path/to/eval/data" --model mobilenetv4 --epochs 20 --save-model "my_model.pth"
   ```
3. Use the saved model with the inference application

## Project Structure

```
.
├── gradio_inference.py     # Main Gradio application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # This documentation
├── utils/                  # Utility modules
│   ├── ModelCreator.py     # Model architecture definitions
│   ├── Evaluator.py        # Model evaluation utilities
│   ├── DatasetHandler.py   # Dataset handling utilities
│   ├── Trainer.py          # Model training utilities
│   └── Callback.py         # Training callbacks
└── main.py                 # Main training script
```

## Performance

The performance of the models depends on the quality of training data and the specific architecture used. In general, these models can achieve accuracy rates of 85-95% on standard eye disease datasets.

## Customization

You can customize the application in several ways:
- Add example images in the Gradio interface
- Extend the list of supported classes by modifying the CLASSES variable in gradio_inference.py
- Add support for additional model architectures in ModelCreator.py

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- The models are built using PyTorch and the TIMM library
- The web interface is built using Gradio
- Special thanks to the open-source community for making this project possible
