import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import pandas as pd
import os
import json
from PIL import Image
from torchvision import transforms

# Import Pre-trained Image Models (TIMM)
import timm

from EyeDiseaseDetection.util.Trainer import (
    EyeDiseaseClassifier,
    train_model,
    evaluate_model,
)


class DataHandler:
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        data_map: dict = None,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_map = (
            data_map
            if data_map
            else {
                "train": os.path.join(data_path, "train_dataset.pt"),
                "val": os.path.join(data_path, "val_dataset.pt"),
                "test": os.path.join(data_path, "test_dataset.pt"),
            }
        )

    def load_data(self) -> dict:
        DataLoaders = {}
        for split, path in self.data_map.items():
            dataset = torch.load(path)  # Placeholder for actual dataset loading
            DataLoaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True if split == "train" else False,
                num_workers=self.num_workers,
            )
        return DataLoaders


class ModelHandler:
    def __init__(self, model_name: str, dataloaders: dict, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloaders
        self.config_path = config_path
        self.config = self.load_config() if config_path else {}
        self.model = self.initialize_model(model_name=model_name, config=self.config)
        self.history = None
        self.hyper_params = self.config.get(
            "hyper_params", {"learning_rate": 0.001, "weight_decay": 1e-5}
        )

    def load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            config = json.load(f)
        return config

    def initialize_model(
        self, model_name: str = None, config: dict = None
    ) -> EyeDiseaseClassifier:
        """
        Initialize the model based on the model name and hyperparameters.

        Returns:
            nn.Module: The initialized model.
        """
        if config is None:
            config = {}

        model = EyeDiseaseClassifier(
            model_name=model_name,
            num_classes=config.get("num_classes", 9),
            pretrained=config.get("pretrained", False),
            in_channels=config.get("in_channels", 3),
            dropout_rate=config.get("drop_rate", 0.2),
        )
        model.to(self.device)
        return model

    def train_model(
        self, num_epochs: int = 25, use_amp: bool = False, save_dir: str = None
    ):
        """
        Train the model using the provided dataloaders and hyperparameters.

        Args:
            num_epochs (int): Number of epochs to train the model.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hyper_params.get("learning_rate", 0.001),
            weight_decay=self.hyper_params.get("weight_decay", 1e-5),
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.model, self.history = train_model(
            model=self.model,
            dataloaders=self.dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            num_epochs=num_epochs,
            use_amp=use_amp,
            save_dir=save_dir,
        )

        return self.model, self.history

    def test_model(self, class_names: dict = None, save_dir: str = None):
        """
        Evaluate the model using the test dataloader.

        Returns:
            dict: Evaluation metrics.
        """
        metrics = evaluate_model(
            model=self.model,
            test_loader=self.dataloader["test"],
            device=self.device,
            class_names=class_names,
            save_dir=save_dir,
        )
        return metrics

    def save_model(self, save_path: str, save_config: bool = True):
        """
        Save the trained model to the specified path.

        Args:
            save_path (str): Path to save the model.
            save_config (bool): Whether to save model configuration alongside weights.
        """
        # Save model state
        torch.save(self.model.state_dict(), save_path)

        # Optionally save configuration
        if save_config and self.config:
            config_path = os.path.splitext(save_path)[0] + "_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            print(f"Model configuration saved to {config_path}")

        print(f"Model saved to {save_path}")

    def load_model(self, model_path: str, config_path: str = None):
        """
        Load a pretrained model from a specified path.

        Args:
            model_path (str): Path to the saved model weights.
            config_path (str): Path to model configuration (optional).

        Returns:
            EyeDiseaseClassifier: The loaded model.
        """
        # Load configuration if provided
        if config_path:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        elif not self.config:
            # Try to find config with same name pattern
            potential_config = os.path.splitext(model_path)[0] + "_config.json"
            if os.path.exists(potential_config):
                with open(potential_config, "r") as f:
                    self.config = json.load(f)

        # Re-initialize model architecture if config has changed
        if self.config:
            self.model = self.initialize_model(
                model_name=self.model.model_name, config=self.config
            )

        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        return self.model

    def inference(
        self,
        image_path: str = None,
        image: Image.Image = None,
        preprocess: callable = None,
    ):
        """
        Run inference on a single image.

        Args:
            image_path (str, optional): Path to the image file.
            image (PIL.Image, optional): Already loaded PIL image.
            preprocess (callable, optional): Custom preprocessing function.

        Returns:
            tuple: Predicted class index and probability.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Load image if path is provided
        if image_path and not image:
            image = Image.open(image_path).convert("RGB")
        elif not image:
            raise ValueError("Either image_path or image must be provided")

        # Default preprocessing if none provided
        if preprocess is None:
            preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Preprocess the image
        img_tensor = preprocess(image).unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            with autocast():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            # Get the predicted class and probability
            prob, predicted_class = torch.max(probabilities, 1)

        return predicted_class.item(), prob.item()

    def batch_inference(
        self, image_paths: list = None, images: list = None, preprocess: callable = None
    ):
        """
        Run inference on a batch of images.

        Args:
            image_paths (list, optional): List of paths to image files.
            images (list, optional): List of already loaded PIL images.
            preprocess (callable, optional): Custom preprocessing function.

        Returns:
            tuple: Lists of predicted class indices and probabilities.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Process images
        if image_paths and not images:
            images = [Image.open(path).convert("RGB") for path in image_paths]
        elif not images:
            raise ValueError("Either image_paths or images must be provided")

        # Default preprocessing if none provided
        if preprocess is None:
            preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Preprocess all images
        img_tensors = torch.stack([preprocess(img) for img in images]).to(self.device)

        # Perform inference
        with torch.no_grad():
            with autocast():
                outputs = self.model(img_tensors)
                probabilities = torch.softmax(outputs, dim=1)

            # Get the predicted classes and probabilities
            probs, predicted_classes = torch.max(probabilities, 1)

        return predicted_classes.tolist(), probs.tolist()
