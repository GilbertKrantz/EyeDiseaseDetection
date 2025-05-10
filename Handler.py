import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import pandas as pd
import os
import json
from PIL import Image
from torchvision import transforms
from collections import Counter

import logging

from util.Trainer import EyeDiseaseClassifier, train_model, evaluate_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataHandler:
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        data_map: dict = None,
        use_weighted_sampler: bool = True,
        beta: float = 0.9999,  # For effective number of samples weighting
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler
        self.beta = beta
        self.data_map = (
            data_map
            if data_map
            else {
                "train": os.path.join(data_path, "train_dataset.pt"),
                "val": os.path.join(data_path, "val_dataset.pt"),
                "test": os.path.join(data_path, "test_dataset.pt"),
            }
        )
        self.class_weights = None

    def calculate_class_weights(self, dataset):
        """Calculate class weights based on dataset distribution"""
        # Get all targets from the dataset
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        else:
            # Fallback for datasets without a direct targets attribute
            targets = [
                sample[1] for sample in dataset
            ]  # Assuming dataset returns (input, target)

        # Count occurrences of each class
        class_counts = Counter(targets)
        num_samples = len(targets)
        num_classes = len(class_counts)

        # Method 1: Inverse frequency weighting
        weights_inv = {
            cls: num_samples / (count * num_classes)
            for cls, count in class_counts.items()
        }

        # Method 2: Effective Number of Samples weighting (from paper "Class-Balanced Loss")
        weights_ens = {}
        for cls, count in class_counts.items():
            effective_num = 1.0 - self.beta**count
            weights_ens[cls] = (1.0 - self.beta) / effective_num

        # Normalize weights_ens
        total_ens = sum(weights_ens.values())
        weights_ens = {
            cls: weight / total_ens * num_classes for cls, weight in weights_ens.items()
        }

        # Store both weighting methods (default to ens)
        self.class_weights = {
            "inverse": weights_inv,
            "ens": weights_ens,
            "counts": class_counts,
        }

        return weights_ens

    def create_weighted_sampler(self, dataset):
        """Create a weighted sampler for the dataset based on class distribution"""
        # Get all targets from the dataset
        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        else:
            # Fallback for datasets without a direct targets attribute
            targets = [sample[1] for sample in dataset]

        # Calculate weights if not already done
        if self.class_weights is None:
            logging.info("Calculating class weights.")
            class_weights = self.calculate_class_weights(dataset)
            logging.info("Class weights calculated.")
        else:
            logging.info("Using previously calculated class weights.")
            class_weights = self.class_weights["ens"]

        # Assign weight to each sample based on its class
        sample_weights = [class_weights[target] for target in targets]
        sample_weights = torch.DoubleTensor(sample_weights)

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return sampler

    def get_loss_class_weights(self, device=None):
        """Return class weights tensor for loss function"""
        if self.class_weights is None:
            return None

        # Get list of classes (should be sorted)
        classes = sorted(self.class_weights["ens"].keys())
        weights = [self.class_weights["ens"][cls] for cls in classes]

        # Convert to tensor
        weights_tensor = torch.FloatTensor(weights)

        # Move to device if specified
        if device:
            weights_tensor = weights_tensor.to(device)

        return weights_tensor

    def load_data(self) -> dict:
        """Load datasets and create dataloaders with optional weighted sampling"""
        DataLoaders = {}
        for split, path in self.data_map.items():
            dataset = torch.load(path)  # Load dataset

            # Only apply weighted sampling to training data
            if split == "train" and self.use_weighted_sampler:
                logging.info(f"Using weighted sampling for {split} dataset.")
                # Calculate class weights for the first time
                if self.class_weights is None:
                    logging.info("Calculating class weights.")
                    self.calculate_class_weights(dataset)
                    logging.info("Class weights calculated.")

                else:
                    logging.info("Using previously calculated class weights.")
                # Create weighted sampler
                sampler = self.create_weighted_sampler(dataset)

                # Create dataloader with sampler (note: can't use shuffle with sampler)
                DataLoaders[split] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    num_workers=self.num_workers,
                )
            else:
                # Regular dataloader for validation/test sets
                DataLoaders[split] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == "train"),  # Only shuffle training data
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
            dropout_rate=config.get("drop_rate", 0.2),
            unfreeze_last_n_layers=config.get("unfreeze_last_n_layers", 0),
        )
        model.to(self.device)
        return model

    def train_model(
        self,
        num_epochs: int = 25,
        use_amp: bool = False,
        save_dir: str = None,
        class_weights: torch.Tensor = None,
    ):
        """
        Train the model using the provided dataloaders and hyperparameters.

        Args:
            num_epochs (int): Number of epochs to train the model.
            class_weights (torch.Tensor): Optional class weights for loss function.
        """

        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        criterion = (
            nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )
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
