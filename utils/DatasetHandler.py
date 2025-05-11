import os
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import transforms, datasets
import torchvision.models as models
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import gc


# Modified FilteredImageDataset class with Pterygium filtering
class FilteredImageDataset(Dataset):
    def __init__(self, dataset, excluded_classes=None):
        """
        Create a filtered dataset that excludes specific classes.

        Args:
            dataset: Original dataset (ImageFolder or similar)
            excluded_classes: List of class names to exclude (e.g., ["Pterygium"])
        """
        self.dataset = dataset
        self.excluded_classes = excluded_classes or []

        # Get original class information
        self.orig_classes = dataset.classes
        self.orig_class_to_idx = dataset.class_to_idx

        # Create indices of samples to keep (excluding specified classes)
        self.indices = []
        for idx, (_, target) in enumerate(dataset.samples):
            class_name = self.orig_classes[target]
            if class_name not in self.excluded_classes:
                self.indices.append(idx)

        # Create new class mapping without excluded classes
        remaining_classes = [
            c for c in self.orig_classes if c not in self.excluded_classes
        ]
        self.classes = remaining_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(remaining_classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Create a mapping from old indices to new indices
        self.target_mapping = {}
        for old_class, old_idx in self.orig_class_to_idx.items():
            if old_class in self.class_to_idx:
                self.target_mapping[old_idx] = self.class_to_idx[old_class]

        print(f"Filtered out classes: {self.excluded_classes}")
        print(f"Remaining classes: {self.classes}")
        print(
            f"Original dataset size: {len(dataset)}, Filtered dataset size: {len(self.indices)}"
        )

    def __getitem__(self, index):
        """Get item from the filtered dataset with remapped class labels."""
        orig_idx = self.indices[index]
        img, old_target = self.dataset[orig_idx]

        # Remap target to new class index
        new_target = self.target_mapping[old_target]

        return img, new_target

    def __len__(self):
        """Return the number of samples in the filtered dataset."""
        return len(self.indices)

    # Allow transform to be updated
    def set_transform(self, transform):
        """Update the transform for the dataset."""
        self.dataset.transform = transform
