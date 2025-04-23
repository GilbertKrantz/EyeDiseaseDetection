import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from collections import Counter
import numpy as np
from PIL import Image
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm


class BalancedImageDataset(Dataset):
    """
    A PyTorch Dataset that balances class distribution by adding underrepresented 
    class images from an augmented dataset to the original dataset.
    
    Supports regular Dataset objects as well as torch.utils.data.Subset objects.
    """
    
    def __init__(
        self,
        original_dataset: Union[Dataset, Subset],
        augmented_dataset: Union[Dataset, Subset],
        target_attribute: str = "targets",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        balance_strategy: str = "upsample_to_max",
        verbose: bool = True
    ):
        """
        Initialize the balanced dataset.
        
        Args:
            original_dataset: The original dataset with class imbalance (Dataset or Subset)
            augmented_dataset: Dataset containing augmented images (Dataset or Subset)
            target_attribute: Attribute name containing class labels (default: "targets")
            transform: Optional transform to be applied to images
            target_transform: Optional transform to be applied to labels
            balance_strategy: Strategy for balancing ("upsample_to_max" or "target_count")
            verbose: Whether to display progress bars (default: True)
        """
        self.original_dataset = original_dataset
        self.augmented_dataset = augmented_dataset
        self.target_attribute = target_attribute
        self.transform = transform
        self.target_transform = target_transform
        self.balance_strategy = balance_strategy
        self.verbose = verbose
        
        # Extract class information
        print("Analyzing class distribution in original dataset...")
        self.original_classes = self._get_class_counts(original_dataset)
        print("Analyzing class distribution in augmented dataset...")
        self.augmented_classes = self._get_class_counts(augmented_dataset)
        
        # Build balanced dataset
        print("Building balanced dataset...")
        self.indices, self.targets = self._build_balanced_dataset()
        print(f"Balanced dataset created with {len(self.indices)} samples")
        
    def _get_targets_from_dataset(self, dataset: Union[Dataset, Subset]) -> List:
        """Extract targets from a dataset, handling both Dataset and Subset cases."""
        # Case 1: If dataset is a Subset
        if isinstance(dataset, Subset):
            # Try to get targets from the dataset attribute of the Subset
            if hasattr(dataset.dataset, self.target_attribute):
                all_targets = getattr(dataset.dataset, self.target_attribute)
                # Only return targets for indices in the subset
                if isinstance(all_targets, list) or isinstance(all_targets, np.ndarray):
                    return [all_targets[i] for i in dataset.indices]
                else:
                    # If targets is not a list-like object, iterate through indices
                    return []  # Will fall back to iteration method
            else:
                return []  # Will fall back to iteration method
                
        # Case 2: Regular dataset with targets attribute
        elif hasattr(dataset, self.target_attribute):
            return getattr(dataset, self.target_attribute)
            
        # Case 3: No targets attribute found
        return []
        
    def _get_class_counts(self, dataset: Union[Dataset, Subset]) -> Dict[int, int]:
        """Extract class distribution from dataset, handling both Dataset and Subset."""
        targets = self._get_targets_from_dataset(dataset)
        
        if targets:
            return Counter(targets)
        else:
            # If targets are not directly accessible, iterate through dataset
            extracted_targets = []
            iterator = tqdm(range(len(dataset))) if self.verbose else range(len(dataset))
            for i in iterator:
                _, target = dataset[i]
                extracted_targets.append(target)
            return Counter(extracted_targets)
        
    def _build_balanced_dataset(self) -> Tuple[List[Tuple[int, bool]], List[int]]:
        """
        Build a balanced dataset by selecting samples from both datasets.
        Returns a list of (index, is_augmented) tuples and corresponding targets.
        """
        indices = []
        targets = []
        
        # Determine target count for each class
        if self.balance_strategy == "upsample_to_max":
            max_count = max(self.original_classes.values())
            target_counts = {cls: max_count for cls in self.original_classes.keys()}
        else:
            # You can implement other strategies like "target_count" here
            max_count = max(self.original_classes.values())
            target_counts = {cls: max_count for cls in self.original_classes.keys()}
        
        # First, include all samples from original dataset
        print("Adding samples from original dataset...")
        orig_targets = self._get_targets_from_dataset(self.original_dataset)
        
        # Use different approach based on whether targets are directly accessible
        if orig_targets:
            # If we have targets list
            class_iterator = tqdm(self.original_classes.keys(), desc="Processing classes") if self.verbose else self.original_classes.keys()
            
            for cls in class_iterator:
                for i, target in enumerate(orig_targets):
                    if target == cls:
                        indices.append((i, False))  # (index, is_augmented)
                        targets.append(target)
        else:
            # If we need to iterate through the dataset
            iterator = tqdm(range(len(self.original_dataset)), desc="Adding original samples") if self.verbose else range(len(self.original_dataset))
            for i in iterator:
                _, target = self.original_dataset[i]
                indices.append((i, False))
                targets.append(target)
        
        # Then add samples from augmented dataset to balance classes
        print("Balancing classes with augmented samples...")
        aug_classes_iterator = tqdm(self.original_classes.items(), desc="Balancing classes") if self.verbose else self.original_classes.items()
        
        for cls, count in aug_classes_iterator:
            needed = target_counts[cls] - count
            if needed <= 0:
                continue  # This class is already well-represented
                
            # Find matching samples in augmented dataset
            aug_samples = []
            aug_targets = self._get_targets_from_dataset(self.augmented_dataset)
            
            if aug_targets:
                # If we have targets list
                for i, target in enumerate(aug_targets):
                    if target == cls:
                        aug_samples.append(i)
            else:
                # Otherwise iterate through dataset (with progress bar if verbose)
                iterator = tqdm(range(len(self.augmented_dataset)), 
                               desc=f"Finding class {cls} in augmented data") if self.verbose else range(len(self.augmented_dataset))
                for i in iterator:
                    _, target = self.augmented_dataset[i]
                    if target == cls:
                        aug_samples.append(i)
            
            # Add needed samples (with repetition if necessary)
            if aug_samples:
                selected_indices = np.random.choice(aug_samples, size=needed, replace=len(aug_samples) < needed)
                for idx in selected_indices:
                    indices.append((idx, True))  # (index, is_augmented)
                    targets.append(cls)
        
        return indices, targets
    
    def __len__(self) -> int:
        """Return the length of the balanced dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from the balanced dataset."""
        index, is_augmented = self.indices[idx]
        dataset = self.augmented_dataset if is_augmented else self.original_dataset
        
        image, target = dataset[index]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
            
        return image, target