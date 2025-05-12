#!/usr/bin/env python
# Eye Disease Detection - Main Application
# Date: May 11, 2025

import sys
import argparse
import random
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split, Dataset

# Import custom modules
from utils.ModelCreator import EyeDetectionModels
from utils.DatasetHandler import FilteredImageDataset
from utils.Evaluator import ClassificationEvaluator
from utils.Comparator import compare_models
from utils.Trainer import model_train


# Set random seeds for reproducibility
def set_seed(seed=42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transform() -> Compose:
    """
    Get standard data transform for both training and validation/testing.

    Returns:
        transform: Standard transform for all datasets
    """
    # Standard transform as specified
    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return transform


def load_data(
    args,
) -> tuple[DataLoader, DataLoader, DataLoader, FilteredImageDataset]:
    """
    Load and prepare datasets from separate directories for training and evaluation.

    Args:
        args: Command line arguments

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        dataset_ref: Reference to the evaluation dataset for class information
    """
    print(f"Loading training dataset from: {args.train_dir}")
    print(f"Loading evaluation dataset from: {args.eval_dir}")

    # Get standard transform
    transform = get_transform()

    # Load training dataset
    train_dataset = ImageFolder(args.train_dir, transform=transform)
    print(f"Training dataset classes: {train_dataset.classes}")
    print(f"Training dataset size: {len(train_dataset)}")

    # Load evaluation dataset
    eval_dataset = ImageFolder(args.eval_dir, transform=transform)
    print(f"Evaluation dataset classes: {eval_dataset.classes}")

    # Apply class filtering if requested
    excluded_classes = args.exclude_classes.split(",") if args.exclude_classes else None
    if excluded_classes and any(excluded_classes):
        train_dataset = FilteredImageDataset(train_dataset, excluded_classes)
        eval_dataset = FilteredImageDataset(eval_dataset, excluded_classes)
        print(f"After filtering - Classes: {eval_dataset.classes}")
    else:
        train_dataset = FilteredImageDataset(train_dataset)
        eval_dataset = FilteredImageDataset(eval_dataset)
        print("No classes excluded.")

    print(f"After filtering - Train size: {len(train_dataset)}")
    print(f"After filtering - Eval size: {len(eval_dataset)}")

    # Split evaluation dataset into validation and test sets
    val_size = int(
        len(eval_dataset) * (args.val_split / (args.val_split + args.test_split))
    )
    test_size = len(eval_dataset) - val_size

    val_dataset, test_dataset = random_split(eval_dataset, [val_size, test_size])

    print(
        f"Split sizes - Train: {len(train_dataset)}, "
        f"Validation: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Use eval_dataset as the reference for class information
    return train_loader, val_loader, test_loader, eval_dataset


def train_single_model(
    args,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset: FilteredImageDataset,
) -> None:
    """Train a single model specified by the arguments."""

    print(f"Creating {args.model} model...")

    # Initialize model creator
    model_creator = EyeDetectionModels(
        num_classes=len(dataset.classes), freeze_layers=(not args.unfreeze_all)
    )

    # Get model
    if args.model in model_creator.models:
        model = model_creator.models[args.model]()
    else:
        available_models = list(model_creator.models.keys())
        print(
            f"Error: Model '{args.model}' not found. Available models: {available_models}"
        )
        sys.exit(1)

    # Train and evaluate model
    results = model_train(model, train_loader, val_loader, dataset, epochs=args.epochs)

    # Test the model
    if results["accuracy"] is not None:
        print("\nEvaluating on test set...")
        evaluator = ClassificationEvaluator(class_names=dataset.classes)
        test_results = evaluator.evaluate_model(model, test_loader)
        print(f"Test accuracy: {test_results['accuracy']:.4f}")

        # Save model if requested
        if args.save_model:
            save_path = args.save_model
            try:
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
    else:
        print("Training failed. Cannot evaluate on test set.")


def compare_multiple_models(
    args,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset: FilteredImageDataset,
) -> None:
    """Compare multiple models."""

    print("Preparing to compare multiple models...")

    # Initialize model creator
    model_creator = EyeDetectionModels(
        num_classes=len(dataset.classes), freeze_layers=(not args.unfreeze_all)
    )

    # Get list of models to compare
    model_names = args.compare_models.split(",")
    models = []
    names = []

    for model_name in model_names:
        model_name = model_name.strip()
        if model_name in model_creator.models:
            print(f"Adding {model_name} to comparison...")
            models.append(model_creator.models[model_name]())
            names.append(model_name)
        else:
            print(f"Warning: Model '{model_name}' not found, skipping.")

    if not models:
        print("No valid models to compare. Exiting.")
        return

    # Run comparison
    compare_models(
        models,
        train_loader,
        val_loader,
        test_loader,
        dataset,
        epochs=args.epochs,
        names=names,
    )


def main() -> None:
    """Main function to run the eye disease detection application."""

    # Set up argument parser with example usage
    parser = argparse.ArgumentParser(
        description="Eye Disease Detection using Deep Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Train a single model
  python main.py --train-dir "/path/to/augmented_dataset" --eval-dir "/path/to/original_dataset" --model mobilenetv4 --epochs 20 --save-model best_model.pth
  
  # Compare multiple models
  python main.py --train-dir "/path/to/augmented_dataset" --eval-dir "/path/to/original_dataset" --compare-models mobilenetv4,levit,efficientvit --epochs 15
""",
    )

    # Dataset and data loading arguments
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to the training dataset directory (Augmented Dataset)",
    )
    data_group.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Path to the evaluation dataset directory (Original Dataset)",
    )
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.5,
        help="Validation split ratio within evaluation set",
    )
    data_group.add_argument(
        "--test-split",
        type=float,
        default=0.5,
        help="Test split ratio within evaluation set",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    data_group.add_argument(
        "--exclude-classes",
        type=str,
        default=None,
        help="Comma-separated list of class names to exclude",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model",
        type=str,
        default="mobilenetv4",
        help="Model architecture to use. Options: mobilenetv4, levit, efficientvit, gernet, regnetx",
    )
    model_group.add_argument(
        "--unfreeze-all", action="store_true", help="Unfreeze all layers for training"
    )
    model_group.add_argument(
        "--compare-models",
        type=str,
        default=None,
        help="Comma-separated list of models to compare",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    train_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    train_group.add_argument(
        "--save-model", type=str, default=None, help="Path to save the trained model"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Display GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Using {device_count} GPU{'s' if device_count > 1 else ''}")
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available, using CPU")

    # Load data
    train_loader, val_loader, test_loader, dataset = load_data(args)

    # Check if comparing multiple models
    if args.compare_models:
        compare_multiple_models(args, train_loader, val_loader, test_loader, dataset)
    else:
        train_single_model(args, train_loader, val_loader, test_loader, dataset)


if __name__ == "__main__":
    # Example usage for direct execution:
    # python main.py --train-dir "/kaggle/input/eye-disease-image-dataset/Augmented Dataset/Augmented Dataset" \
    #                --eval-dir "/kaggle/input/eye-disease-image-dataset/Original Dataset/Original Dataset" \
    #                --model mobilenetv4 --epochs 10
    main()
