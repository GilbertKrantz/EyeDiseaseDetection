import os
import time
import copy
import argparse
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from itertools import cycle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GrayscaleToRGB(nn.Module):
    """
    Learnable transformation from grayscale to RGB.
    This will learn how to map preprocessed eye images (grayscale) to an RGB-like space
    that better matches what pretrained models expect.

    Added safeguards to prevent CUDA errors from invalid operations.
    """

    def __init__(self, learn_parameters=True):
        super(GrayscaleToRGB, self).__init__()

        # Initialize 3 different transformation matrices (1 for each output channel)
        if learn_parameters:
            # Learnable parameters with different initializations for different output channels
            self.channel1_transform = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
            self.channel2_transform = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))
            self.channel3_transform = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))

            self.bias1 = nn.Parameter(torch.tensor(0.0))
            self.bias2 = nn.Parameter(torch.tensor(0.0))
            self.bias3 = nn.Parameter(torch.tensor(0.0))
        else:
            # Simple static transformation - identity mapping with different weights
            self.register_buffer("channel1_transform", torch.tensor([1.2, 0.0, 0.0]))
            self.register_buffer("channel2_transform", torch.tensor([0.0, 1.0, 0.0]))
            self.register_buffer("channel3_transform", torch.tensor([0.0, 0.0, 0.8]))

            self.register_buffer("bias1", torch.tensor(0.0))
            self.register_buffer("bias2", torch.tensor(0.0))
            self.register_buffer("bias3", torch.tensor(0.0))

        # Activation function to add non-linearity and keep values in reasonable range
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # For already RGB images, just pass through
        if x.size(1) == 3:
            return x

        # For grayscale images, apply our transformation
        if x.size(1) == 1:
            # Make sure we don't have any NaN or Inf values
            if torch.isnan(x).any() or torch.isinf(x).any():
                # Replace NaN/Inf with zeros to prevent CUDA errors
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

            # Ensure x is positive for pow(x, 0.5) operation
            # Clamp x to a small positive value to avoid issues with pow(x, 0.5)
            x_positive = torch.clamp(x, min=1e-6)

            # Create input tensor for transformation with safer operations
            x_enhanced = torch.cat(
                [
                    x,  # Original grayscale
                    torch.pow(x_positive, 0.5),  # Enhance darker regions (safely)
                    torch.pow(x, 2),  # Enhance brighter regions
                ],
                dim=1,
            )  # Now x_enhanced has 3 channels

            # Calculate each output channel with learnable parameters
            r = self.activation(
                torch.sum(
                    self.channel1_transform.view(1, 3, 1, 1) * x_enhanced,
                    dim=1,
                    keepdim=True,
                )
                + self.bias1
            )
            g = self.activation(
                torch.sum(
                    self.channel2_transform.view(1, 3, 1, 1) * x_enhanced,
                    dim=1,
                    keepdim=True,
                )
                + self.bias2
            )
            b = self.activation(
                torch.sum(
                    self.channel3_transform.view(1, 3, 1, 1) * x_enhanced,
                    dim=1,
                    keepdim=True,
                )
                + self.bias3
            )

            # Additional safety check for the output
            result = torch.cat([r, g, b], dim=1)
            if torch.isnan(result).any() or torch.isinf(result).any():
                # If we still got NaN/Inf, create a safe fallback
                # Just duplicate the original grayscale image across 3 channels
                print("Warning: NaN detected in RGB conversion, using fallback")
                return x.repeat(1, 3, 1, 1)

            return result

        raise ValueError(f"Expected input to have 1 or 3 channels, got {x.size(1)}")


class EyeDiseaseClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        dropout_rate=0.2,
        pretrained=True,
        learnable_transform=True,
        unfreeze_last_n_layers=0,  # New parameter to control unfreezing
    ):
        super(EyeDiseaseClassifier, self).__init__()
        # Grayscale to RGB transformation
        self.transform = GrayscaleToRGB(learn_parameters=learnable_transform)

        # Create model with pretrained weights
        self.model = timm.create_model(model_name, pretrained=pretrained)
        print(f"Loaded model: {model_name}")

        # For MobileViT models, we need to handle spatial dimensions
        self.is_mobilevit = "mobilevit" in model_name.lower()

        # Freeze all backbone parameters by default if using pretrained model
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the last n layers if specified
            if unfreeze_last_n_layers > 0:
                self._unfreeze_last_n_layers(unfreeze_last_n_layers)

        # Determine feature size using a dummy forward pass
        logging.info("Determining feature size from dummy input")
        try:
            dummy_input = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
        except Exception as e:
            logging.error(f"Error during dummy forward pass: {e}")
            # Continue with a fallback
            dummy_output = torch.zeros(1, 1000)  # Fallback to a flat output
            print("Using fallback output shape: ", dummy_output.shape)

        # Print the shape to understand the model output structure
        print(f"Model output shape with dummy input: {dummy_output.shape}")

        # For models that return spatial features (4D tensors)
        if len(dummy_output.shape) == 4:
            # We need to add global pooling
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            # Calculate number of features after pooling
            pooled_output = self.global_pool(dummy_output)
            num_features = pooled_output.reshape(1, -1).shape[1]
            print(f"After pooling, features shape: {pooled_output.shape}")
            print(f"Flattened feature size: {num_features}")
        else:
            # For models that already return a flat feature vector
            num_features = dummy_output.shape[1]
            self.global_pool = nn.Identity()  # No pooling needed

        print(f"Final num_features determined: {num_features}")

        # Custom classifier head with dropout - use the correct input feature size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def _unfreeze_last_n_layers(self, n):
        """Unfreeze the last n layers of the backbone model."""
        # Get a list of all parameter groups in the model
        all_layers = list(self.model.named_parameters())
        num_layers = len(all_layers)

        # Check if we're trying to unfreeze more layers than exist
        if n > num_layers:
            print(
                f"Warning: Requested to unfreeze {n} layers but model only has {num_layers}. Unfreezing all."
            )
            n = num_layers

        # Unfreeze the last n layers
        print(f"Unfreezing last {n} layers of the backbone model")
        for name, param in all_layers[-n:]:
            param.requires_grad = True
            print(f"Unfrozen: {name}")

    def unfreeze_layers(self, n=None, layer_names=None):
        """
        Method to unfreeze specific layers after model initialization.

        Args:
            n: Integer, number of lowest layers to unfreeze (counting from the end)
            layer_names: List of strings, specific layer names to unfreeze
        """
        if n is not None:
            self._unfreeze_last_n_layers(n)

        if layer_names is not None:
            # Unfreeze specific layers by name
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
                    print(f"Unfrozen by name: {name}")

    def get_trainable_params(self):
        """
        Return two parameter groups: backbone and classifier parameters.
        Useful for applying different learning rates.
        """
        backbone_params = [p for p in self.model.parameters() if p.requires_grad]
        classifier_params = self.classifier.parameters()

        return {"backbone": backbone_params, "classifier": classifier_params}

    def forward(self, x):
        # Transform grayscale to RGB-like features if needed
        x = self.transform(x)

        # Extract features with the backbone model
        features = self.model(x)

        # Debug shape information
        if not hasattr(self, "_shape_printed"):
            logging.info(f"Feature shape from backbone: {features.shape}")
            self._shape_printed = True

        # For 4D tensors (spatial outputs), apply global pooling and flatten
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = torch.flatten(features, 1)  # Flatten to [batch_size, features]
            # Debug pooled shape
            if not hasattr(self, "_pooled_shape_printed"):
                print(f"After pooling and flattening: {features.shape}")
                self._pooled_shape_printed = True

        # Final classification
        return self.classifier(features)


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    use_amp=False,
    early_stopping_patience=5,
    save_dir="./model_outputs",
):

    os.makedirs(save_dir, exist_ok=True)

    # Initialize logging
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_aucs = []
    val_aucs = []

    # For early stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0

    # Mixed precision training
    scaler = GradScaler() if use_amp else None

    # Track training time
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            all_labels = []
            all_probs = []

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    if use_amp and phase == "train":
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                # Store labels and probabilities for ROC-AUC calculation
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Convert to numpy arrays for ROC-AUC calculation
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # Calculate ROC-AUC for multi-class (one-vs-rest)
            n_classes = all_probs.shape[1]

            # Check if we have more than one class
            if n_classes > 1:
                try:
                    # One-vs-Rest ROC AUC for multiclass
                    epoch_auc = roc_auc_score(
                        np.eye(n_classes)[all_labels],  # Convert to one-hot encoding
                        all_probs,
                        multi_class="ovr",
                        average="macro",
                    )
                except ValueError:
                    # Handle case where not all classes are present in this batch/epoch
                    epoch_auc = 0.0
            else:
                epoch_auc = 0.0  # Default if we can't calculate AUC

            # Log metrics
            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                train_aucs.append(epoch_auc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                val_aucs.append(epoch_auc)

            print(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}"
            )

            # Deep copy the model if best validation accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                    },
                    os.path.join(save_dir, "best_model.pth"),
                )
                no_improve_epochs = 0
            elif phase == "val":
                no_improve_epochs += 1

        # Save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            },
            os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
        )

        print()

        # Early stopping
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_auc": train_aucs,
        "val_auc": val_aucs,
    }

    return model, history


def plot_roc_curves(y_true, y_score, class_names, save_path=None):
    """
    Plot ROC curves for multiclass classification.

    Args:
        y_true (array): True labels (one-hot encoded for multiclass)
        y_score (array): Predicted probabilities
        class_names (list or dict): List of class names or dictionary mapping indices to class names
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    # Handle both list and dictionary class_names
    if isinstance(class_names, dict):
        # If class_names is a dictionary like {0: 'Class_0', 1: 'Class_1', ...}
        label_list = [class_names[i] for i in sorted(class_names.keys())]
        n_classes = len(class_names)
    else:
        # If class_names is a list like ['Class_0', 'Class_1', ...]
        label_list = class_names
        n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))

    colors = cycle(
        [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
    )

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{label_list[i]} (AUC = {roc_auc[i]:.2f})",
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_model(model, test_loader, device, class_names, save_dir="./model_outputs"):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Calculate overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Calculate and print ROC-AUC for multiclass
    # Handle both list and dictionary class_names
    if isinstance(class_names, dict):
        # If class_names is a dictionary like {0: 'Class_0', 1: 'Class_1', ...}
        label_list = [class_names[i] for i in sorted(class_names.keys())]
        n_classes = len(class_names)
        # Make sure we map the numeric labels correctly
        class_indices = sorted(class_names.keys())
    else:
        # If class_names is a list like ['Class_0', 'Class_1', ...]
        label_list = class_names
        n_classes = len(class_names)
        class_indices = list(range(n_classes))

    # Convert labels to one-hot encoding for ROC-AUC calculation
    y_true_onehot = np.eye(n_classes)[all_labels]

    # Calculate ROC-AUC (One-vs-Rest)
    try:
        roc_auc_ovr = roc_auc_score(
            y_true_onehot, all_probs, multi_class="ovr", average="macro"
        )
        print(f"\nROC-AUC (macro average, one-vs-rest): {roc_auc_ovr:.4f}")

        # Print per-class AUC scores
        class_auc_scores = []
        for i, idx in enumerate(class_indices):
            auc_i = roc_auc_score(y_true_onehot[:, i], all_probs[:, i])
            class_auc_scores.append(auc_i)
            print(f"  - {label_list[i]} AUC: {auc_i:.4f}")

        # Plot ROC curves
        os.makedirs(save_dir, exist_ok=True)
        plot_roc_curves(
            y_true_onehot,
            all_probs,
            label_list,  # Use the processed label_list instead of raw class_names
            save_path=os.path.join(save_dir, "roc_curves.png"),
        )
        print(f"\nROC curve plot saved to {os.path.join(save_dir, 'roc_curves.png')}")

    except ValueError as e:
        print(f"Could not calculate ROC-AUC: {e}")
        roc_auc_ovr = 0.0
        class_auc_scores = [0.0] * n_classes

    return {
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "confusion_matrix": cm,
        "roc_auc_ovr": roc_auc_ovr,
        "class_auc_scores": class_auc_scores,
    }


def visualize_channel_transformation(model, sample_image, save_path=None):
    """
    Visualize how the channel transformation module transforms a grayscale image.

    Args:
        model: The trained model with transform module
        sample_image: A sample grayscale image tensor of shape [1, 1, H, W]
        save_path: Path to save the visualization
    """
    model.eval()
    with torch.no_grad():
        # Get the transformed image
        if not hasattr(model, "transform"):
            print("Model does not have a transform module")
            return

        transformed = model.transform(sample_image)

        # Convert tensors to numpy arrays for plotting
        orig_img = sample_image.cpu().squeeze().numpy()
        r_channel = transformed[:, 0, :, :].cpu().squeeze().numpy()
        g_channel = transformed[:, 1, :, :].cpu().squeeze().numpy()
        b_channel = transformed[:, 2, :, :].cpu().squeeze().numpy()

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the original grayscale image
        axs[0, 0].imshow(orig_img, cmap="gray")
        axs[0, 0].set_title("Original Grayscale")
        axs[0, 0].axis("off")

        # Plot each transformed channel
        axs[0, 1].imshow(r_channel, cmap="Reds")
        axs[0, 1].set_title("R Channel")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(g_channel, cmap="Greens")
        axs[1, 0].set_title("G Channel")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(b_channel, cmap="Blues")
        axs[1, 1].set_title("B Channel")
        axs[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a TIMM model for eye disease detection"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet_b0",
        help="Model name from TIMM library",
    )
    parser.add_argument(
        "--num_classes", type=int, default=5, help="Number of eye disease classes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_outputs",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--early_stopping", type=int, default=5, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for the classifier head",
    )
    parser.add_argument(
        "--learnable_transform",
        action="store_true",
        default=True,
        help="Use learnable transformation for grayscale to RGB",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if model exists in TIMM
    available_models = timm.list_models(pretrained=True)
    if args.model_name not in available_models:
        print(f"Model {args.model_name} not found in TIMM library.")
        print("Available models include:")
        for i, model in enumerate(available_models[:10]):
            print(f"- {model}")
        print(f"... and {len(available_models) - 10} more models.")
        print("Use 'timm.list_models(pretrained=True)' to see all available models")
        return

    print(f"Creating model: {args.model_name} with learnable grayscale transformation")
    model = EyeDiseaseClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        learnable_transform=args.learnable_transform,
    )
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Define learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Define class names (replace with your actual class names)
    class_names = [f"Class_{i}" for i in range(args.num_classes)]

    # Rest of the code remains the same...

    # Placeholder for dataloaders - replace this with your actual code
    print(
        "\nWARNING: You need to uncomment and modify the following code to use your dataloaders"
    )

    """
    # Train the model
    model, history = train_model(
        model=model,
        dataloaders={'train': train_loader, 'val': val_loader},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        use_amp=args.use_amp,
        early_stopping_patience=args.early_stopping,
        save_dir=args.output_dir
    )
    
    # Optional: Visualize the learned transformation on a sample image
    if args.learnable_transform and 'test_loader' in locals():
        # Get a sample image from the test set
        for sample_batch, _ in test_loader:
            sample_image = sample_batch[:1].to(device)  # Take just one image
            visualize_channel_transformation(
                model=model,
                sample_image=sample_image,
                save_path=os.path.join(args.output_dir, 'channel_transform_vis.png')
            )
            break
    
    # Evaluate the model
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=args.output_dir
    )
    """

    print("\nTo use this script with your own data:")
    print("1. Make sure your dataloaders are properly defined")
    print("2. Uncomment the training and evaluation code")
    print("3. Replace the placeholder class_names with your eye disease classes")
    print("4. Run the script with your desired arguments")
    print("\nExample usage:")
    print(
        "python train_eye_disease.py --model_name efficientnet_b0 --num_classes 5 --num_epochs 20 --learning_rate 0.001 --use_amp --learnable_transform"
    )


if __name__ == "__main__":
    main()
