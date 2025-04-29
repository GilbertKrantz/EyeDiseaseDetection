import os
import time
import copy
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

import timm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

class EyeDiseaseClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.2, pretrained=True, in_channels=1):
        super(EyeDiseaseClassifier, self).__init__()
        
        # Handle grayscale images by modifying the first conv layer to accept fewer channels
        if in_channels != 3:
            # Create model with pretrained weights but modify the first layer
            self.model = timm.create_model(model_name, pretrained=pretrained)
            
            # Get the first convolutional layer
            if hasattr(self.model, 'conv_stem'):
                # For EfficientNet and similar models
                first_conv = self.model.conv_stem
            elif hasattr(self.model, 'conv1'):
                # For ResNet and similar models
                first_conv = self.model.conv1
            elif hasattr(self.model, 'features') and hasattr(self.model.features[0], 'conv'):
                # For DenseNet and similar models
                first_conv = self.model.features[0].conv
            elif hasattr(self.model, 'features') and isinstance(self.model.features[0], nn.Conv2d):
                # For VGG and similar models
                first_conv = self.model.features[0]
            else:
                raise ValueError(f"Cannot adapt first layer for model {model_name}. Please use a 3-channel input or choose another model.")
            
            # Create a new first conv layer with the same parameters but in_channels=1
            new_first_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Initialize the weights for the new layer - average the weights across the channel dimension
            if pretrained:
                with torch.no_grad():
                    # For each output channel, average the weights across input channels
                    new_weight = first_conv.weight.data.mean(dim=1, keepdim=True)
                    new_first_conv.weight.copy_(new_weight)
                    if first_conv.bias is not None:
                        new_first_conv.bias.copy_(first_conv.bias)
            
            # Replace the first conv layer
            if hasattr(self.model, 'conv_stem'):
                self.model.conv_stem = new_first_conv
            elif hasattr(self.model, 'conv1'):
                self.model.conv1 = new_first_conv
            elif hasattr(self.model, 'features') and hasattr(self.model.features[0], 'conv'):
                self.model.features[0].conv = new_first_conv
            elif hasattr(self.model, 'features') and isinstance(self.model.features[0], nn.Conv2d):
                self.model.features[0] = new_first_conv
        else:
            # Regular case - use the standard model
            self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of features from the model
        if hasattr(self.model, 'fc'):
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # Remove the classification layer
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Linear):
                num_features = self.model.classifier.in_features
                self.model.classifier = nn.Identity()
            else:
                num_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Identity()
        elif hasattr(self.model, 'head'):
            num_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise ValueError(f"Model architecture for {model_name} not supported yet!")
        
        # Custom classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)


def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                device, num_epochs, use_amp=False, early_stopping_patience=5,
                save_dir='./model_outputs'):
    
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
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
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
                with torch.set_grad_enabled(phase == 'train'):
                    if use_amp and phase == 'train':
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
                        
                        if phase == 'train':
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
            
            if phase == 'train' and scheduler is not None:
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
                        multi_class='ovr',
                        average='macro'
                    )
                except ValueError:
                    # Handle case where not all classes are present in this batch/epoch
                    epoch_auc = 0.0
            else:
                epoch_auc = 0.0  # Default if we can't calculate AUC
            
            # Log metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                train_aucs.append(epoch_auc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                val_aucs.append(epoch_auc)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
            
            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
                no_improve_epochs = 0
            elif phase == 'val':
                no_improve_epochs += 1
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print()
        
        # Early stopping
        if no_improve_epochs >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'train_auc': train_aucs,
        'val_auc': val_aucs
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
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f'{label_list[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_model(model, test_loader, device, class_names, save_dir='./model_outputs'):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
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
        roc_auc_ovr = roc_auc_score(y_true_onehot, all_probs, multi_class='ovr', average='macro')
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
            save_path=os.path.join(save_dir, 'roc_curves.png')
        )
        print(f"\nROC curve plot saved to {os.path.join(save_dir, 'roc_curves.png')}")
    
    except ValueError as e:
        print(f"Could not calculate ROC-AUC: {e}")
        roc_auc_ovr = 0.0
        class_auc_scores = [0.0] * n_classes
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'roc_auc_ovr': roc_auc_ovr,
        'class_auc_scores': class_auc_scores
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train a TIMM model for eye disease detection')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', 
                        help='Model name from TIMM library')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of eye disease classes')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--output_dir', type=str, default='./model_outputs',
                        help='Directory to save model checkpoints')
    parser.add_argument('--use_amp', action='store_true', 
                        help='Use mixed precision training')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for the classifier head')
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
    
    print(f"Creating model: {args.model_name} with {args.in_channels} input channels")
    model = EyeDiseaseClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        in_channels=args.in_channels
    )
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Define class names (replace with your actual class names)
    class_names = [f"Class_{i}" for i in range(args.num_classes)]
    
    # Rest of the code remains the same...
    
    # Placeholder for dataloaders - replace this with your actual code
    print("\nWARNING: You need to uncomment and modify the following code to use your dataloaders")
    
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
    print("3. Replace the placeholder class_names with your disease classes")
    print("4. Run the script with your desired arguments")
    print("\nExample usage:")
    print("python train_eye_disease.py --model_name efficientnet_b0 --num_classes 5 --num_epochs 20 --learning_rate 0.001 --use_amp")


if __name__ == "__main__":
    main()