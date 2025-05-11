import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import gc

from Evaluator import ClassificationEvaluator
from Callback import EarlyStopping


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    early_stopping,
    epochs=15,
    use_ddp=False,
):
    """
    Train the model and perform validation using multiple GPUs.
    Supports both DataParallel (DP) and DistributedDataParallel (DDP) modes.

    Args:
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        early_stopping: Early stopping handler
        epochs: Maximum number of epochs to train
        use_ddp: Whether to use DistributedDataParallel (True) or DataParallel (False)
    """
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(
            f"Warning: Requested multi-GPU training but only {num_gpus} GPU(s) available. Continuing with available resources."
        )
    else:
        print(f"Using {num_gpus} GPUs for training")

    # Setup device and model
    if num_gpus >= 2:
        if use_ddp:
            # For DistributedDataParallel
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            # Initialize process group
            dist.init_process_group(backend="nccl")
            local_rank = dist.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            model = model.to(device)
            model = DDP(model, device_ids=[local_rank])
        else:
            # For DataParallel (simpler to use)
            device = torch.device("cuda:0")
            model = model.to(device)
            model = torch.nn.DataParallel(model)
    else:
        # Single GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Store validation predictions and labels for final evaluation
    all_val_labels = []
    all_val_preds = []
    all_val_scores = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_scores.append(probs.cpu().numpy())

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        all_scores = np.vstack(all_scores) if all_scores else np.array([])

        # Store validation results for the final epoch
        all_val_labels = all_labels
        all_val_preds = all_preds
        all_val_scores = all_scores

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Check early stopping
        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Free up memory
        del all_labels, all_preds, all_scores
        gc.collect()
        torch.cuda.empty_cache()

    # Clean up DDP if used
    if num_gpus >= 2 and use_ddp:
        dist.destroy_process_group()

    return (
        model,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        all_val_labels,
        all_val_preds,
        all_val_scores,
    )


def model_train(model, train_loader, val_loader, dataset, epochs=20):
    model_name = type(model).__name__
    if hasattr(model, "pretrained_cfg") and "name" in model.pretrained_cfg:
        model_name = model.pretrained_cfg["name"]

    print(f"\n{'='*20} Training {model_name} {'='*20}\n")

    class_names = dataset.classes
    num_classes = len(class_names)
    learning_rate = 0.001

    try:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )
        early_stopping = EarlyStopping(patience=5)

        (
            model,
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            val_labels,
            val_preds,
            val_scores,
        ) = train_model(
            model,
            nn.CrossEntropyLoss(),
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            early_stopping,
            epochs=epochs,
            use_ddp=False,
        )

        print(f"\n{'='*20} Evaluation for {model_name} {'='*20}\n")
        evaluator = ClassificationEvaluator(
            num_classes=num_classes,
            class_names=class_names,
        )

        evaluator.plot_training_history(train_losses, val_losses, train_accs, val_accs)
        # Process validation predictions and labels
        try:
            evaluator.plot_confusion_matrix(val_labels, val_preds)
            evaluator.plot_per_class_accuracy(val_labels, val_preds)

            # Get metrics from the updated function including kappa
            accuracy, report_dict, roc_auc_dict, pr_auc_dict, kappa = (
                evaluator.compute_metrics(
                    val_labels,
                    val_preds,
                    val_scores,
                    model_name,
                )
            )

            # Build a results dictionary including kappa
            results = {
                "accuracy": accuracy,
                "report": report_dict,
                "roc_auc": roc_auc_dict,
                "pr_auc": pr_auc_dict,
                "kappa": kappa,
            }

            return results
        except Exception as viz_error:
            print(f"Error in visualization: {viz_error}")
            import traceback

            traceback.print_exc()
            return {"accuracy": None}

    except Exception as e:
        print(f"Error occurred when training {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return {"accuracy": None}
    finally:
        # Clean up memory
        if "optimizer" in locals():
            del optimizer
        if "scheduler" in locals():
            del scheduler
        if "early_stopping" in locals():
            del early_stopping
        if "train_losses" in locals():
            del train_losses
            del val_losses
            del train_accs
            del val_accs
            del val_labels
            del val_preds
            del val_scores

        gc.collect()
        torch.cuda.empty_cache()
