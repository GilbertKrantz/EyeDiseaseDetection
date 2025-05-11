import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize


class ClassificationEvaluator:
    """
    A class to evaluate and visualize classification model performance.

    This class provides methods to compute various classification metrics
    and generate visualizations for model evaluation.
    """

    def __init__(self, class_names):
        """
        Initialize the evaluator with class names.

        Parameters:
        - class_names: list of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

    def _ensure_numpy(self, data):
        """Convert tensor to numpy if needed."""
        if torch.is_tensor(data):
            return data.cpu().numpy()
        return np.array(data)

    def evaluate_model(self, model, test_loader):
        """
        Evaluate a trained model on test dataset.

        Parameters:
        - model: PyTorch model to evaluate
        - test_loader: DataLoader containing test data

        Returns:
        - results: Dictionary containing evaluation metrics
        """
        model.eval()
        device = next(model.parameters()).device

        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.append(
                    torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                )

        all_scores = np.vstack(all_scores)

        # Compute metrics
        results = self.compute_metrics(all_labels, all_preds, all_scores)
        return results

    def compute_metrics(self, y_true, y_pred, y_scores, model_name=""):
        """
        Compute comprehensive classification metrics.

        Parameters:
        - y_true: true labels
        - y_pred: predicted labels
        - y_scores: predicted probability scores
        - model_name: name of the model (optional)

        Returns:
        - Dictionary containing all metrics
        """
        # Ensure numpy arrays
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)
        y_scores = self._ensure_numpy(y_scores)

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {accuracy:.4f}")

        # Calculate and display Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Cohen's Kappa Score: {kappa:.4f}")

        # Generate classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        # Print formatted classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # Calculate ROC curves and AUC for each class
        print("\nCalculating ROC curves...")
        roc_auc_dict = self.plot_roc_curves(y_true, y_scores)

        # Calculate PR curves and AUC for each class
        print("\nCalculating PR curves...")
        pr_auc_dict = self.plot_pr_curves(y_true, y_scores)

        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        self.plot_confusion_matrix(y_true, y_pred)

        # Plot per-class accuracy
        print("\nCalculating per-class accuracy...")
        self.plot_per_class_accuracy(y_true, y_pred)

        # Return metrics dictionary
        return {
            "accuracy": accuracy,
            "report": report,
            "roc_auc": roc_auc_dict,
            "pr_auc": pr_auc_dict,
            "kappa": kappa,
        }

    def plot_roc_curves(self, y_true, y_scores):
        """
        Plot ROC curves for multi-class classification.

        Parameters:
        - y_true: true labels
        - y_scores: predicted probability scores

        Returns:
        - Dictionary containing AUC values for each class
        """
        y_true = self._ensure_numpy(y_true)
        y_scores = self._ensure_numpy(y_scores)

        # Binarize the labels for one-vs-rest ROC calculation
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        plt.figure(figsize=(12, 8))

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label=f"{self.class_names[i]} (area = {roc_auc[i]:.2f})",
            )

        # Plot the diagonal (random classifier)
        plt.plot([0, 1], [0, 1], "k--", lw=2)

        # Calculate and plot micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'Micro-average (area = {roc_auc["micro"]:.2f})',
            lw=2,
            linestyle=":",
            color="deeppink",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return roc_auc

    def plot_pr_curves(self, y_true, y_scores):
        """
        Plot Precision-Recall curves for multi-class classification.

        Parameters:
        - y_true: true labels
        - y_scores: predicted probability scores

        Returns:
        - Dictionary containing average precision values for each class
        """
        y_true = self._ensure_numpy(y_true)
        y_scores = self._ensure_numpy(y_scores)

        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        # Compute PR curve and average precision for each class
        precision = {}
        recall = {}
        avg_precision = {}

        plt.figure(figsize=(12, 8))

        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], y_scores[:, i]
            )
            avg_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])

            plt.plot(
                recall[i],
                precision[i],
                lw=2,
                label=f"{self.class_names[i]} (AP = {avg_precision[i]:.2f})",
            )

        # Calculate and plot micro-average PR curve
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_scores.ravel()
        )
        avg_precision["micro"] = average_precision_score(
            y_true_bin.ravel(), y_scores.ravel()
        )

        plt.plot(
            recall["micro"],
            precision["micro"],
            label=f'Micro-average (AP = {avg_precision["micro"]:.2f})',
            lw=2,
            linestyle=":",
            color="deeppink",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return avg_precision

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix.

        Parameters:
        - y_true: true labels
        - y_pred: predicted labels
        """
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)

        # Get unique values in both arrays
        unique_values = np.unique(np.concatenate([y_true, y_pred]))
        print(f"Unique values in confusion matrix data: {unique_values}")

        # Create the confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    def plot_per_class_accuracy(self, y_true, y_pred):
        """
        Plot per-class accuracy.

        Parameters:
        - y_true: true labels
        - y_pred: predicted labels
        """
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)

        # Create the confusion matrix with explicit labels
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        # Calculate per-class accuracy
        per_class_accuracy = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if i < cm.shape[0] and np.sum(cm[i, :]) > 0:
                per_class_accuracy[i] = cm[i, i] / np.sum(cm[i, :])

        # Create the bar plot
        plt.figure(figsize=(14, 7))
        plt.bar(range(self.num_classes), per_class_accuracy, color="skyblue")
        plt.xticks(range(self.num_classes), self.class_names, rotation=45, ha="right")
        plt.xlabel("Classes")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy")
        plt.tight_layout()
        plt.show()

        return per_class_accuracy

    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """
        Plot accuracy and loss curves from training history.

        Parameters:
        - train_losses: list of training losses
        - val_losses: list of validation losses
        - train_accs: list of training accuracies
        - val_accs: list of validation accuracies
        """
        plt.figure(figsize=(12, 5))

        # Accuracy curve
        plt.subplot(1, 2, 1)
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)

        # Loss curve
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
