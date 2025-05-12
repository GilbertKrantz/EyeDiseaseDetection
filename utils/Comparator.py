from torch.utils.data import DataLoader, Dataset
from Trainer import model_train
from Evaluator import ClassificationEvaluator


def compare_models(
    models: list,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset: Dataset,
    epochs: int = 20,
    names: list = None,
) -> None:
    """
    Compare multiple models on validation and test datasets.
    Args:
        models (list): List of models to compare.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        dataset (Dataset): Dataset object containing class names.
        epochs (int): Number of epochs for training.
        names (list): List of model names. If None, default names will be used.
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(models))]

    val_results = {}
    test_results = {}
    best_model_obj = None
    best_accuracy = -1
    best_model_name = ""

    # Summary dictionaries for metrics
    val_roc_auc_summary = {}
    test_roc_auc_summary = {}
    val_pr_auc_summary = {}
    test_pr_auc_summary = {}
    val_kappa_summary = {}
    test_kappa_summary = {}

    for i, (model, name) in enumerate(zip(models, names)):
        evaluator = ClassificationEvaluator(
            num_classes=len(dataset.classes),
            class_names=dataset.classes,
        )

        print(f"\n\n{'#'*30} Training {name} ({i+1}/{len(models)}) {'#'*30}\n")
        model_results = model_train(model, train_loader, val_loader, dataset, epochs)

        # Extract accuracy from results
        accuracy = model_results.get("accuracy")
        val_results[name] = accuracy

        # Extract and store metrics
        if "roc_auc" in model_results and "micro" in model_results["roc_auc"]:
            val_roc_auc_summary[name] = model_results["roc_auc"]["micro"]
        else:
            val_roc_auc_summary[name] = None

        if "pr_auc" in model_results and "micro" in model_results["pr_auc"]:
            val_pr_auc_summary[name] = model_results["pr_auc"]["micro"]
        else:
            val_pr_auc_summary[name] = None

        # Store kappa score
        if "kappa" in model_results:
            val_kappa_summary[name] = model_results["kappa"]
        else:
            val_kappa_summary[name] = None

        # Evaluate on test set
        if accuracy is not None:
            print(f"\n{'='*20} Testing {name} on Test Set {'='*20}\n")
            test_model_results = evaluator.evaluate_model(model, test_loader)

            # Extract accuracy from test results
            test_accuracy = test_model_results.get("accuracy")
            test_results[name] = test_accuracy

            # Extract and store test metrics
            if (
                "roc_auc" in test_model_results
                and "micro" in test_model_results["roc_auc"]
            ):
                test_roc_auc_summary[name] = test_model_results["roc_auc"]["micro"]
            else:
                test_roc_auc_summary[name] = None

            if (
                "pr_auc" in test_model_results
                and "micro" in test_model_results["pr_auc"]
            ):
                test_pr_auc_summary[name] = test_model_results["pr_auc"]["micro"]
            else:
                test_pr_auc_summary[name] = None

            # Store test kappa score
            if "kappa" in test_model_results:
                test_kappa_summary[name] = test_model_results["kappa"]
            else:
                test_kappa_summary[name] = None

            # Track best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_obj = copy.deepcopy(model)
                best_model_name = name

    # Print comprehensive comparison
    print("\n\n" + "=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 100)
    print(
        f"{'Model':<20}{'Val Acc':<10}{'Test Acc':<10}{'Val ROC AUC':<14}{'Test ROC AUC':<14}{'Val PR AUC':<14}{'Test PR AUC':<14}{'Val Kappa':<14}{'Test Kappa':<14}"
    )
    print("-" * 100)

    for name in val_results.keys():
        val_acc = val_results[name]
        test_acc = test_results.get(name, None)
        val_roc = val_roc_auc_summary.get(name, None)
        test_roc = test_roc_auc_summary.get(name, None)
        val_pr = val_pr_auc_summary.get(name, None)
        test_pr = test_pr_auc_summary.get(name, None)
        val_kappa = val_kappa_summary.get(name, None)
        test_kappa = test_kappa_summary.get(name, None)

        # Format values for display
        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "Failed"
        test_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        val_roc_str = f"{val_roc:.4f}" if val_roc is not None else "N/A"
        test_roc_str = f"{test_roc:.4f}" if test_roc is not None else "N/A"
        val_pr_str = f"{val_pr:.4f}" if val_pr is not None else "N/A"
        test_pr_str = f"{test_pr:.4f}" if test_pr is not None else "N/A"
        val_kappa_str = f"{val_kappa:.4f}" if val_kappa is not None else "N/A"
        test_kappa_str = f"{test_kappa:.4f}" if test_kappa is not None else "N/A"

        print(
            f"{name:<20}{val_acc_str:<10}{test_acc_str:<10}{val_roc_str:<14}{test_roc_str:<14}{val_pr_str:<14}{test_pr_str:<14}{val_kappa_str:<14}{test_kappa_str:<14}"
        )

    # Identify best model based on test metrics
    if test_results:
        # Best model by accuracy
        best_acc_model = max(
            test_results.items(), key=lambda x: x[1] if x[1] is not None else -1
        )
        print(
            f"\nBest model by accuracy: {best_acc_model[0]} (Test Accuracy: {best_acc_model[1]:.4f})"
        )

        # Best model by ROC AUC (if available)
        if any(v is not None for v in test_roc_auc_summary.values()):
            best_roc_model = max(
                [(k, v) for k, v in test_roc_auc_summary.items() if v is not None],
                key=lambda x: x[1] if x[1] is not None else -1,
            )
            print(
                f"Best model by ROC AUC: {best_roc_model[0]} (Test ROC AUC: {best_roc_model[1]:.4f})"
            )

        # Best model by PR AUC (if available)
        if any(v is not None for v in test_pr_auc_summary.values()):
            best_pr_model = max(
                [(k, v) for k, v in test_pr_auc_summary.items() if v is not None],
                key=lambda x: x[1] if x[1] is not None else -1,
            )
            print(
                f"Best model by PR AUC: {best_pr_model[0]} (Test PR AUC: {best_pr_model[1]:.4f})"
            )

        # Best model by Kappa (if available)
        if any(v is not None for v in test_kappa_summary.values()):
            best_kappa_model = max(
                [(k, v) for k, v in test_kappa_summary.items() if v is not None],
                key=lambda x: x[1] if x[1] is not None else -1,
            )
            print(
                f"Best model by Cohen's Kappa: {best_kappa_model[0]} (Test Kappa: {best_kappa_model[1]:.4f})"
            )

        # Save the best model (by accuracy)
        if best_model_obj is not None:
            try:
                model_save_path = (
                    f"best_model_{best_model_name.lower().replace(' ', '_')}.pth"
                )
                torch.save(best_model_obj.state_dict(), model_save_path)
                print(f"Best model saved to {model_save_path}")
            except Exception as save_error:
                print(f"Error saving best model: {save_error}")
    else:
        print("\nNo models successfully completed testing.")

    print("=" * 100)
