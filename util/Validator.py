import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def validate_dataset(dataloader, class_names, num_batches=5, display_samples=True):
    """
    Validate a dataset by checking:
    1. Class distribution
    2. Image properties (shape, value range, normalization)
    3. Label integrity
    4. Batch consistency

    Args:
        dataloader: DataLoader to validate
        class_names: List or dict of class names
        num_batches: Number of batches to examine
        display_samples: Whether to display sample images
    """

    print("\n=== DATASET VALIDATION ===")

    # Statistics to collect
    total_samples = 0
    label_counter = Counter()
    image_shapes = set()
    image_dims = set()
    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []
    nan_count = 0
    infinity_count = 0

    # Sample storage for display
    sample_images = []
    sample_labels = []

    # Process batches
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break

        # Basic batch info
        batch_size = images.shape[0]
        total_samples += batch_size

        # Convert to numpy for easier analysis
        images_np = images.numpy()
        labels_np = labels.numpy()

        # Store samples for display
        if display_samples and i == 0:
            # Store up to 5 samples from first batch
            num_samples = min(5, batch_size)
            sample_images = images_np[:num_samples]
            sample_labels = labels_np[:num_samples]

        # Check for NaN and infinity
        nan_count += np.isnan(images_np).sum()
        infinity_count += np.isinf(images_np).sum()

        # Image shape and dimensions
        image_shapes.add(images_np.shape[1:])
        image_dims.add(images_np.shape[1])  # Number of channels

        # Value range
        min_vals.append(images_np.min())
        max_vals.append(images_np.max())
        mean_vals.append(images_np.mean())
        std_vals.append(images_np.std())

        # Label counts
        label_counter.update(labels_np)

    # Report findings
    print(f"\n1. GENERAL INFORMATION:")
    print(f"   Samples analyzed: {total_samples}")
    print(f"   Image shapes: {image_shapes}")
    print(f"   Channel dimensions: {image_dims}")

    # Check for common issues
    if len(image_shapes) > 1:
        print("   ❌ WARNING: Inconsistent image shapes detected")

    expected_channels = 3  # Most models expect 3 channels (RGB)
    if 1 in image_dims and 3 not in image_dims:
        print("   ℹ️ Grayscale images detected (1 channel)")
        print(
            "      Your model has a GrayscaleToRGB transform module which should handle this"
        )
    elif 3 in image_dims:
        print("   ✓ RGB images detected (3 channels)")
    else:
        print(f"   ❌ WARNING: Unusual channel count: {image_dims}")

    print(f"\n2. IMAGE PROPERTIES:")
    print(f"   Min value: {np.min(min_vals):.4f}")
    print(f"   Max value: {np.max(max_vals):.4f}")
    print(f"   Mean value: {np.mean(mean_vals):.4f}")
    print(f"   Std dev: {np.mean(std_vals):.4f}")

    # Check normalization
    if np.min(min_vals) >= 0 and np.max(max_vals) <= 1:
        print("   ✓ Images appear to be normalized to [0,1] range")
    elif np.min(min_vals) >= -1 and np.max(max_vals) <= 1:
        print("   ✓ Images appear to be normalized to [-1,1] range")
    elif np.min(min_vals) >= -3 and np.max(max_vals) <= 3:
        print("   ✓ Images appear to be normalized with ImageNet mean/std")
    elif np.max(max_vals) > 10:
        print("   ❌ WARNING: Images may not be properly normalized")
        print("      Consider normalizing to [0,1] or [-1,1] range")

    if nan_count > 0:
        print(f"   ❌ WARNING: {nan_count} NaN values detected")
    else:
        print("   ✓ No NaN values detected")

    if infinity_count > 0:
        print(f"   ❌ WARNING: {infinity_count} infinity values detected")
    else:
        print("   ✓ No infinity values detected")

    print(f"\n3. LABEL DISTRIBUTION:")
    # Sort labels by index
    sorted_labels = sorted(label_counter.items())
    total_labels = sum(label_counter.values())

    # Process class names
    if isinstance(class_names, dict):
        # If class_names is a dictionary like {0: 'Class_0', 1: 'Class_1', ...}
        name_lookup = class_names
    else:
        # If class_names is a list like ['Class_0', 'Class_1', ...]
        name_lookup = {i: name for i, name in enumerate(class_names)}

    # Print distribution
    print(f"   {'Label':<6} {'Class':<20} {'Count':<8} {'Percentage':<10}")
    print(f"   {'-'*6} {'-'*20} {'-'*8} {'-'*10}")
    for label, count in sorted_labels:
        class_name = name_lookup.get(label, f"Unknown Class {label}")
        percentage = (count / total_labels) * 100
        print(f"   {label:<6} {class_name:<20} {count:<8} {percentage:>6.2f}%")

    # Check class balance
    class_counts = list(label_counter.values())
    if class_counts:
        min_count = min(class_counts)
        max_count = max(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        if imbalance_ratio > 10:
            print(
                f"   ❌ WARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.1f})"
            )
            print("      Consider class weighting or resampling techniques")
        elif imbalance_ratio > 3:
            print(
                f"   ⚠️ Moderate class imbalance detected (ratio: {imbalance_ratio:.1f})"
            )
        else:
            print(
                f"   ✓ Class distribution is relatively balanced (ratio: {imbalance_ratio:.1f})"
            )

    # Display sample images
    if display_samples and sample_images:
        print("\n4. SAMPLE IMAGES:")

        fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 3))
        if len(sample_images) == 1:
            axes = [axes]  # Make iterable for single image case

        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            # Handle different channel configurations
            if img.shape[0] == 1:  # Grayscale
                display_img = img[0]
                axes[i].imshow(display_img, cmap="gray")
            elif img.shape[0] == 3:  # RGB
                # Transpose from (C,H,W) to (H,W,C) for matplotlib
                display_img = np.transpose(img, (1, 2, 0))

                # Handle different normalization schemes
                if np.min(img) < 0 or np.max(img) > 1:
                    # Rescale for display if not in [0,1]
                    display_img = (display_img - np.min(display_img)) / (
                        np.max(display_img) - np.min(display_img)
                    )

                axes[i].imshow(display_img)

            # Get class name
            class_name = name_lookup.get(label, f"Class {label}")
            axes[i].set_title(f"{class_name}\nLabel: {label}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    print("\n5. SUGGESTIONS:")
    if len(image_shapes) > 1:
        print("   • Make sure all images are resized to the same dimensions")

    if nan_count > 0 or infinity_count > 0:
        print(
            "   • Check your preprocessing pipeline for divisions by zero or log of zero"
        )
        print(
            "   • Add nan_to_num in your transforms to handle NaN and infinity values"
        )

    if np.max(max_vals) > 10:
        print("   • Apply normalization to your images:")
        print(
            "     - For [0,1] range: transforms.Normalize([0.5], [0.5]) for grayscale"
        )
        print(
            "     - For [0,1] range: transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) for RGB"
        )
        print("     - Or use ImageNet normalization for transfer learning")

    if 1 in image_dims and "GrayscaleToRGB" not in str(dataloader.dataset):
        print("   • Your images are grayscale but models expect RGB:")
        print("     - Use your existing GrayscaleToRGB module, or")
        print("     - Add transforms.Grayscale(3) to convert to 3 identical channels")

    if imbalance_ratio > 3:
        print("   • For the class imbalance, consider:")
        print("     - Using weighted sampling: WeightedRandomSampler")
        print("     - Adding class weights to your loss function")
        print("     - Data augmentation for minority classes")
        print("     - Undersampling majority classes or oversampling minority classes")


# Example usage:
"""
# Validate training data
validate_dataset(dataloaders['train'], class_names, display_samples=True)

# Validate validation data
validate_dataset(dataloaders['val'], class_names, display_samples=True)
"""
