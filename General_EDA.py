import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from collections import Counter
import glob
from datetime import datetime
import logging
import statistics
from typing import List, Dict, Tuple, Any, Optional, Union
import shutil
from tqdm import tqdm
import gc


class Logger:
    """Class to handle logging operations throughout the EDA process."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to store logs
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"eda_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("EyeDatasetEDA")
        
    def info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
        
    def error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: Error message to log
        """
        self.logger.error(message)
        
    def warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Warning message to log
        """
        self.logger.warning(message)


class ImageDataLoader:
    """Class to handle loading and basic processing of image data."""
    
    def __init__(self, data_dir: str, supported_extensions: List[str] = None, logger: Logger = None):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the image dataset
            supported_extensions: List of supported image file extensions
            logger: Logger instance
        """
        self.data_dir = data_dir
        self.supported_extensions = supported_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        self.logger = logger or Logger()
        self.image_paths = []
        self.metadata = {}
        self.class_distribution = {}
        
        # Validate data directory
        if not os.path.exists(data_dir):
            self.logger.error(f"Data directory {data_dir} does not exist")
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
            
        self.logger.info(f"ImageDataLoader initialized with data directory: {data_dir}")
    
    def scan_directory(self) -> None:
        """Scan the data directory to find all image files and organize them."""
        self.logger.info("Scanning directory for image files...")
        
        self.image_paths = []
        for ext in self.supported_extensions:
            pattern = os.path.join(self.data_dir, f"**/*{ext}")
            self.image_paths.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(self.data_dir, f"**/*{ext.upper()}")
            self.image_paths.extend(glob.glob(pattern, recursive=True))
        
        self.logger.info(f"Found {len(self.image_paths)} image files")
        
        # Try to extract class information from directory structure
        self._extract_class_info()
    
    def _extract_class_info(self) -> None:
        """Extract class information from directory structure."""
        classes = []
        
        for path in self.image_paths:
            # Assuming class name is the parent directory name
            class_name = os.path.basename(os.path.dirname(path))
            classes.append(class_name)
            self.metadata[path] = {'class': class_name}
        
        self.class_distribution = Counter(classes)
        self.logger.info(f"Detected {len(self.class_distribution)} classes: {dict(self.class_distribution)}")
    
    def load_image(self, image_path: str, as_grayscale: bool = False, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Load an image from file.
        
        Args:
            image_path: Path to the image file
            as_grayscale: Whether to load as grayscale
            target_size: Optional size to resize to (width, height)
            
        Returns:
            Loaded image as numpy array
        """
        try:
            if as_grayscale:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            if img is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
                
            if target_size:
                img = cv2.resize(img, target_size)
                
            return img
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
            
    def get_sample_batch(self, num_samples: int = 5, as_grayscale: bool = False, 
                         target_size: Tuple[int, int] = None, random_seed: int = None) -> Dict[str, np.ndarray]:
        """Get a sample batch of images.
        
        Args:
            num_samples: Number of samples to load
            as_grayscale: Whether to load as grayscale
            target_size: Optional size to resize to (width, height)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary of image paths and corresponding image arrays
        """
        if not self.image_paths:
            self.scan_directory()
            
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if num_samples > len(self.image_paths):
            self.logger.warning(f"Requested {num_samples} samples, but only {len(self.image_paths)} images available")
            num_samples = len(self.image_paths)
            
        sample_indices = np.random.choice(len(self.image_paths), num_samples, replace=False)
        sample_paths = [self.image_paths[i] for i in sample_indices]
        
        samples = {}
        for path in sample_paths:
            img = self.load_image(path, as_grayscale, target_size)
            if img is not None:
                samples[path] = img
                
        return samples
    
    def load_all_images(self, as_grayscale: bool = False, target_size: Tuple[int, int] = None, 
                       max_samples: int = None) -> Dict[str, np.ndarray]:
        """Load all images from the dataset.
        
        Args:
            as_grayscale: Whether to load as grayscale
            target_size: Optional size to resize to (width, height)
            max_samples: Maximum number of samples to load
            
        Returns:
            Dictionary of image paths and corresponding image arrays
        """
        if not self.image_paths:
            self.scan_directory()
            
        if max_samples and max_samples < len(self.image_paths):
            self.logger.info(f"Loading {max_samples} images out of {len(self.image_paths)}")
            paths = self.image_paths[:max_samples]
        else:
            self.logger.info(f"Loading all {len(self.image_paths)} images")
            paths = self.image_paths
            
        images = {}
        for path in tqdm(paths, desc="Loading images"):
            img = self.load_image(path, as_grayscale, target_size)
            if img is not None:
                images[path] = img
                
        return images


class ImageAnalyzer:
    """Class to perform analysis on eye image data."""
    
    def __init__(self, logger: Logger = None):
        """Initialize the image analyzer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or Logger()
        self.analysis_results = {}
    
    def compute_basic_stats(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute basic statistics of the images.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            
        Returns:
            Dictionary of basic statistics
        """
        self.logger.info("Computing basic image statistics...")
        
        stats = {
            'count': len(images),
            'image_shapes': [],
            'aspect_ratios': [],
            'mean_pixel_values': [],
            'std_pixel_values': [],
            'min_pixel_values': [],
            'max_pixel_values': [],
        }
        
        for path, img in images.items():
            stats['image_shapes'].append(img.shape)
            
            if len(img.shape) >= 2:
                aspect_ratio = img.shape[1] / img.shape[0]  # width/height
                stats['aspect_ratios'].append(aspect_ratio)
                
            if len(img.shape) == 2:  # Grayscale
                stats['mean_pixel_values'].append(np.mean(img))
                stats['std_pixel_values'].append(np.std(img))
                stats['min_pixel_values'].append(np.min(img))
                stats['max_pixel_values'].append(np.max(img))
            elif len(img.shape) == 3:  # Color
                # Per channel stats
                for c in range(img.shape[2]):
                    channel = img[:, :, c]
                    stats.setdefault(f'channel_{c}_means', []).append(np.mean(channel))
                    stats.setdefault(f'channel_{c}_stds', []).append(np.std(channel))
                    stats.setdefault(f'channel_{c}_mins', []).append(np.min(channel))
                    stats.setdefault(f'channel_{c}_maxs', []).append(np.max(channel))
                
                # Overall stats
                stats['mean_pixel_values'].append(np.mean(img))
                stats['std_pixel_values'].append(np.std(img))
                stats['min_pixel_values'].append(np.min(img))
                stats['max_pixel_values'].append(np.max(img))
        
        # Compute aggregate statistics
        if stats['image_shapes']:
            shape_counter = Counter(str(shape) for shape in stats['image_shapes'])
            stats['shape_distribution'] = dict(shape_counter)
            stats['most_common_shape'] = shape_counter.most_common(1)[0][0]
        
        if stats['aspect_ratios']:
            stats['mean_aspect_ratio'] = np.mean(stats['aspect_ratios'])
            stats['min_aspect_ratio'] = np.min(stats['aspect_ratios'])
            stats['max_aspect_ratio'] = np.max(stats['aspect_ratios'])
        
        if stats['mean_pixel_values']:
            stats['overall_mean_pixel_value'] = np.mean(stats['mean_pixel_values'])
            stats['overall_std_pixel_value'] = np.mean(stats['std_pixel_values'])
            stats['overall_min_pixel_value'] = np.min(stats['min_pixel_values'])
            stats['overall_max_pixel_value'] = np.max(stats['max_pixel_values'])
        
        self.analysis_results['basic_stats'] = stats
        
        # Clear memory
        del images
        gc.collect()
        
        return stats
    
    def detect_anomalies(self, images: Dict[str, np.ndarray], z_score_threshold: float = 3.0) -> Dict[str, List[str]]:
        """Detect anomalies in the image dataset.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            z_score_threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary of anomaly types with lists of anomalous image paths
        """
        self.logger.info("Detecting image anomalies...")
        
        anomalies = {
            'corrupted_images': [],
            'unusual_shapes': [],
            'unusual_aspect_ratios': [],
            'very_dark_images': [],
            'very_bright_images': [],
            'low_contrast_images': [],
            'high_contrast_images': [],
            'unusual_min_max': [],
        }
        
        # Calculate mean and std for shapes, aspect ratios, etc.
        heights = []
        widths = []
        aspect_ratios = []
        means = []
        stds = []
        contrast_values = []
        
        # First pass to gather statistics
        for path, img in images.items():
            if img is None or img.size == 0:
                anomalies['corrupted_images'].append(path)
                continue
                
            if len(img.shape) >= 2:
                heights.append(img.shape[0])
                widths.append(img.shape[1])
                aspect_ratios.append(img.shape[1] / img.shape[0])
                
            if len(img.shape) == 2:  # Grayscale
                means.append(np.mean(img))
                stds.append(np.std(img))
                contrast_values.append(np.max(img) - np.min(img))
            elif len(img.shape) == 3:  # Color
                means.append(np.mean(img))
                stds.append(np.std(img))
                contrast_values.append(np.max(img) - np.min(img))
        
        # Calculate z-scores and identify anomalies
        if heights and widths:
            mean_height, std_height = np.mean(heights), np.std(heights)
            mean_width, std_width = np.mean(widths), np.std(widths)
            
            for path, img in images.items():
                if img is None or img.size == 0:
                    continue
                    
                if len(img.shape) >= 2:
                    height, width = img.shape[0], img.shape[1]
                    height_z = abs(height - mean_height) / (std_height if std_height > 0 else 1)
                    width_z = abs(width - mean_width) / (std_width if std_width > 0 else 1)
                    
                    if height_z > z_score_threshold or width_z > z_score_threshold:
                        anomalies['unusual_shapes'].append(path)
        
        if aspect_ratios:
            mean_ar, std_ar = np.mean(aspect_ratios), np.std(aspect_ratios)
            
            for path, img in images.items():
                if img is None or img.size == 0:
                    continue
                    
                if len(img.shape) >= 2:
                    ar = img.shape[1] / img.shape[0]
                    ar_z = abs(ar - mean_ar) / (std_ar if std_ar > 0 else 1)
                    
                    if ar_z > z_score_threshold:
                        anomalies['unusual_aspect_ratios'].append(path)
        
        if means and stds:
            mean_mean, std_mean = np.mean(means), np.std(means)
            mean_std, std_std = np.mean(stds), np.std(stds)
            mean_contrast, std_contrast = np.mean(contrast_values), np.std(contrast_values)
            
            for path, img in images.items():
                if img is None or img.size == 0:
                    continue
                
                img_mean = np.mean(img)
                img_std = np.std(img)
                img_contrast = np.max(img) - np.min(img)
                
                mean_z = (img_mean - mean_mean) / (std_mean if std_mean > 0 else 1)
                std_z = (img_std - mean_std) / (std_std if std_std > 0 else 1)
                contrast_z = (img_contrast - mean_contrast) / (std_contrast if std_contrast > 0 else 1)
                
                if mean_z < -z_score_threshold:
                    anomalies['very_dark_images'].append(path)
                if mean_z > z_score_threshold:
                    anomalies['very_bright_images'].append(path)
                    
                if std_z < -z_score_threshold or contrast_z < -z_score_threshold:
                    anomalies['low_contrast_images'].append(path)
                if std_z > z_score_threshold or contrast_z > z_score_threshold:
                    anomalies['high_contrast_images'].append(path)
                    
                if np.min(img) > 50 or np.max(img) < 200:  # Unusual min/max values for typical 8-bit images
                    anomalies['unusual_min_max'].append(path)
        
        self.analysis_results['anomalies'] = anomalies
        return anomalies
    
    def perform_pca_analysis(self, images: Dict[str, np.ndarray], n_components: int = 10, 
                            sample_size: int = 100) -> Dict[str, Any]:
        """Perform PCA analysis on the image dataset.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            n_components: Number of PCA components
            sample_size: Number of images to sample for PCA
            
        Returns:
            Dictionary of PCA analysis results
        """
        self.logger.info(f"Performing PCA analysis with {n_components} components...")
        
        # Prepare images for PCA
        image_paths = list(images.keys())
        if len(image_paths) > sample_size:
            selected_paths = np.random.choice(image_paths, sample_size, replace=False)
        else:
            selected_paths = image_paths
            
        # Ensure all images are the same shape for PCA
        first_img = images[selected_paths[0]]
        target_shape = first_img.shape[:2]  # (height, width)
        
        image_matrices = []
        final_paths = []
        
        for path in selected_paths:
            img = images[path]
            
            # Skip images with different number of channels
            if len(img.shape) != len(first_img.shape):
                continue
                
            # Resize if needed
            if img.shape[:2] != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]))
            
            # Convert to grayscale if color
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img
                
            # Flatten the image
            flat_img = gray_img.flatten()
            image_matrices.append(flat_img)
            final_paths.append(path)
            
        if not image_matrices:
            self.logger.error("No valid images for PCA analysis")
            return {}
            
        # Perform PCA
        X = np.array(image_matrices)
        pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
        transformed = pca.fit_transform(X)
        
        # Prepare results
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components': n_components,
            'components': pca.components_.tolist(),
            'transformed_data': transformed.tolist(),
            'image_paths': final_paths
        }
        
        self.analysis_results['pca'] = pca_results
        return pca_results


class ImagePreprocessor:
    """Class to preprocess eye images for further analysis."""
    
    def __init__(self, output_dir: str = "preprocessed", logger: Logger = None):
        """Initialize the image preprocessor.
        
        Args:
            output_dir: Directory to save preprocessed images
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or Logger()
        os.makedirs(output_dir, exist_ok=True)
        
    def preprocess_image(self, img: np.ndarray, preprocessing_steps: List[str] = None) -> np.ndarray:
        """Preprocess a single image with the specified steps.
        
        Args:
            img: Input image as numpy array
            preprocessing_steps: List of preprocessing steps to apply
            
        Returns:
            Preprocessed image as numpy array
        """
        if img is None:
            return None
            
        # Default preprocessing steps
        preprocessing_steps = preprocessing_steps or [
            'resize', 'grayscale', 'histogram_equalization', 
            'gaussian_blur', 'normalize'
        ]
        
        processed_img = img.copy()
        
        for step in preprocessing_steps:
            if step == 'resize':
                # Resize to a standard size
                processed_img = cv2.resize(processed_img, (256, 256))
                
            elif step == 'grayscale' and len(processed_img.shape) == 3:
                # Convert to grayscale if it's a color image
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                
            elif step == 'histogram_equalization' and len(processed_img.shape) == 2:
                # Apply histogram equalization to enhance contrast
                processed_img = cv2.equalizeHist(processed_img)
                
            elif step == 'clahe' and len(processed_img.shape) == 2:
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed_img = clahe.apply(processed_img)
                
            elif step == 'gaussian_blur':
                # Apply Gaussian blur to reduce noise
                processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
                
            elif step == 'median_blur':
                # Apply median blur to reduce salt-and-pepper noise
                processed_img = cv2.medianBlur(processed_img, 5)
                
            elif step == 'normalize':
                # Normalize pixel values to [0, 1] range
                processed_img = processed_img.astype(np.float32) / 255.0
                
            elif step == 'standardize':
                # Standardize to mean=0, std=1
                mean = np.mean(processed_img)
                std = np.std(processed_img)
                processed_img = (processed_img - mean) / (std if std > 0 else 1)
                
            elif step == 'adjust_gamma':
                # Adjust gamma to enhance features
                gamma = 1.2
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
                processed_img = cv2.LUT(processed_img.astype(np.uint8), lookup_table)

        return processed_img
    
    def batch_preprocess(self, images: Dict[str, np.ndarray], preprocessing_steps: List[str] = None, 
                        save_images: bool = True) -> Dict[str, np.ndarray]:
        """Preprocess a batch of images.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            preprocessing_steps: List of preprocessing steps to apply
            save_images: Whether to save preprocessed images to disk
            
        Returns:
            Dictionary of image paths and corresponding preprocessed image arrays
        """
        self.logger.info(f"Batch preprocessing {len(images)} images...")
        
        preprocessed_images = {}
        
        for path, img in tqdm(images.items(), desc="Preprocessing images"):
            preprocessed_img = self.preprocess_image(img, preprocessing_steps)
            
            if preprocessed_img is not None:
                preprocessed_images[path] = preprocessed_img
                
                if save_images:
                    # Create file path for saving preprocessed image
                    filename = os.path.basename(path)
                    output_path = os.path.join(self.output_dir, filename)
                    
                    # Convert back to uint8 if necessary
                    save_img = preprocessed_img
                    if np.issubdtype(save_img.dtype, np.floating):
                        save_img = (save_img * 255).astype(np.uint8)
                        
                    # Save the image
                    cv2.imwrite(output_path, save_img)
        
        self.logger.info(f"Finished preprocessing {len(preprocessed_images)} images")
        return preprocessed_images


class Visualizer:
    """Class to create visualizations for the eye image dataset."""
    
    def __init__(self, output_dir: str = "visualizations", logger: Logger = None):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or Logger()
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_sample_images(self, images: Dict[str, np.ndarray], num_samples: int = 9, 
                               figsize: Tuple[int, int] = (12, 12), save_path: str = None) -> None:
        """Visualize a grid of sample images.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            num_samples: Number of samples to visualize
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        self.logger.info(f"Visualizing {num_samples} sample images...")
        
        # Select random samples
        image_paths = list(images.keys())
        if len(image_paths) > num_samples:
            selected_paths = np.random.choice(image_paths, num_samples, replace=False)
        else:
            selected_paths = image_paths[:num_samples]
            
        # Determine grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(selected_paths))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()
        
        for i, path in enumerate(selected_paths):
            if i < len(axes):
                img = images[path]
                
                # Handle grayscale vs. color images
                if len(img.shape) == 2:
                    axes[i].imshow(img, cmap='gray')
                else:
                    axes[i].imshow(img)
                    
                # Show image filename as title
                axes[i].set_title(os.path.basename(path), fontsize=8)
                axes[i].axis('off')
        
        # Hide unused axes
        for i in range(len(selected_paths), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sample_images.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del selected_paths, image_paths
        gc.collect()
    
    def visualize_class_distribution(self, class_distribution: Dict[str, int], 
                                    figsize: Tuple[int, int] = (10, 6), save_path: str = None) -> None:
        """Visualize the class distribution of the dataset.
        
        Args:
            class_distribution: Dictionary of class names and their counts
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        self.logger.info("Visualizing class distribution...")
        
        plt.figure(figsize=figsize)
        
        # Sort classes by count in descending order
        sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_classes)
        
        # Create bar chart
        bars = plt.bar(classes, counts, color='steelblue')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'class_distribution.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del classes, counts
        gc.collect()
        
    def visualize_image_sizes(self, images: Dict[str, np.ndarray], 
                             figsize: Tuple[int, int] = (12, 8), save_path: str = None) -> None:
        """Visualize the distribution of image sizes.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        self.logger.info("Visualizing image size distribution...")
        
        # Extract image dimensions
        widths = []
        heights = []
        aspect_ratios = []
        
        for img in images.values():
            if img is not None and len(img.shape) >= 2:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        
        del img  # Clear memory
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot width distribution
        axes[0, 0].hist(widths, bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot height distribution
        axes[0, 1].hist(heights, bins=20, color='salmon', edgecolor='black')
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot aspect ratio distribution
        axes[1, 0].hist(aspect_ratios, bins=20, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot width vs height scatter
        axes[1, 1].scatter(widths, heights, alpha=0.5, c='purple', s=20)
        axes[1, 1].set_title('Width vs. Height')
        axes[1, 1].set_xlabel('Width (pixels)')
        axes[1, 1].set_ylabel('Height (pixels)')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'image_sizes.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del widths, heights, aspect_ratios
        gc.collect()
        
    def visualize_pixel_value_distribution(self, images: Dict[str, np.ndarray], sample_size: int = 20,
                                        figsize: Tuple[int, int] = (12, 8), save_path: str = None) -> None:
        """Visualize the pixel value distribution.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            sample_size: Number of images to sample for visualization
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        self.logger.info("Visualizing pixel value distribution...")
        
        # Sample images
        image_paths = list(images.keys())
        if len(image_paths) > sample_size:
            selected_paths = np.random.choice(image_paths, sample_size, replace=False)
        else:
            selected_paths = image_paths
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Initialize histograms instead of collecting all values
        overall_hist = np.zeros(256)
        channel_hists = [np.zeros(256) for _ in range(3)]
        
        # Values for boxplots (use sampling to reduce memory)
        box_sample_size = 10000
        all_samples = []
        channel_samples = [[] for _ in range(3)]
        
        # Process images one at a time
        for path in selected_paths:
            img = images[path]
            
            if img is None:
                continue
            
            # For overall histogram
            if len(img.shape) == 2:  # Grayscale
                # Update histogram in-place without storing all values
                hist, _ = np.histogram(img, bins=256, range=(0, 255))
                overall_hist += hist
                
                # Sample for boxplot
                if len(all_samples) < box_sample_size:
                    flat_img = img.flatten()
                    sample_idx = np.random.choice(len(flat_img), min(1000, len(flat_img)), replace=False)
                    all_samples.extend(flat_img[sample_idx])
                
            elif len(img.shape) == 3 and img.shape[2] == 3:  # Color
                # Overall histogram
                hist, _ = np.histogram(img, bins=256, range=(0, 255))
                overall_hist += hist
                
                # Channel histograms
                for c in range(3):
                    channel_hist, _ = np.histogram(img[:,:,c], bins=256, range=(0, 255))
                    channel_hists[c] += channel_hist
                    
                    # Sample for boxplot
                    if len(channel_samples[c]) < box_sample_size:
                        channel_img = img[:,:,c].flatten()
                        sample_idx = np.random.choice(len(channel_img), 
                                                    min(500, len(channel_img)), 
                                                    replace=False)
                        channel_samples[c].extend(channel_img[sample_idx])
                
                # Sample for overall boxplot
                if len(all_samples) < box_sample_size:
                    flat_img = img.flatten()
                    sample_idx = np.random.choice(len(flat_img), min(1000, len(flat_img)), replace=False)
                    all_samples.extend(flat_img[sample_idx])
            
            # Clear the img from memory if we're done with it
            del img
        
        # Plot overall pixel value distribution
        axes[0, 0].bar(range(256), overall_hist, color='gray', alpha=0.7)
        axes[0, 0].set_title('Overall Pixel Value Distribution')
        axes[0, 0].set_xlabel('Pixel Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot cumulative distribution
        cum_hist = np.cumsum(overall_hist)
        if cum_hist.max() > 0:  # Avoid division by zero
            cum_hist = cum_hist / cum_hist.max()
        axes[0, 1].plot(range(256), cum_hist, color='gray')
        axes[0, 1].set_title('Cumulative Pixel Value Distribution')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Cumulative Frequency')
        
        # Plot channel-specific distributions if we have color images
        has_color = any(np.sum(hist) > 0 for hist in channel_hists)
        if has_color:
            colors = ['red', 'green', 'blue']
            for c in range(3):
                axes[1, 0].bar(range(256), channel_hists[c], color=colors[c], alpha=0.5, label=colors[c].capitalize())
                
            axes[1, 0].set_title('RGB Channel Distributions')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            
            # Plot box plots of channel values
            if all(len(samples) > 0 for samples in channel_samples):
                axes[1, 1].boxplot(channel_samples, labels=['Red', 'Green', 'Blue'], 
                            patch_artist=True, boxprops=dict(facecolor='whitesmoke'))
                axes[1, 1].set_title('Channel Value Boxplots')
                axes[1, 1].set_ylabel('Pixel Value')
        else:
            # If no color images, create alternative plots
            if all_samples:
                axes[1, 0].boxplot(all_samples, patch_artist=True, 
                            boxprops=dict(facecolor='whitesmoke'))
                axes[1, 0].set_title('Pixel Value Boxplot')
                axes[1, 0].set_ylabel('Pixel Value')
            
            # Plot a sample histogram for a random image
            if selected_paths:
                sample_path = selected_paths[0]
                sample_img = images[sample_path]
                if sample_img is not None:
                    sample_hist, _ = np.histogram(sample_img, bins=50, range=(0, 255))
                    axes[1, 1].bar(range(50), sample_hist, color='black', alpha=0.7)
                    axes[1, 1].set_title(f'Sample Image Histogram: {os.path.basename(sample_path)}')
                    axes[1, 1].set_xlabel('Pixel Value')
                    axes[1, 1].set_ylabel('Frequency')
                    del sample_img  # Free memory
        
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'pixel_distribution.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del overall_hist, channel_hists, all_samples, channel_samples
        gc.collect()
        
    def visualize_pca_results(self, pca_results: Dict[str, Any], 
                             figsize: Tuple[int, int] = (12, 10), save_path: str = None) -> None:
        """Visualize PCA results.
        
        Args:
            pca_results: Dictionary of PCA analysis results
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        if not pca_results:
            self.logger.warning("No PCA results to visualize")
            return
            
        self.logger.info("Visualizing PCA results...")
        
        # Extract data from PCA results
        explained_variance_ratio = pca_results.get('explained_variance_ratio', [])
        cumulative_explained_variance = pca_results.get('cumulative_explained_variance', [])
        transformed_data = np.array(pca_results.get('transformed_data', []))
        
        if not explained_variance_ratio or transformed_data.size == 0:
            self.logger.warning("Insufficient PCA data for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot explained variance ratio
        components = list(range(1, len(explained_variance_ratio) + 1))
        axes[0, 0].bar(components, explained_variance_ratio, color='steelblue')
        axes[0, 0].set_title('Explained Variance by Component')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        
        # Plot cumulative explained variance
        axes[0, 1].plot(components, cumulative_explained_variance, 'o-', color='orangered')
        axes[0, 1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.7)  # 90% threshold line
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot first two principal components
        if transformed_data.shape[1] >= 2:
            axes[1, 0].scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.6)
            axes[1, 0].set_title('First Two Principal Components')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)
            
        # Plot first three principal components if available
        if transformed_data.shape[1] >= 3:
            ax3d = fig.add_subplot(2, 2, 4, projection='3d')
            ax3d.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], alpha=0.6)
            ax3d.set_title('First Three Principal Components')
            ax3d.set_xlabel('PC1')
            ax3d.set_ylabel('PC2')
            ax3d.set_zlabel('PC3')
        else:
            # If we don't have 3 components, show something else
            axes[1, 1].text(0.5, 0.5, "Not enough components for 3D plot", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'pca_visualization.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del explained_variance_ratio, cumulative_explained_variance, transformed_data
        gc.collect()
        
    def visualize_preprocessing_steps(self, original_image: np.ndarray, 
                                    preprocessing_steps: List[str] = None,
                                    figsize: Tuple[int, int] = (15, 10), 
                                    save_path: str = None) -> None:
        """Visualize the effect of preprocessing steps on a sample image.
        
        Args:
            original_image: Original image to preprocess
            preprocessing_steps: List of preprocessing steps to apply
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        self.logger.info("Visualizing preprocessing steps...")
        
        # Default preprocessing steps
        preprocessing_steps = preprocessing_steps or [
            'grayscale', 'histogram_equalization', 'clahe', 
            'gaussian_blur', 'median_blur'
        ]
        
        # Create figure with subplots
        n_steps = len(preprocessing_steps) + 1  # +1 for original image
        rows = int(np.ceil(n_steps / 3))
        cols = min(n_steps, 3)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = np.array([axes])  # Ensure axes is 2D
        axes = axes.flatten()
        
        # Display original image
        if len(original_image.shape) == 2:
            axes[0].imshow(original_image, cmap='gray')
        else:
            axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Apply each preprocessing step and display result
        current_img = original_image.copy()
        for i, step in enumerate(preprocessing_steps, 1):
            
            # Apply preprocessing step
            if step == 'grayscale' and len(current_img.shape) == 3:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
                
            elif step == 'histogram_equalization' and len(current_img.shape) == 2:
                current_img = cv2.equalizeHist(current_img)
                
            elif step == 'clahe' and len(current_img.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                current_img = clahe.apply(current_img)
                
            elif step == 'gaussian_blur':
                current_img = cv2.GaussianBlur(current_img, (5, 5), 0)
                
            elif step == 'median_blur':
                current_img = cv2.medianBlur(current_img, 5)
                
            elif step == 'normalize':
                # Convert back to uint8 for visualization
                current_img = ((current_img - np.min(current_img)) / 
                            (np.max(current_img) - np.min(current_img)) * 255).astype(np.uint8)
                
            elif step == 'adjust_gamma':
                # Adjust gamma to enhance features
                gamma = 1.2
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
                current_img = cv2.LUT(current_img.astype(np.uint8), lookup_table)
            
            # Display result
            if i < len(axes):
                if len(current_img.shape) == 2:
                    axes[i].imshow(current_img, cmap='gray')
                else:
                    axes[i].imshow(current_img)
                axes[i].set_title(f"After {step.replace('_', ' ').title()}")
                axes[i].axis('off')
        
        # Hide unused axes
        for i in range(n_steps, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'preprocessing_steps.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del original_image, current_img
        gc.collect()
        self.logger.info("Finished visualizing preprocessing steps")
        
    def visualize_anomalies(self, images: Dict[str, np.ndarray], anomalies: Dict[str, List[str]],
                           num_examples: int = 2, figsize: Tuple[int, int] = (15, 10),
                           save_path: str = None) -> None:
        """Visualize example images from each anomaly category.
        
        Args:
            images: Dictionary of image paths and corresponding image arrays
            anomalies: Dictionary of anomaly types with lists of anomalous image paths
            num_examples: Number of example images to show per anomaly type
            figsize: Figure size for the plot
            save_path: Path to save the visualization
        """
        self.logger.info("Visualizing anomaly examples...")
        
        # Count non-empty anomaly categories
        valid_categories = [cat for cat, paths in anomalies.items() if paths]
        if not valid_categories:
            self.logger.warning("No anomalies to visualize")
            return
            
        # Create figure with subplots
        n_categories = len(valid_categories)
        n_cols = min(num_examples, 4)
        n_rows = n_categories
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, category in enumerate(valid_categories):
            # Get the anomaly paths
            anomaly_paths = anomalies[category]
            
            # Display category name next to the first image
            axes[i, 0].text(-0.3, 0.5, category.replace('_', ' ').title(), 
                         rotation=90, va='center', ha='right', transform=axes[i, 0].transAxes,
                         fontsize=10, fontweight='bold')
            
            # Sample examples for this category
            sample_size = min(len(anomaly_paths), num_examples)
            if sample_size < 1:
                continue
                
            sample_paths = np.random.choice(anomaly_paths, sample_size, replace=False)
            
            # Display sample images
            for j, path in enumerate(sample_paths):
                if j < n_cols:
                    img = images.get(path)
                    if img is not None:
                        if len(img.shape) == 2:
                            axes[i, j].imshow(img, cmap='gray')
                        else:
                            axes[i, j].imshow(img)
                    axes[i, j].set_title(os.path.basename(path), fontsize=8)
                    axes[i, j].axis('off')
            
            # Hide unused axes
            for j in range(sample_size, n_cols):
                axes[i, j].axis('off')
                
        plt.tight_layout()
        
        # Save if requested
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'anomaly_examples.png')
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del images, anomalies
        gc.collect()


class ReportGenerator:
    """Class to generate EDA reports for eye image datasets."""
    
    def __init__(self, output_dir: str = "reports", logger: Logger = None):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or Logger()
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_html_report(self, data_dir: str, analysis_results: Dict[str, Any],
                           visualization_paths: List[str] = None) -> str:
        """Generate an HTML report of the EDA results.
        
        Args:
            data_dir: Directory containing the image dataset
            analysis_results: Dictionary of analysis results
            visualization_paths: List of paths to visualization images
            
        Returns:
            Path to the generated HTML report
        """
        self.logger.info("Generating HTML report...")
        
        report_path = os.path.join(self.output_dir, f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Eye Dataset EDA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .viz-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                .viz-item {{ flex: 1; min-width: 300px; margin-bottom: 20px; }}
                .viz-img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .anomaly-table {{ width: 100%; margin-bottom: 20px; }}
                .highlight {{ background-color: #ffffcc; }}
                .footer {{ margin-top: 40px; text-align: center; font-size: 0.8em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="section">
                <h1>Eye Dataset Exploratory Data Analysis Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Dataset:</strong> {os.path.basename(data_dir)}</p>
                <p><strong>Dataset Path:</strong> {data_dir}</p>
            </div>
        """
        
        # Dataset summary section
        basic_stats = analysis_results.get('basic_stats', {})
        if basic_stats:
            html_content += f"""
            <div class="section">
                <h2>Dataset Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Images</td><td>{basic_stats.get('count', 'N/A')}</td></tr>
                    <tr><td>Most Common Shape</td><td>{basic_stats.get('most_common_shape', 'N/A')}</td></tr>
                    <tr><td>Mean Aspect Ratio</td><td>{basic_stats.get('mean_aspect_ratio', 'N/A'):.3f}</td></tr>
                    <tr><td>Overall Mean Pixel Value</td><td>{basic_stats.get('overall_mean_pixel_value', 'N/A'):.3f}</td></tr>
                    <tr><td>Overall Standard Deviation</td><td>{basic_stats.get('overall_std_pixel_value', 'N/A'):.3f}</td></tr>
                </table>
            </div>
            """
            
        # Class distribution section
        class_distribution = analysis_results.get('class_distribution', {})
        if class_distribution:
            html_content += """
            <div class="section">
                <h2>Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
            """
            
            total = sum(class_distribution.values())
            for cls, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100 if total > 0 else 0
                html_content += f"<tr><td>{cls}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
                
            html_content += """
                </table>
            </div>
            """
            
        # Image size distribution section
        if basic_stats.get('shape_distribution'):
            html_content += """
            <div class="section">
                <h2>Image Size Distribution</h2>
                <table>
                    <tr><th>Shape</th><th>Count</th><th>Percentage</th></tr>
            """
            
            total = sum(basic_stats['shape_distribution'].values())
            for shape, count in sorted(basic_stats['shape_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100 if total > 0 else 0
                html_content += f"<tr><td>{shape}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
                
            html_content += """
                </table>
            </div>
            """
            
        # Anomaly detection section
        anomalies = analysis_results.get('anomalies', {})
        if anomalies:
            html_content += """
            <div class="section">
                <h2>Anomaly Detection Results</h2>
                <table class="anomaly-table">
                    <tr><th>Anomaly Type</th><th>Count</th><th>Percentage</th></tr>
            """
            
            total_images = basic_stats.get('count', 0)
            for anomaly_type, anomaly_paths in anomalies.items():
                count = len(anomaly_paths)
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                html_content += f"<tr><td>{anomaly_type.replace('_', ' ').title()}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
                
            html_content += """
                </table>
            </div>
            """
            
        # PCA analysis section
        pca_results = analysis_results.get('pca', {})
        if pca_results:
            html_content += """
            <div class="section">
                <h2>PCA Analysis</h2>
                <table>
                    <tr><th>Component</th><th>Explained Variance</th><th>Cumulative Variance</th></tr>
            """
            
            explained_variance = pca_results.get('explained_variance_ratio', [])
            cumulative_variance = pca_results.get('cumulative_explained_variance', [])
            
            for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
                html_content += f"<tr><td>PC{i+1}</td><td>{var:.4f}</td><td>{cum_var:.4f}</td></tr>"
                
            html_content += """
                </table>
            </div>
            """
            
        # Visualization section
        if visualization_paths:
            html_content += """
            <div class="section">
                <h2>Visualizations</h2>
                <div class="viz-container">
            """
            
            for viz_path in visualization_paths:
                if os.path.exists(viz_path):
                    filename = os.path.basename(viz_path)
                    title = filename.replace('_', ' ').replace('.png', '').title()
                    
                    # Create relative path for HTML
                    relative_path = os.path.relpath(viz_path, self.output_dir)
                    
                    html_content += f"""
                    <div class="viz-item">
                        <h3>{title}</h3>
                        <img class="viz-img" src="{relative_path}" alt="{title}">
                    </div>
                    """
                    
            html_content += """
                </div>
            </div>
            """
            
        # Recommendations section
        html_content += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
        """
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Class balance recommendations
        if class_distribution:
            max_count = max(class_distribution.values())
            min_count = min(class_distribution.values())
            if max_count > min_count * 3:
                recommendations.append("Dataset shows significant class imbalance. Consider data augmentation or oversampling of minority classes.")
                
        # Anomaly recommendations
        if anomalies:
            for anomaly_type, anomaly_paths in anomalies.items():
                if len(anomaly_paths) > 0:
                    recommendations.append(f"Review {len(anomaly_paths)} images identified as {anomaly_type.replace('_', ' ')}.")
            
        # Image size recommendations
        if basic_stats.get('shape_distribution', {}):
            if len(basic_stats['shape_distribution']) > 1:
                recommendations.append("Dataset contains images of different sizes. Consider standardizing image sizes for better model performance.")
                
        # Add recommendations to HTML
        if recommendations:
            for rec in recommendations:
                html_content += f"<li class='highlight'>{rec}</li>"
        else:
            html_content += "<li>No specific recommendations based on the current analysis.</li>"
                
        html_content += """
            </ul>
        </div>
        """
        
        # Close HTML
        html_content += """
            <div class="footer">
                <p>Report generated by Eye Dataset EDA Tool</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"HTML report saved to {report_path}")
        return report_path
        
    def generate_markdown_report(self, data_dir: str, analysis_results: Dict[str, Any]) -> str:
        """Generate a Markdown report of the EDA results.
        
        Args:
            data_dir: Directory containing the image dataset
            analysis_results: Dictionary of analysis results
            
        Returns:
            Path to the generated Markdown report
        """
        self.logger.info("Generating Markdown report...")
        
        report_path = os.path.join(self.output_dir, f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        # Start building Markdown content
        md_content = f"""
# Eye Dataset Exploratory Data Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** {os.path.basename(data_dir)}  
**Dataset Path:** {data_dir}

## Dataset Summary
"""
        
        # Basic stats section
        basic_stats = analysis_results.get('basic_stats', {})
        if basic_stats:
            md_content += f"""
| Metric | Value |
|--------|-------|
| Total Images | {basic_stats.get('count', 'N/A')} |
| Most Common Shape | {basic_stats.get('most_common_shape', 'N/A')} |
| Mean Aspect Ratio | {basic_stats.get('mean_aspect_ratio', 'N/A'):.3f} |
| Overall Mean Pixel Value | {basic_stats.get('overall_mean_pixel_value', 'N/A'):.3f} |
| Overall Standard Deviation | {basic_stats.get('overall_std_pixel_value', 'N/A'):.3f} |
"""
            
        # Class distribution section
        class_distribution = analysis_results.get('class_distribution', {})
        if class_distribution:
            md_content += """
## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
"""
            
            total = sum(class_distribution.values())
            for cls, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100 if total > 0 else 0
                md_content += f"| {cls} | {count} | {percentage:.2f}% |\n"
                
        # Image size distribution section
        if basic_stats.get('shape_distribution'):
            md_content += """
## Image Size Distribution

| Shape | Count | Percentage |
|-------|-------|------------|
"""
            
            total = sum(basic_stats['shape_distribution'].values())
            for shape, count in sorted(basic_stats['shape_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100 if total > 0 else 0
                md_content += f"| {shape} | {count} | {percentage:.2f}% |\n"
                
        # Anomaly detection section
        anomalies = analysis_results.get('anomalies', {})
        if anomalies:
            md_content += """
## Anomaly Detection Results

| Anomaly Type | Count | Percentage |
|--------------|-------|------------|
"""
            
            total_images = basic_stats.get('count', 0)
            for anomaly_type, anomaly_paths in anomalies.items():
                count = len(anomaly_paths)
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                md_content += f"| {anomaly_type.replace('_', ' ').title()} | {count} | {percentage:.2f}% |\n"

        # PCA analysis section
        pca_results = analysis_results.get('pca', {})
        if pca_results:
            md_content += """
## PCA Analysis

| Component | Explained Variance | Cumulative Variance |
|-----------|-------------------|---------------------|
"""
            
            explained_variance = pca_results.get('explained_variance_ratio', [])
            cumulative_variance = pca_results.get('cumulative_explained_variance', [])
            
            for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
                md_content += f"| PC{i+1} | {var:.4f} | {cum_var:.4f} |\n"
                
        # Recommendations section
        md_content += """
## Recommendations

"""
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # Class balance recommendations
        if class_distribution:
            max_count = max(class_distribution.values())
            min_count = min(class_distribution.values())
            if max_count > min_count * 3:
                recommendations.append("Dataset shows significant class imbalance. Consider data augmentation or oversampling of minority classes.")
                
        # Anomaly recommendations
        if anomalies:
            for anomaly_type, anomaly_paths in anomalies.items():
                if len(anomaly_paths) > 0:
                    recommendations.append(f"Review {len(anomaly_paths)} images identified as {anomaly_type.replace('_', ' ')}.")
            
        # Image size recommendations
        if basic_stats.get('shape_distribution', {}):
            if len(basic_stats['shape_distribution']) > 1:
                recommendations.append("Dataset contains images of different sizes. Consider standardizing image sizes for better model performance.")
                
        # Add recommendations to Markdown
        if recommendations:
            for rec in recommendations:
                md_content += f"- **{rec}**\n"
        else:
            md_content += "- No specific recommendations based on the current analysis.\n"
                
        # Write Markdown to file
        with open(report_path, 'w') as f:
            f.write(md_content)
            
        self.logger.info(f"Markdown report saved to {report_path}")
        return report_path


class EyeDatasetEDA:
    """Main class that orchestrates the EDA process for eye datasets."""
    
    def __init__(self, data_dir: str, output_dir: str = "eye_eda_results"):
        """Initialize the EDA process.
        
        Args:
            data_dir: Directory containing the image dataset
            output_dir: Directory to save all output files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger(os.path.join(output_dir, "logs"))
        self.logger.info(f"Initializing EDA for dataset: {data_dir}")
        
        # Initialize components
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data_loader = ImageDataLoader(data_dir, logger=self.logger)
        self.analyzer = ImageAnalyzer(logger=self.logger)
        self.preprocessor = ImagePreprocessor(os.path.join(output_dir, "preprocessed"), logger=self.logger)
        self.visualizer = Visualizer(os.path.join(output_dir, "visualizations"), logger=self.logger)
        self.report_generator = ReportGenerator(os.path.join(output_dir, "reports"), logger=self.logger)
        
        # Store analysis results and visualizations
        self.analysis_results = {}
        self.visualization_paths = []
        
        self.logger.info("EDA components initialized")
    
    def run_basic_analysis(self, max_samples: int = None, save_visualizations: bool = True) -> Dict[str, Any]:
        """Run a basic analysis of the dataset.
        
        Args:
            max_samples: Maximum number of samples to analyze
            save_visualizations: Whether to save visualizations
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Running basic analysis...")
        
        # Load dataset
        self.data_loader.scan_directory()
        
        # Get sample images
        sample_images = self.data_loader.get_sample_batch(num_samples=25)
        
        # Load all images for analysis
        images = self.data_loader.load_all_images(max_samples=max_samples)
        
        # Run basic image statistics
        basic_stats = self.analyzer.compute_basic_stats(images)
        
        # Visualize sample images
        if save_visualizations:
            viz_path = os.path.join(self.visualizer.output_dir, 'sample_images.png')
            self.visualizer.visualize_sample_images(sample_images, save_path=viz_path)
            self.visualization_paths.append(viz_path)
            
            # Visualize class distribution
            if self.data_loader.class_distribution:
                viz_path = os.path.join(self.visualizer.output_dir, 'class_distribution.png')
                self.visualizer.visualize_class_distribution(self.data_loader.class_distribution, save_path=viz_path)
                self.visualization_paths.append(viz_path)
                
            # Visualize image sizes
            viz_path = os.path.join(self.visualizer.output_dir, 'image_sizes.png')
            self.visualizer.visualize_image_sizes(images, save_path=viz_path)
            self.visualization_paths.append(viz_path)
            
            # Visualize pixel distribution
            viz_path = os.path.join(self.visualizer.output_dir, 'pixel_distribution.png')
            self.visualizer.visualize_pixel_value_distribution(images, save_path=viz_path)
            self.visualization_paths.append(viz_path)
            
        # Store results
        self.analysis_results['basic_stats'] = basic_stats
        self.analysis_results['class_distribution'] = self.data_loader.class_distribution
        
        return self.analysis_results
        
    def run_advanced_analysis(self, max_samples: int = None, save_visualizations: bool = True) -> Dict[str, Any]:
        """Run an advanced analysis of the dataset.
        
        Args:
            max_samples: Maximum number of samples to analyze
            save_visualizations: Whether to save visualizations
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Running advanced analysis...")
        
        # Ensure basic analysis has been run
        if not self.analysis_results:
            self.run_basic_analysis(max_samples, save_visualizations)
            
        # Load all images for analysis if not already loaded
        if 'images' not in locals():
            images = self.data_loader.load_all_images(max_samples=max_samples)
            
        # Run anomaly detection
        anomalies = self.analyzer.detect_anomalies(images)
        
        # Run PCA analysis
        pca_results = self.analyzer.perform_pca_analysis(images)
                
        # Visualizations
        if save_visualizations:
            # Visualize preprocessing steps on a sample image
            if images:
                sample_img = next(iter(images.values()))
                viz_path = os.path.join(self.visualizer.output_dir, 'preprocessing_steps.png')
                self.visualizer.visualize_preprocessing_steps(sample_img, save_path=viz_path)
                self.visualization_paths.append(viz_path)
                
            # Visualize PCA results
            if pca_results:
                viz_path = os.path.join(self.visualizer.output_dir, 'pca_results.png')
                self.visualizer.visualize_pca_results(pca_results, save_path=viz_path)
                self.visualization_paths.append(viz_path)
                
            # Visualize anomalies
            if anomalies:
                viz_path = os.path.join(self.visualizer.output_dir, 'anomalies.png')
                self.visualizer.visualize_anomalies(images, anomalies, save_path=viz_path)
                self.visualization_paths.append(viz_path)
                
        # Store results
        self.analysis_results['anomalies'] = anomalies
        self.analysis_results['pca'] = pca_results
        
        return self.analysis_results
        
    def preprocess_dataset(self, preprocessing_steps: List[str] = None, 
                         max_samples: int = None) -> Dict[str, np.ndarray]:
        """Preprocess the dataset with the specified steps.
        
        Args:
            preprocessing_steps: List of preprocessing steps to apply
            max_samples: Maximum number of samples to preprocess
            
        Returns:
            Dictionary of preprocessed images
        """
        self.logger.info("Preprocessing dataset...")
        
        # Load images if not already loaded
        images = self.data_loader.load_all_images(max_samples=max_samples)
        
        # Apply preprocessing steps
        preprocessed_images = self.preprocessor.batch_preprocess(images, preprocessing_steps)
        
        return preprocessed_images
        
    def generate_reports(self) -> Tuple[str, str]:
        """Generate reports from the analysis results.
        
        Returns:
            Tuple of paths to the generated HTML and Markdown reports
        """
        self.logger.info("Generating reports...")
        
        # Ensure analysis has been run
        if not self.analysis_results:
            self.logger.warning("No analysis results available. Running basic analysis first.")
            self.run_basic_analysis()
            
        # Generate HTML report
        html_report_path = self.report_generator.generate_html_report(
            self.data_dir, 
            self.analysis_results,
            self.visualization_paths
        )
        
        # Generate Markdown report
        md_report_path = self.report_generator.generate_markdown_report(
            self.data_dir, 
            self.analysis_results
        )
        
        return html_report_path, md_report_path
        
    def run_full_eda(self, max_samples: int = None) -> Dict[str, Any]:
        """Run the complete EDA pipeline.
        
        Args:
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Running full EDA pipeline...")
        
        # Run basic analysis
        self.run_basic_analysis(max_samples=max_samples)
        
        # Run advanced analysis
        self.run_advanced_analysis(max_samples=max_samples)
        
        # Preprocess dataset with default steps
        preprocessed_images = self.preprocess_dataset(max_samples=max_samples)
        
        # Generate reports
        html_report, md_report = self.generate_reports()
        
        self.logger.info(f"Full EDA pipeline completed. Reports saved to {html_report} and {md_report}")
        
        return self.analysis_results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Eye Dataset Exploratory Data Analysis")
    parser.add_argument("--data_dir", required=True, help="Directory containing the eye image dataset")
    parser.add_argument("--output_dir", default="eye_eda_results", help="Directory to save output files")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to analyze")
    parser.add_argument("--basic_only", action="store_true", help="Run only basic analysis")
    
    args = parser.parse_args()
    
    # Initialize EDA
    eda = EyeDatasetEDA(args.data_dir, args.output_dir)
    
    if args.basic_only:
        # Run only basic analysis
        results = eda.run_basic_analysis(max_samples=args.max_samples)
    else:
        # Run full EDA pipeline
        results = eda.run_full_eda(max_samples=args.max_samples)
        
    # Generate reports
    html_report, md_report = eda.generate_reports()
    
    print(f"EDA completed. Results saved to {args.output_dir}")
    print(f"HTML report: {html_report}")
    print(f"Markdown report: {md_report}")
