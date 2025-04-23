import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class CustomImageTransform(object):
    """
    Custom transform that applies:
    1. Grayscaling
    2. Histogram Equalization
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Gaussian Blur
    5. Median Blur
    
    Can be used in a transforms.Compose pipeline.
    """
    
    def __init__(self, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                 gaussian_kernel_size=5, gaussian_sigma=0, median_kernel_size=5):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.median_kernel_size = median_kernel_size
        
    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: A PyTorch tensor of shape [C, H, W] in range [0, 1]
        
        Returns:
            Transformed PyTorch tensor
        """
        # Convert tensor to numpy array (HWC format)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # 1. Convert to grayscale
        if img_np.shape[2] == 3:  # RGB image
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            img_gray = img_np.squeeze()
        
        # 2. Histogram Equalization
        img_eq = cv2.equalizeHist(img_gray)
        
        # 3. CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
        img_clahe = clahe.apply(img_eq)
        
        # 4. Gaussian Blur
        img_gaussian = cv2.GaussianBlur(
            img_clahe, 
            (self.gaussian_kernel_size, self.gaussian_kernel_size),
            self.gaussian_sigma
        )
        
        # 5. Median Blur
        img_median = cv2.medianBlur(img_gaussian, self.median_kernel_size)
        
        # Convert back to tensor [C, H, W] and normalize to [0, 1]
        result = torch.from_numpy(img_median).float() / 255.0
        result = result.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        return result
