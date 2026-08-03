"""
Image preprocessing utilities for TB X-ray analysis.

This module prepares chest X-ray images
before passing them into deep learning models.
"""

from PIL import Image
import torch
from torchvision import transforms


# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path):
    """
    Load and preprocess an X-ray image.

    Args:
        image_path (str): Path to X-ray image

    Returns:
        torch.Tensor: Preprocessed image tensor
    """

    image = Image.open(image_path)

    # Convert grayscale X-ray to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transformations
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor