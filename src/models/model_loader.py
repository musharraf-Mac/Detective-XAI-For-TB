"""
Model loading utilities for TB XAI system.
"""

import torch


def load_model(model_path):
    """
    Load trained PyTorch model.

    Args:
        model_path (str): path to saved model

    Returns:
        model: loaded PyTorch model
    """

    model = torch.load(
        model_path,
        map_location=torch.device("cpu")
    )

    model.eval()

    return model