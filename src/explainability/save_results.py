"""
Utilities for saving Grad-CAM outputs.
"""

import os


def create_result_directory(path="results/grad_cam"):

    """
    Create directory for Grad-CAM outputs.
    """

    os.makedirs(
        path,
        exist_ok=True
    )

    return path