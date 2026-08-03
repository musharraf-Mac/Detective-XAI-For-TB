"""
Grad-CAM visualization utilities.
"""

import cv2
import numpy as np


def overlay_heatmap(image_path, heatmap, output_path):

    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image_path: original X-ray path
        heatmap: generated Grad-CAM heatmap
        output_path: save location
    """

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )


    heatmap = np.uint8(
        255 * heatmap
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )


    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )