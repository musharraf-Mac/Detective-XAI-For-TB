"""
Grad-CAM implementation skeleton.

Used to generate visual explanations
for TB classification models.
"""


class GradCAM:

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.

        Args:
            model: trained deep learning model
            target_layer: layer used for visualization
        """

        self.model = model
        self.target_layer = target_layer


    def generate(self, image):
        """
        Generate heatmap.

        This will be implemented
        in the next commits.
        """

        pass