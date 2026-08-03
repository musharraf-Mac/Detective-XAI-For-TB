"""
Main XAI pipeline controller.
"""

from src.explainability.save_results import create_result_directory


class XAIPipeline:

    def __init__(self, model):

        self.model = model
        self.output_dir = create_result_directory()


    def explain(self, image):

        """
        Complete explanation workflow.

        Model integration will be added later.
        """

        print(
            "Grad-CAM pipeline ready"
        )

        return None