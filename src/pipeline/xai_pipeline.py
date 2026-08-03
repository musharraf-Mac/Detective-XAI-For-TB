"""
Main XAI pipeline controller.
"""


class XAIPipeline:

    def __init__(self, model):
        self.model = model


    def explain(self, image):

        """
        Runs Grad-CAM explanation.

        Model connection will be added
        after trained model integration.
        """

        pass