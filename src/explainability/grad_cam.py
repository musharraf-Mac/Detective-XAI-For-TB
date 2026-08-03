"""
Grad-CAM implementation for TB X-ray model explanation.
"""

import torch
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model, target_layer):
        
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()


    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()


        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()


        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )


    def generate(self, image, class_index=None):

        self.model.zero_grad()

        output = self.model(image)


        if class_index is None:
            class_index = torch.argmax(output)


        loss = output[:, class_index]

        loss.backward()


        gradients = self.gradients
        activations = self.activations


        weights = torch.mean(
            gradients,
            dim=(2,3),
            keepdim=True
        )


        heatmap = torch.sum(
            weights * activations,
            dim=1
        )


        heatmap = F.relu(heatmap)


        heatmap = heatmap.squeeze()

        heatmap /= torch.max(heatmap)


        return heatmap.cpu().numpy()