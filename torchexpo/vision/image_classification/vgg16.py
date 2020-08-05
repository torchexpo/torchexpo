"""VGG16 Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class VGG16(TorchExpo):
    """VGG16 Model"""

    name = "VGG16"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.vgg16(pretrained=True)
        super().__init__(self.model, self.name, self.example)
