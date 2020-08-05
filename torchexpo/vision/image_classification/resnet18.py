"""ResNet18 Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class ResNet18(TorchExpo):
    """ResNet18 Model"""

    name = "ResNet18"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.resnet18(pretrained=True)
        super().__init__(self.model, self.name, self.example)
