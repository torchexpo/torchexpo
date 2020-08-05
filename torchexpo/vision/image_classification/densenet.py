"""DenseNet Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class DenseNet(TorchExpo):
    """DenseNet Model"""

    name = "DenseNet"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.densenet161(pretrained=True)
        super().__init__(self.model, self.name, self.example)
