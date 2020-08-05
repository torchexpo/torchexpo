"""MNASNet Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class MNASNet(TorchExpo):
    """MNASNet Model"""

    name = "MNASNet"
    example = torch.rand(1, 3, 224, 224)

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.mnasnet1_0(pretrained=True)
        super().__init__(self.model, self.name, self.example)
