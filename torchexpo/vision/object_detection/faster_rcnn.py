"""Faster R-CNN Model"""
import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


class FasterRCNN(TorchExpo):
    """Faster R-CNN Model"""

    name = "Faster R-CNN"
    example = torch.rand(1, 3, 300, 500)

    def __init__(self):
        """Initialize Model"""
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        super().__init__(self.model, self.name, self.example)
