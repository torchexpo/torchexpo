import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def resnet18():
    """ResNet18 Model"""
    model = torchvision.models.resnet18(pretrained=True)
    obj = TorchExpo(model, "ResNet18", torch.rand(1, 3, 224, 224))
    return obj