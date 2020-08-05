import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def alexnet():
    """AlexNet Model"""
    model = torchvision.models.alexnet(pretrained=True)
    obj = TorchExpo(model, "AlexNet", torch.rand(1, 3, 224, 224))
    return obj
