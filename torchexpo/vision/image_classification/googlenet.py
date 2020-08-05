import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def googlenet():
    """GoogleNet Model"""
    model = torchvision.models.googlenet(pretrained=True)
    obj = TorchExpo(model, "GoogLeNet", torch.rand(1, 3, 224, 224))
    return obj
