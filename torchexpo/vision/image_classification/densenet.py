import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def densenet():
    """DenseNet Model"""
    model = torchvision.models.densenet161(pretrained=True)
    obj = TorchExpo(model, "DenseNet", torch.rand(1, 3, 224, 224))
    return obj