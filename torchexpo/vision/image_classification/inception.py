import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def inception():
    """Inception Model"""
    model = torchvision.models.inception_v3(pretrained=True)
    obj = TorchExpo(model, "Inception", torch.rand(1, 3, 224, 224))
    return obj
