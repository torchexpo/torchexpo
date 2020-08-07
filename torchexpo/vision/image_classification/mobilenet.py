import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def mobilenet():
    """MobileNet Model"""
    model = torchvision.models.mobilenet_v2(pretrained=True)
    obj = TorchExpo(model, "MobileNet", torch.rand(1, 3, 224, 224))
    return obj