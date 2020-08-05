import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def mnasnet():
    """MNASNet Model"""
    model = torchvision.models.mnasnet1_0(pretrained=True)
    obj = TorchExpo(model, "MNASNet", torch.rand(1, 3, 224, 224))
    return obj
