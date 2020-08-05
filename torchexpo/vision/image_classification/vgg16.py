import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def vgg16():
    """VGG16 Model"""
    model = torchvision.models.vgg16(pretrained=True)
    obj = TorchExpo(model, "VGG16", torch.rand(1, 3, 224, 224))
    return obj
