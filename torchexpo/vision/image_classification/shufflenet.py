import torch
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def shufflenet():
    """ShuffleNet Model"""
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    obj = TorchExpo(model, "ShuffleNet", torch.rand(1, 3, 224, 224))
    return obj
