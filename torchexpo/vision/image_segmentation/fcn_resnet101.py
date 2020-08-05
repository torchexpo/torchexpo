import abc
import torch
import torch.nn as nn
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def fcn_resnet101():
    """FCN-ResNet101 Model"""
    model = FCNResNet101()
    obj = TorchExpo(model, "FCN-ResNet101", torch.rand(1, 3, 224, 224))
    return obj

class FCNResNet101(nn.Module):
    """TorchExpo FCN-ResNet101 Scriptable Module"""
    def __init__(self):
        super(FCNResNet101, self).__init__()
        self.fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

    @abc.abstractmethod
    def forward(self, tensor):
        """Model Forward"""
        output = self.fcn(tensor)['out']
        return output[0]
