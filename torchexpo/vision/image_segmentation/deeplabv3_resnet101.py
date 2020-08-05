import abc
import torch
import torch.nn as nn
import torchvision
from torchexpo.core.torchexpo import TorchExpo


def deeplabv3_resnet101():
    """DeepLabV3-ResNet101 Model"""
    model = DeepLabV3ResNet101()
    obj = TorchExpo(model, "DeepLabV3-ResNet101", torch.rand(1, 3, 224, 224))
    return obj

class DeepLabV3ResNet101(nn.Module):
    """TorchExpo DeepLabV3-ResNet101 Scriptable Module"""
    def __init__(self):
        super(DeepLabV3ResNet101, self).__init__()
        self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    @abc.abstractmethod
    def forward(self, tensor):
        """Model Forward"""
        output = self.deeplab(tensor)['out']
        return output[0]
