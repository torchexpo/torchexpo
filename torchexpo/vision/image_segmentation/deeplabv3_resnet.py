import abc
import torch.nn as nn
import torchvision
from torchexpo.modules import ImageSegmentationModule


def deeplabv3_resnet50():
    """DeepLabV3-ResNet50 Model"""
    model = DeepLabV3ResNet(torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True))
    obj = ImageSegmentationModule(model, "DeepLabV3-ResNet50", model_example="default")
    return obj

def deeplabv3_resnet101():
    """DeepLabV3-ResNet101 Model"""
    model = DeepLabV3ResNet(torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True))
    obj = ImageSegmentationModule(model, "DeepLabV3-ResNet101", model_example="default")
    return obj

class DeepLabV3ResNet(nn.Module):
    """TorchExpo DeepLabV3-ResNet Scriptable Module"""
    def __init__(self, model):
        super(DeepLabV3ResNet, self).__init__()
        self.deeplab = model

    @abc.abstractmethod
    def forward(self, tensor):
        """Model Forward"""
        output = self.deeplab(tensor)['out']
        return output[0]