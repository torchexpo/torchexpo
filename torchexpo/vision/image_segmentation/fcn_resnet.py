import abc
import torch.nn as nn
import torchvision
from torchexpo.modules import ImageSegmentationModule


def fcn_resnet50():
    """FCN-ResNet50 Model pre-trained on COCO train2017"""
    model = FCNResNet(torchvision.models.segmentation.fcn_resnet50(pretrained=True))
    obj = ImageSegmentationModule(model, "FCN-ResNet50", model_example="default")
    return obj

def fcn_resnet101():
    """FCN-ResNet101 Model pre-trained on COCO train2017"""
    model = FCNResNet(torchvision.models.segmentation.fcn_resnet101(pretrained=True))
    obj = ImageSegmentationModule(model, "FCN-ResNet101", model_example="default")
    return obj

class FCNResNet(nn.Module):
    """TorchExpo FCN-ResNet Scriptable Module"""
    def __init__(self, model):
        super(FCNResNet, self).__init__()
        self.fcn = model

    @abc.abstractmethod
    def forward(self, tensor):
        """Model Forward"""
        output = self.fcn(tensor)['out']
        return output[0]