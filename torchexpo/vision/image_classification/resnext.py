import torchvision
from torchexpo.modules import ImageClassificationModule


def resnext50_32x4d():
    """ResNext-50 32x4d Model"""
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    obj = ImageClassificationModule(model, "ResNext-50 32x4d")
    return obj

def resnext101_32x8d():
    """ResNext-101 32x8d Model"""
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    obj = ImageClassificationModule(model, "ResNext-101 32x8d")
    return obj