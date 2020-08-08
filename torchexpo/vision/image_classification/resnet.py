import torchvision
from torchexpo.modules import ImageClassificationModule


def resnet18():
    """ResNet18 Model"""
    model = torchvision.models.resnet18(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet18")
    return obj

def resnet34():
    """ResNet34 Model"""
    model = torchvision.models.resnet34(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet34")
    return obj

def resnet50():
    """ResNet50 Model"""
    model = torchvision.models.resnet50(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet50")
    return obj

def resnet101():
    """ResNet101 Model"""
    model = torchvision.models.resnet101(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet101")
    return obj

def resnet152():
    """ResNet152 Model"""
    model = torchvision.models.resnet152(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet152")
    return obj