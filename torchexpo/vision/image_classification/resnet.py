import torchvision
from torchexpo.modules import ImageClassificationModule


def resnet18():
    """ResNet18 Model pre-trained on ImageNet"""
    model = torchvision.models.resnet18(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet18", model_example="default")
    return obj

def resnet34():
    """ResNet34 Model pre-trained on ImageNet"""
    model = torchvision.models.resnet34(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet34", model_example="default")
    return obj

def resnet50():
    """ResNet50 Model pre-trained on ImageNet"""
    model = torchvision.models.resnet50(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet50", model_example="default")
    return obj

def resnet101():
    """ResNet101 Model pre-trained on ImageNet"""
    model = torchvision.models.resnet101(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet101", model_example="default")
    return obj

def resnet152():
    """ResNet152 Model pre-trained on ImageNet"""
    model = torchvision.models.resnet152(pretrained=True)
    obj = ImageClassificationModule(model, "ResNet152", model_example="default")
    return obj