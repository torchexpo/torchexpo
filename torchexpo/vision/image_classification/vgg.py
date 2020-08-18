import torchvision
from torchexpo.modules import ImageClassificationModule


def vgg11():
    """VGG11 Model pre-trained on ImageNet"""
    model = torchvision.models.vgg11(pretrained=True)
    obj = ImageClassificationModule(model, "VGG11", model_example="default")
    return obj

def vgg11_bn():
    """VGG11_BN Model pre-trained on ImageNet"""
    model = torchvision.models.vgg11_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG11_BN", model_example="default")
    return obj

def vgg13():
    """VGG13 Model pre-trained on ImageNet"""
    model = torchvision.models.vgg13(pretrained=True)
    obj = ImageClassificationModule(model, "VGG13", model_example="default")
    return obj

def vgg13_bn():
    """VGG13_BN Model pre-trained on ImageNet"""
    model = torchvision.models.vgg13_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG13_BN", model_example="default")
    return obj

def vgg16():
    """VGG16 Model pre-trained on ImageNet"""
    model = torchvision.models.vgg16(pretrained=True)
    obj = ImageClassificationModule(model, "VGG16", model_example="default")
    return obj

def vgg16_bn():
    """VGG16_BN Model pre-trained on ImageNet"""
    model = torchvision.models.vgg16_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG16_BN", model_example="default")
    return obj

def vgg19():
    """VGG19 Model pre-trained on ImageNet"""
    model = torchvision.models.vgg19(pretrained=True)
    obj = ImageClassificationModule(model, "VGG19", model_example="default")
    return obj

def vgg19_bn():
    """VGG19_BN Model pre-trained on ImageNet"""
    model = torchvision.models.vgg19_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG19_BN", model_example="default")
    return obj