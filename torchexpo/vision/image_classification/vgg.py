import torchvision
from torchexpo.modules import ImageClassificationModule


def vgg11():
    """VGG11 Model"""
    model = torchvision.models.vgg11(pretrained=True)
    obj = ImageClassificationModule(model, "VGG11")
    return obj

def vgg11_bn():
    """VGG11_BN Model"""
    model = torchvision.models.vgg11_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG11_BN")
    return obj

def vgg13():
    """VGG13 Model"""
    model = torchvision.models.vgg13(pretrained=True)
    obj = ImageClassificationModule(model, "VGG13")
    return obj

def vgg13_bn():
    """VGG13_BN Model"""
    model = torchvision.models.vgg13_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG13_BN")
    return obj

def vgg16():
    """VGG16 Model"""
    model = torchvision.models.vgg16(pretrained=True)
    obj = ImageClassificationModule(model, "VGG16")
    return obj

def vgg16_bn():
    """VGG16_BN Model"""
    model = torchvision.models.vgg16_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG16_BN")
    return obj

def vgg19():
    """VGG19 Model"""
    model = torchvision.models.vgg19(pretrained=True)
    obj = ImageClassificationModule(model, "VGG19")
    return obj

def vgg19_bn():
    """VGG19_BN Model"""
    model = torchvision.models.vgg19_bn(pretrained=True)
    obj = ImageClassificationModule(model, "VGG19_BN")
    return obj