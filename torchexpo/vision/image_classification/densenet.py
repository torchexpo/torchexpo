import torchvision
from torchexpo.modules import ImageClassificationModule


def densenet121():
    """DenseNet-121 Model pre-trained on ImageNet"""
    model = torchvision.models.densenet121(pretrained=True)
    obj = ImageClassificationModule(model, "DenseNet-121", model_example="default")
    return obj

def densenet161():
    """DenseNet-161 Model pre-trained on ImageNet"""
    model = torchvision.models.densenet161(pretrained=True)
    obj = ImageClassificationModule(model, "DenseNet-161", model_example="default")
    return obj

def densenet169():
    """DenseNet-169 Model pre-trained on ImageNet"""
    model = torchvision.models.densenet169(pretrained=True)
    obj = ImageClassificationModule(model, "DenseNet-169", model_example="default")
    return obj

def densenet201():
    """DenseNet-201 Model pre-trained on ImageNet"""
    model = torchvision.models.densenet201(pretrained=True)
    obj = ImageClassificationModule(model, "DenseNet-201", model_example="default")
    return obj