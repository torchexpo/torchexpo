import torchvision
from torchexpo.modules import ImageClassificationModule


def squeezenet1_0():
    """SqueezeNet 1.0 Model pre-trained on ImageNet"""
    model = torchvision.models.squeezenet1_0(pretrained=True)
    obj = ImageClassificationModule(model, "SqueezeNet1_0", model_example="default")
    return obj

def squeezenet1_1():
    """SqueezeNet 1.1 Model pre-trained on ImageNet"""
    model = torchvision.models.squeezenet1_1(pretrained=True)
    obj = ImageClassificationModule(model, "SqueezeNet1_1", model_example="default")
    return obj