import torchvision
from torchexpo.modules import ImageClassificationModule


def wide_resnet50_2():
    """Wide ResNet-50-2 Model"""
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    obj = ImageClassificationModule(model, "Wide ResNet50 2", model_example="default")
    return obj

def wide_resnet101_2():
    """Wide ResNet-101-2 Model"""
    model = torchvision.models.wide_resnet101_2(pretrained=True)
    obj = ImageClassificationModule(model, "Wide ResNet101 2", model_example="default")
    return obj