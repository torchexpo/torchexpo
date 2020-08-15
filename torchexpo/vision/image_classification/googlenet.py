import torchvision
from torchexpo.modules import ImageClassificationModule


def googlenet():
    """GoogLeNet (Inception v1) Model"""
    model = torchvision.models.googlenet(pretrained=True)
    obj = ImageClassificationModule(model, "GoogLeNet", model_example="default")
    return obj