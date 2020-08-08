import torchvision
from torchexpo.modules import ImageClassificationModule


def mobilenet_v2():
    """MobileNet V2 Model"""
    model = torchvision.models.mobilenet_v2(pretrained=True)
    obj = ImageClassificationModule(model, "MobileNet_V2")
    return obj