import torchvision
from torchexpo.modules import ImageClassificationModule


def mobilenet_v2():
    """MobileNet V2 Model pre-trained on ImageNet"""
    model = torchvision.models.mobilenet_v2(pretrained=True)
    obj = ImageClassificationModule(model, "MobileNet_V2", model_example="default")
    return obj