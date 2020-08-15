import torchvision
from torchexpo.modules import ImageClassificationModule


def alexnet():
    """AlexNet Model"""
    model = torchvision.models.alexnet(pretrained=True)
    obj = ImageClassificationModule(model=model, model_name="AlexNet", model_example="default")
    return obj