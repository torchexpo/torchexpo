import torchvision
from torchexpo.modules import ImageClassificationModule


def inceptionv3():
    """Inception v3 Model pre-trained on ImageNet"""
    model = torchvision.models.inception_v3(pretrained=True)
    obj = ImageClassificationModule(model, "InceptionV3", model_example="default")
    return obj