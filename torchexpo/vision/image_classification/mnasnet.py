import torchvision
from torchexpo.modules import ImageClassificationModule


def mnasnet0_5():
    """MNASNet 0.5 Model"""
    model = torchvision.models.mnasnet0_5(pretrained=True)
    obj = ImageClassificationModule(model, "MNASNet0_5", model_example="default")
    return obj

# def mnasnet0_75():
#     """MNASNet 0.75 Model"""
#     model = torchvision.models.mnasnet0_75(pretrained=True)
#     obj = ImageClassificationModule(model, "MNASNet0_75", model_example="default")
#     return obj

def mnasnet1_0():
    """MNASNet 1.0 Model"""
    model = torchvision.models.mnasnet1_0(pretrained=True)
    obj = ImageClassificationModule(model, "MNASNet1_0", model_example="default")
    return obj

# def mnasnet1_3():
#     """MNASNet 0.5 Model"""
#     model = torchvision.models.mnasnet1_3(pretrained=True)
#     obj = ImageClassificationModule(model, "MNASNet1_3", model_example="default")
#     return obj