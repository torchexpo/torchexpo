import torchvision
from torchexpo.modules import ImageClassificationModule


def shufflenet_v2_x0_5():
    """ShuffleNet V2 0.5x Model pre-trained on ImageNet"""
    model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
    obj = ImageClassificationModule(model, "ShuffleNet_v2_x0_5", model_example="default")
    return obj

def shufflenet_v2_x1_0():
    """ShuffleNet V2 1.0x Model pre-trained on ImageNet"""
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    obj = ImageClassificationModule(model, "ShuffleNet_v2_x1_0", model_example="default")
    return obj

# def shufflenet_v2_x1_5():
#     """ShuffleNet V2 1.5x Model pre-trained on ImageNet"""
#     model = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
#     obj = ImageClassificationModule(model, "ShuffleNet_v2_x1_5", model_example="default")
#     return obj

# def shufflenet_v2_x2_0():
#     """ShuffleNet V2 2.0x Model pre-trained on ImageNet"""
#     model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
#     obj = ImageClassificationModule(model, "ShuffleNet_v2_x2_0", model_example="default")
#     return obj