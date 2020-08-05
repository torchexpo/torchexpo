import torchexpo
from torchexpo.vision.image_classification import *


print(torchexpo.__version__)
resnet18 = ResNet18()
resnet18.extract_torchscript()
resnet18.extract_onnx()
