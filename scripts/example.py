import torchexpo
from torchexpo.vision import image_classification


print(torchexpo.__version__)
resnet18 = image_classification.resnet18()
resnet18.extract_torchscript()
resnet18.extract_onnx()
