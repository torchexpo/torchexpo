import torchexpo
from torchexpo.vision import image_classification


print(torchexpo.__version__)
model = image_classification.resnet18()
model.extract_torchscript()
model.extract_onnx()