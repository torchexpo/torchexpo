import torchexpo
from torchexpo.vision.image_classification import ResNet18

print(torchexpo.__version__)
resnet18 = ResNet18()
resnet18.extract_torchscript()
