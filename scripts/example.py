import torchexpo


resnet18 = torchexpo.vision.image_classification.ResNet18()
resnet18.extract_torchscript()
