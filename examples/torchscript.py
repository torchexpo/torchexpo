"""torchexpo torchscript example"""
import torchexpo


resnet18 = torchexpo.vision.ResNet18()
resnet18.extract_torchscript()
